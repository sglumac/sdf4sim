"""An example showing evaluation of an engine"""

from typing import NamedTuple, Callable, Dict, Any
from fractions import Fraction
from itertools import takewhile, count, product
import matplotlib.pyplot as plt  # pylint: disable=import-error
import numpy as np  # pylint: disable=import-error
from sdf4sim import cs, sdf
from sdf4sim.example.control import show_figure


InertiaParameters = NamedTuple('SideParameters', [
    ('inertia', float), ('damping', float), ('omega0', float),
])

EngineParameters = NamedTuple('EngineParameters', [
    ('w_alpha', float), ('w_omega', float)
])

AlphaParameters = NamedTuple('AlphaParameters', [
    ('source', Callable[[float], float])
])

EngineExperiment = NamedTuple('EngineExperiment', [
    ('inertia', InertiaParameters),
    ('engine', EngineParameters),
    ('alpha', AlphaParameters),
])


def impulse(t: float) -> float:
    """Simple 1 sec impulse"""
    return 1. if t < 1. else 0.


def generate_parameters(non_default: Dict[str, Dict[str, float]]) -> EngineExperiment:
    """Generate parameters of a two mass inertia"""
    parameters: Dict[str, Dict[str, Any]] = {
        'inertia': {
            'inertia': 10.0, 'damping': 10.0, 'omega0': 0.,
        },
        'engine': {
            'w_alpha': 1., 'w_omega': 1,
        },
        'alpha': {
            'source': impulse
        }
    }
    for simulator, params in non_default.items():
        for parameter, value in params.items():
            parameters[simulator][parameter] = value
    return EngineExperiment(
        inertia=InertiaParameters(**parameters['inertia']),
        engine=EngineParameters(**parameters['engine']),
        alpha=AlphaParameters(**parameters['alpha']),
    )


def first_order_response(x0, xEnd, lmbda, ts):
    """Exponential"""
    return x0 + (xEnd - x0) * (1 - np.exp(lmbda * ts))


class RotationalInertia(cs.Simulator):
    """Rotational inertia"""

    def __init__(self, params: InertiaParameters, step_size: Fraction):
        self._step_size = step_size
        self._omega = params.omega0
        self._J = params.inertia
        self._d = params.damping

    @property
    def inputs(self):
        return {'tau': sdf.InputPort(float, 1)}

    @property
    def outputs(self):
        return {'omega': sdf.OutputPort(float, 1)}

    def calculate(self, input_tokens):
        tau = input_tokens['tau'][0]
        omegaEnd = tau / self._d
        self._omega = first_order_response(
            self._omega, omegaEnd, -self._d / self._J, self._step_size)
        return {'omega': [self._omega]}


class Engine(cs.Simulator):
    """Engine represented as a linear engine map"""

    def __init__(self, parameters: EngineParameters, step_size: Fraction):
        self._w_alpha = parameters.w_alpha
        self._w_omega = parameters.w_omega

    @property
    def inputs(self):
        return {
            'omega': sdf.InputPort(float, 1),
            'alpha': sdf.InputPort(float, 1)
        }

    @property
    def outputs(self):
        return {'tau': sdf.OutputPort(float, 1)}

    def calculate(self, input_tokens):
        omega = input_tokens['omega'][0]
        alpha = input_tokens['alpha'][0]
        return {'tau': [self._w_alpha * alpha + self._w_omega * omega]}


class Alpha(cs.Simulator):
    """Alpha simulator"""

    def __init__(self, parameters: AlphaParameters, step_size: Fraction):
        self._step_size = step_size
        self._signal = parameters.source
        self._t = 0.

    @property
    def inputs(self):
        return dict()

    @property
    def outputs(self):
        return {'alpha': sdf.OutputPort(float, 1)}

    def calculate(self, input_tokens):
        self._t += self._step_size
        return {'alpha': [self._signal(self._t)]}


def cs_network(parameters: EngineExperiment) -> cs.Network:
    """A network with an engine and a rotational inertia"""
    def construct_engine(step_size: Fraction) -> sdf.Agent:
        return Engine(parameters.engine, step_size)

    def construct_inertia(step_size: Fraction) -> sdf.Agent:
        return RotationalInertia(parameters.inertia, step_size)

    def construct_alpha(step_size: Fraction) -> sdf.Agent:
        return Alpha(parameters.alpha, step_size)

    slaves: cs.SimulatorContructors = {
        'engine': construct_engine,
        'inertia': construct_inertia,
        'alpha': construct_alpha,
    }
    connections: cs.Connections = {
        cs.Dst('engine', 'omega'): cs.Src('inertia', 'omega'),
        cs.Dst('engine', 'alpha'): cs.Src('alpha', 'alpha'),
        cs.Dst('inertia', 'tau'): cs.Src('engine', 'tau'),
    }
    return slaves, connections


def rate_converters():
    """Two ZOH converters"""
    return {
        cs.Connection(cs.Src('inertia', 'omega'), cs.Dst('engine', 'omega')): cs.Zoh,
        cs.Connection(cs.Src('alpha', 'alpha'), cs.Dst('engine', 'alpha')): cs.Zoh,
        cs.Connection(cs.Src('engine', 'tau'), cs.Dst('inertia', 'tau')): cs.Zoh,
    }


def gauss_seidel(parameters: EngineExperiment, h=Fraction(1, 2), inverse=False) -> cs.Cosimulation:
    """The SDF representation of a co-simulation master"""
    step_sizes: cs.StepSizes = {'engine': h, 'inertia': h, 'alpha': h}
    tokens: cs.InitialTokens = {
        sdf.Dst('engine', 'omega'): [] if inverse else [0.],
        sdf.Dst('engine', 'alpha'): [],
        sdf.Dst('inertia', 'tau'): [0.] if inverse else [],
    }
    return cs_network(parameters), step_sizes, rate_converters(), tokens


def gauss_jacobi(parameters: EngineExperiment, h=Fraction(1, 2)) -> cs.Cosimulation:
    """The SDF representation of a co-simulation master"""
    step_sizes: cs.StepSizes = {'engine': h, 'inertia': h, 'alpha': h}
    tokens: cs.InitialTokens = {
        sdf.Dst('engine', 'omega'): [0.],
        sdf.Dst('engine', 'alpha'): [],
        sdf.Dst('inertia', 'tau'): [0.],
    }
    return cs_network(parameters), step_sizes, rate_converters(), tokens


def analytic_solution(parameters: EngineExperiment, h=Fraction(1, 2), end_time=Fraction(10, 1)):
    """analytic"""
    w_omega = parameters.engine.w_omega
    w_alpha = parameters.engine.w_alpha
    d = parameters.inertia.damping

    lmbda = - (d - w_omega) / parameters.inertia.inertia

    omega0 = parameters.inertia.omega0
    omegainf0 = w_alpha * 1. / (d - w_omega)

    omega1 = first_order_response(omega0, omegainf0, lmbda, 1.)
    omegainf1 = 0.

    ts = list(takewhile(lambda tp: tp <= end_time, count(0, h)))
    omegas = [
        first_order_response(omega0, omegainf0, lmbda, t) if t <= 1.
        else first_order_response(omega1, omegainf1, lmbda, t - 1.)
        for t in ts
    ]
    taus = [
        w_omega * omega + (w_alpha if t <= 1 else 0)
        for omega, t in zip(omegas, ts)
    ]
    return ts, omegas, taus


def get_cs_responses(parameters: EngineExperiment, h=Fraction(1, 10), end_time=Fraction(10, 1)):
    """Three co-simulations and the analytic response"""

    cosimulations = {
        'GS12': gauss_seidel(parameters, h, False),
        'GS21': gauss_seidel(parameters, h, True),
        'GJ': gauss_jacobi(parameters, h)
    }

    signals = [
        ('engine', 'tau'), ('engine', 'omega'),
        ('inertia', 'tau'), ('inertia', 'omega')
    ]
    responses = dict()

    for title, cosimulation in cosimulations.items():
        results = cs.execute(cosimulation, end_time)
        for simulator, port in signals:
            responses[title, simulator, port] = cs.get_signal_samples(
                cosimulation, results, simulator, port)

    return responses


def demo(fig_file=None):
    """Demo function"""
    default_parameters = generate_parameters(dict())
    responses = get_cs_responses(default_parameters)

    fig, axs = plt.subplots(2, 1, sharex=True)
    for lbl, (ts, vals) in responses.items():
        ax = axs[0] if lbl[2] == 'tau' else axs[1]
        ax.plot(ts, vals, label=lbl)

    for ax in axs:
        ax.legend()
    show_figure(fig, fig_file)


def _get_power_diff(title, responses, analytic):
    """helper"""
    _, omega, tau = analytic
    analyticp = np.array(omega[1:]) * np.array(tau[1:])
    csp = np.array(responses[title, 'inertia', 'omega'][1]) * \
        np.array(responses[title, 'engine', 'tau'][1])

    return analyticp - csp


def _get_power_residual(title, responses, h):
    """helper"""
    num = min(
        len(responses[title, 'inertia', 'omega'][1]),
        len(responses[title, 'inertia', 'tau'][1]),
        len(responses[title, 'engine', 'omega'][1]),
        len(responses[title, 'engine', 'tau'][1]),
    )
    pinertia = np.array(responses[title, 'inertia', 'omega'][1][:num]) * \
        np.array(responses[title, 'inertia', 'tau'][1][:num])
    pengine = np.array(responses[title, 'engine', 'omega'][1][:num]) * \
        np.array(responses[title, 'engine', 'tau'][1][:num])

    return float(np.cumsum(np.abs(pengine - pinertia))[-1] * h)


def _get_abs_power_residual(title, responses, h):
    """helper"""
    num = min(
        len(responses[title, 'inertia', 'omega'][1]),
        len(responses[title, 'inertia', 'tau'][1]),
        len(responses[title, 'engine', 'omega'][1]),
        len(responses[title, 'engine', 'tau'][1]),
    )
    pinertia = np.array(responses[title, 'inertia', 'omega'][1][:num]) * \
        np.array(responses[title, 'inertia', 'tau'][1][:num])
    pengine = np.array(responses[title, 'engine', 'omega'][1][:num]) * \
        np.array(responses[title, 'engine', 'tau'][1][:num])

    return float(np.abs(np.cumsum(pengine - pinertia)[-1]) * h)


def _energy_diff(title, responses, analytic, h):
    """helper"""
    return float(np.abs(np.cumsum(_get_power_diff(title, responses, analytic))[-1] * h))


def _abs_energy_diff(title, responses, analytic, h):
    """helper"""
    return float(np.cumsum(np.abs(_get_power_diff(title, responses, analytic)))[-1] * h)


def _get_abs_energy(title, analytic, h):
    """helper"""
    _, omega, tau = analytic
    analyticp = np.array(omega[1:]) * np.array(tau[1:])
    return float(np.cumsum(np.abs(analyticp)[-1] * h))


def _get_error_measures(parameters, hs, end_time, errors, signals):
    """helper function"""

    for h in hs:
        sol = analytic_solution(parameters, h, end_time)
        analytic = {
            ('inertia', 'omega'): sol[1],
            ('engine', 'tau'): sol[2],
        }
        responses = get_cs_responses(parameters, h)
        for title in ['GS12', 'GS21', 'GJ']:
            for simulator, port in signals:
                _, vals = responses[title, simulator, port]
                abserrs = np.abs(np.array(vals) - analytic[simulator, port][1:])
                errors[title, simulator, port, 'maxabs'].append(float(max(abserrs)))
                errors[title, simulator, port, 'intabs'].append(float(np.cumsum(abserrs)[-1] * h))
            errors[title, 'energyDiff'].append(_energy_diff(title, responses, sol, h))
            errors[title, 'absEnergyDiff'].append(_abs_energy_diff(title, responses, sol, h))
            errors[title, 'powerResidual'].append(_get_power_residual(title, responses, h))
            errors[title, 'absPowerResidual'].append(_get_abs_power_residual(title, responses, h))
            errors[title, 'absEnergy'].append(_get_abs_energy(title, sol, h))

    return errors


def plot_quality_evaluation(fig_file=None):
    """Show different quality evaluations"""
    parameters = generate_parameters(dict())
    end_time = Fraction(10, 1)
    hs = [Fraction(1, den) for den in 2 ** np.arange(1, 9)]

    signals = [('engine', 'tau'), ('inertia', 'omega')]
    errvals = ['maxabs', 'intabs']
    errors = {
        (title, simulator, port, k): []
        for title in ['GS12', 'GS21', 'GJ']
        for simulator, port in signals
        for k in errvals
    }
    for title in ['GS12', 'GS21', 'GJ']:
        errors[title, 'energyDiff'] = []
        errors[title, 'absEnergyDiff'] = []
        errors[title, 'powerResidual'] = []
        errors[title, 'absPowerResidual'] = []
        errors[title, 'absEnergy'] = []
    _get_error_measures(parameters, hs, end_time, errors, signals)

    fig, axs = plt.subplots(2, 7, sharex=True, sharey=True)
    dictaxs = {
        (simulator, port, errval): axs[i][j]
        for i, (simulator, port) in enumerate(signals)
        for j, errval in enumerate(errvals)
    }

    for title, (simulator, port) in product(['GS12', 'GS21', 'GJ'], signals):
        for errval in errvals:
            dictaxs[simulator, port, errval].plot(
                hs,
                errors[title, simulator, port, errval],
                label=f'{title}-{errval}'
            )
    for title in ['GS12', 'GS21', 'GJ']:
        axs[0][2].plot(hs, errors[title, 'energyDiff'], label=f'{title}-energyDiff')
        axs[0][3].plot(hs, errors[title, 'absEnergyDiff'], label=f'{title}-absEnergyDiff')
        axs[0][4].plot(hs, errors[title, 'powerResidual'], label=f'{title}-powerResidual')
        axs[0][5].plot(hs, errors[title, 'absPowerResidual'], label=f'{title}-absPowerResidual')
        axs[0][6].plot(hs, errors[title, 'absEnergy'], label=f'{title}-absEnergy')

    for ax in axs.flatten():
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.legend()
    show_figure(fig, fig_file)


if __name__ == '__main__':
    plot_quality_evaluation()
