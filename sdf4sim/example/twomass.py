"""An example showing the automatic configuration of a two mass oscillator"""

from typing import NamedTuple, Dict
from fractions import Fraction
from math import sqrt, exp, sin, cos
import matplotlib.pyplot as plt  # pylint: disable=import-error
from scipy.integrate import solve_ivp  # pylint: disable=import-error
import numpy as np
from sdf4sim import cs, sdf, autoconfig
from sdf4sim.example.control import show_figure


SideParameters = NamedTuple('SideParameters', [
    ('mass', float), ('spring', float), ('damping', float),
    ('initial_displacement', float), ('initial_velocity', float),
])

MiddleParameters = NamedTuple('MiddleParameters', [
    ('damping', float), ('spring', float), ('initial_displacement', float),
])

TwoMass = NamedTuple('TwoMass', [
    ('left', SideParameters), ('middle', MiddleParameters), ('right', SideParameters),
])


def generate_parameters(non_default: Dict[str, Dict[str, float]]) -> TwoMass:
    """Generate parameters of a two mass oscillator"""
    parameters: Dict[str, Dict[str, float]] = {
        'left': {
            'mass': 10.0, 'spring': 1.0, 'damping': 1.0,
            'initial_displacement': 0.1, 'initial_velocity': 0.1,
        },
        'right': {
            'mass': 10.0, 'spring': 1.0, 'damping': 2.0,
            'initial_displacement': 0.2, 'initial_velocity': 0.1,
        },
        'middle': {
            'damping': 2.0, 'spring': 1.0, 'initial_displacement': 1.0,
        }
    }
    for side, params in non_default.items():
        for parameter, value in params.items():
            parameters[side][parameter] = value
    return TwoMass(
        left=SideParameters(**parameters['left']),
        middle=MiddleParameters(**parameters['middle']),
        right=SideParameters(**parameters['right']),
    )


def second_order_response(a, b, c):
    """Used in SideOscillator class"""
    part1 = -b / (2 * a)

    def oscillatory(disp0, vel0, const, time):
        nonlocal part1
        x_part = const / c
        part2 = sqrt(4 * a * c - b * b) / (2 * a)
        gain2 = disp0 - x_part
        gain1 = (vel0 - gain2 * part1) / part2
        next_disp = gain1 * exp(part1 * time) * sin(part2 * time) \
            + gain2 * exp(part1 * time) * cos(part2 * time) \
            + x_part
        next_vel = gain1 * part1 * exp(part1 * time) * sin(part2 * time) \
            + gain1 * part2 * exp(part1 * time) * cos(part2 * time) \
            + gain2 * part1 * exp(part1 * time) * cos(part2 * time) \
            - gain2 * part2 * exp(part1 * time) * sin(part2 * time)
        return next_disp, next_vel

    def boundary(disp0, vel0, const, time):
        nonlocal part1
        x_part = const / c
        assert False, "Check this"
        gain1 = disp0 - x_part
        gain2 = vel0 - gain1 * part1
        next_disp = gain1 * exp(part1 * time) + gain2 * time * exp(part1 * time) + x_part
        next_vel = gain1 * part1 * exp(part1 * time) + gain2 * exp(part1 * time) \
            + gain2 * time * part1 * exp(part1 * time)
        return next_disp, next_vel

    def exponential(disp0, vel0, const, time):
        nonlocal part1
        x_part = const / c
        assert False, "check this"
        part2 = sqrt(b * b - 4 * a * c) / (2 * a)
        exp1 = part1 - part2
        exp2 = part1 + part2
        gain1 = (vel0 - disp0 * exp2 + x_part * exp2) / (exp1 - exp2)
        gain2 = disp0 - gain1 - x_part
        next_disp = gain1 * exp(exp1 * time) + gain2 * exp(exp2 * time) + x_part
        next_vel = gain1 * exp1 * exp(exp1 * time) + gain2 * exp2 * exp(exp2 * time)
        return next_disp, next_vel

    if b * b - 4 * a * c < 0.:
        return oscillatory
    if b * b - 4 * a * c == 0.:
        return boundary

    return exponential


class SideOscillator(cs.Simulator):
    """Simulator"""
    def __init__(self, params: SideParameters, step_size: Fraction):
        self._step_size = step_size
        self._dx = params.initial_displacement
        self._dv = params.initial_velocity
        self._mass = params.mass
        self._c = params.spring
        self._d = params.damping
        self._step_response = second_order_response(self._mass, self._d, self._c)

    @property
    def inputs(self):
        return {'F': sdf.InputPort(float, 1)}

    @property
    def outputs(self):
        return {'v': sdf.OutputPort(float, 1)}

    def calculate(self, input_tokens):
        force = input_tokens['F'][0]
        self._dx, self._dv = self._step_response(self._dx, self._dv, force, self._step_size)
        return {'v': [self._dv]}


class MiddleOscillator(cs.Simulator):
    """Simulator of a parallel damping and spring element"""
    def __init__(self, params: MiddleParameters, step_size: Fraction):
        self._d = params.damping
        self._c = params.spring
        self._dx = params.initial_displacement
        self._step_size = step_size

    @property
    def inputs(self):
        return {'v_left': sdf.InputPort(float, 1), 'v_right': sdf.InputPort(float, 1)}

    @property
    def outputs(self):
        return {'F_left': sdf.OutputPort(float, 1), 'F_right': sdf.OutputPort(float, 1)}

    def calculate(self, input_tokens):
        v_right = input_tokens['v_right'][0]
        v_left = input_tokens['v_left'][0]
        delta_v = -v_right - v_left
        self._dx += self._step_size * delta_v
        force = self._d * delta_v + self._c * self._dx
        return {'F_left': [force], 'F_right': [force]}


def cs_network(parameters: TwoMass):
    """The network of six mechanical translational elements"""
    def construct_left(step_size: Fraction):
        return SideOscillator(parameters.left, step_size)

    def construct_middle(step_size: Fraction):
        return MiddleOscillator(parameters.middle, step_size)

    def construct_right(step_size: Fraction):
        return SideOscillator(parameters.right, step_size)

    slaves = {
        'left_oscillator': construct_left,
        'middle_oscillator': construct_middle,
        'right_oscillator': construct_right,
    }
    connections = {
        cs.Dst('left_oscillator', 'F'): cs.Src('middle_oscillator', 'F_right'),
        cs.Dst('middle_oscillator', 'v_left'): cs.Src('left_oscillator', 'v'),
        cs.Dst('right_oscillator', 'F'): cs.Src('middle_oscillator', 'F_left'),
        cs.Dst('middle_oscillator', 'v_right'): cs.Src('right_oscillator', 'v'),
    }
    return slaves, connections


def _auto_results_only(csnet: cs.Network, end_time: Fraction, tolerance: float):
    """Helper function"""
    cosimulation = autoconfig.find_configuration(
        csnet, end_time, tolerance
    )
    print(cosimulation[1])
    return cosimulation, cs.execute(cosimulation, end_time)


def automatic_configuration(end_time=Fraction(20), tolerance=0.1, fig_file=None):
    """The demo function"""
    plt.clf()
    non_default = dict()
    csnet = cs_network(generate_parameters(non_default))
    cosimulation, results = _auto_results_only(csnet, end_time, tolerance)
    ts, vals = cs.get_signal_samples(cosimulation, results, 'right_oscillator', 'v')
    plt.stem(ts, vals)
    show_figure(plt.gcf(), fig_file)


def simple_execution(end_time=Fraction(100), fig_file=None):
    """The demo function"""
    plt.clf()
    non_default = dict()
    csnet = cs_network(generate_parameters(non_default))
    slaves, connections = csnet
    step_sizes = {name: Fraction(1, 4) for name in slaves}
    make_zoh: cs.ConverterConstructor = cs.Zoh
    rate_converters = {cs.Connection(src, dst): make_zoh for dst, src in connections.items()}
    initial_tokens = autoconfig.find_initial_tokens(csnet, step_sizes, rate_converters)
    cosimulation = csnet, step_sizes, rate_converters, initial_tokens
    results = cs.execute(cosimulation, end_time)
    print(f"v_right = {results.tokens['middle_oscillator', 'v_right'][-1]}")
    print(f"v_left = {results.tokens['middle_oscillator', 'v_left'][-1]}")
    print(f"F_right = {results.tokens['left_oscillator', 'F'][-1]}")
    print(f"F_left = {results.tokens['right_oscillator', 'F'][-1]}")
    ts, vals = cs.get_signal_samples(cosimulation, results, 'middle_oscillator', 'F_right')
    # ts, vals = cs.get_signal_samples(cosimulation, results, 'right_oscillator', 'v')
    plt.stem(ts, vals)
    show_figure(plt.gcf(), fig_file)


def monolithic_solution(parameters: TwoMass, end_time: Fraction, step_size: Fraction):
    """Helper function to find the monolithic solution"""

    def doscillator(_, state):
        nonlocal parameters
        xleft, vleft, xmiddle, xright, vright = state

        vmiddle = -vright - vleft
        fmiddle = parameters.middle.damping * vmiddle + parameters.middle.spring * xmiddle

        dxmiddle = vmiddle

        dxleft = vleft
        dvleft = (
            fmiddle
            - parameters.left.damping * vleft
            - parameters.left.spring * xleft
        ) / parameters.left.mass

        dxright = vright
        dvright = (
            fmiddle
            - parameters.right.damping * vright
            - parameters.right.spring * xright
        ) / parameters.right.mass

        return dxleft, dvleft, dxmiddle, dxright, dvright

    state0 = [
        parameters.left.initial_displacement,
        parameters.left.initial_velocity,
        parameters.middle.initial_displacement,
        parameters.right.initial_displacement,
        parameters.right.initial_velocity
    ]

    ts = step_size + np.arange(0, end_time, step_size)

    return solve_ivp(doscillator, [0, end_time], state0, t_eval=ts)


def _three_cosimulations(parameters: TwoMass):
    """Helper function"""
    csnet = cs_network(parameters)
    slaves, connections = csnet
    make_zoh: cs.ConverterConstructor = cs.Zoh
    rate_converters = {cs.Connection(src, dst): make_zoh for dst, src in connections.items()}

    cosimulations = list()

    step_sizes_1 = {name: Fraction(1, 4) for name in slaves}
    initial_tokens_1 = autoconfig.find_initial_tokens(csnet, step_sizes_1, rate_converters)
    cosimulations.append((csnet, step_sizes_1, rate_converters, initial_tokens_1))

    step_sizes_2 = {name: Fraction(1, 2) for name in slaves}
    initial_tokens_2 = autoconfig.find_initial_tokens(csnet, step_sizes_2, rate_converters)
    cosimulations.append((csnet, step_sizes_2, rate_converters, initial_tokens_2))

    step_sizes_3 = step_sizes_1
    initial_tokens_3 = autoconfig.null_jacobi_initial_tokens(connections, step_sizes_3)
    cosimulations.append((csnet, step_sizes_3, rate_converters, initial_tokens_3))

    return cosimulations


def _important_signals(cosimulation, end_time, samples, num):
    results = cs.execute(cosimulation, end_time)
    samples[f'tf{num}'], samples[f'vf{num}'] = cs.get_signal_samples(
        cosimulation, results, 'middle_oscillator', 'F_right'
    )
    samples[f'tv{num}'], samples[f'vv{num}'] = cs.get_signal_samples(
        cosimulation, results, 'right_oscillator', 'v')


def _print_defects(cosimulations, end_time):
    """Helper function"""
    defects = [cs.evaluate(cosimulation, end_time) for cosimulation in cosimulations]

    for num, defect in enumerate(defects):
        criterion = max(max(defect.output.values()), max(defect.connection.values()))
        print(f'defect(G{num + 1}) = {criterion}')


def _print_errors(samples, numcs, results, fms):
    """Helper function"""
    def calculate_errors(ts1, xs1, ts2, xs2):
        return np.abs(
            np.interp(np.array(ts1).astype(float), ts2.astype(float), xs2.astype(float)) - xs1
        )

    for num in range(numcs):
        verrs = calculate_errors(samples[f'tv{num}'], samples[f'vv{num}'], results.t, results.y[4])
        ferrs = calculate_errors(samples[f'tf{num}'], samples[f'vf{num}'], results.t, fms)

        print(f'error(G{num + 1}, F) = {max(ferrs)}')
        print(f'error(G{num + 1}, v) = {max(verrs)}')


def _plot_cosimulations(interval, samples, results, fms, plot):
    """Helper function"""
    fig, axs = plt.subplots(2, 1, sharex=True)
    axf, axv = axs
    axf.set_ylabel(r'Force [N]')
    axv.set_ylabel(r'Speed [m/s]')

    axv.set_xlabel('time [s]')

    axv.set_xlim(interval)

    for num, (labelv, labelf) in enumerate(plot['labels']):
        axf.plot(samples[f'tf{num}'], samples[f'vf{num}'], label=labelv)
        axv.plot(samples[f'tv{num}'], samples[f'vv{num}'], label=labelf)

    axf.plot(results.t, fms, 'r--', label=r'monolithic', alpha=0.8)
    axv.plot(results.t, results.y[4], 'r--', label=r'monolithic', alpha=0.8)

    axf.legend()
    axv.legend()

    show_figure(fig, plot['fig_file'])


def three_cosimulations_comparison(end_time=Fraction(20), fig_file=None):
    """Demo function"""
    non_default = dict()
    parameters = generate_parameters(non_default)

    cosimulations = _three_cosimulations(parameters)
    samples = dict()
    for num, cosimulation in enumerate(cosimulations):
        _important_signals(cosimulation, end_time, samples, num)

    results = monolithic_solution(parameters, end_time, Fraction(1, 4))
    vms = -results.y[1] - results.y[4]
    fms = parameters.middle.spring * results.y[2] + parameters.middle.damping * vms

    _print_defects(cosimulations, end_time)
    _print_errors(samples, 3, results, fms)

    plot = {
        'fig_file': fig_file,
        'labels': [
            tuple(r'$G_' + str(num) + r'$, $\widetilde{y}_{' + str(output) + r'1}$'
                  for output in range(2, 4))
            for num in range(1, 4)
        ]
    }
    _plot_cosimulations([0.5, float(end_time)], samples, results, fms, plot)


def three_tolerances_auto(end_time=Fraction(20), fig_file=None):
    """The demo function"""
    non_default = dict()
    parameters = generate_parameters(non_default)

    non_default = dict()
    csnet = cs_network(generate_parameters(non_default))

    cosimulations = list()
    tolerances = [0.9, 0.3, 0.1]
    for tol in tolerances:
        cosimulation, results = _auto_results_only(csnet, end_time, tol)
        cosimulations.append(cosimulation)

    samples = dict()
    for num, cosimulation in enumerate(cosimulations):
        _important_signals(cosimulation, end_time, samples, num)

    results = monolithic_solution(parameters, end_time, Fraction(1, 4))
    vms = -results.y[1] - results.y[4]
    fms = parameters.middle.spring * results.y[2] + parameters.middle.damping * vms

    _print_defects(cosimulations, end_time)
    _print_errors(samples, len(cosimulations), results, fms)
    plot = {
        'fig_file': fig_file,
        'labels': [
            tuple(
                r'$tol = ' + str(tol) + r'$, $\widetilde{y}_{' + str(output) + '1}$'
                for output in range(2, 4)
            )
            for tol in tolerances
        ]
    }
    _plot_cosimulations([0.5, float(end_time)], samples, results, fms, plot)


if __name__ == '__main__':
    three_tolerances_auto()
