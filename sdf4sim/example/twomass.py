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


def automatic_configuration(end_time=Fraction(10), tolerance=0.1, fig_file=None):
    """The demo function"""
    plt.clf()
    non_default = dict()
    csnet = cs_network(generate_parameters(non_default))
    cosimulation = autoconfig.find_configuration(
        csnet, end_time, tolerance
    )
    print(cosimulation[1])
    results = cs.execute(cosimulation, end_time)
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

    def doscillator(time, state):
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

    results = solve_ivp(doscillator, [0, end_time], state0, t_eval=ts)

    print(results)
    return results


def three_cosimulations_comparison(end_time=Fraction(50), fig_file=None):
    non_default = dict()
    parameters = generate_parameters(non_default)

    csnet = cs_network(parameters)
    slaves, connections = csnet
    make_zoh: cs.ConverterConstructor = cs.Zoh
    rate_converters = {cs.Connection(src, dst): make_zoh for dst, src in connections.items()}

    step_sizes_0 = {name: Fraction(1, 4) for name in slaves}
    initial_tokens_0 = autoconfig.find_initial_tokens(csnet, step_sizes_0, rate_converters)
    cosimulation_0 = csnet, step_sizes_0, rate_converters, initial_tokens_0
    results_0 = cs.execute(cosimulation_0, end_time)
    ts_F_0, vals_F_0 = cs.get_signal_samples(
        cosimulation_0, results_0, 'middle_oscillator', 'F_right'
    )
    ts_v_0, vals_v_0 = cs.get_signal_samples(cosimulation_0, results_0, 'right_oscillator', 'v')

    step_sizes_1 = {name: Fraction(1, 2) for name in slaves}
    initial_tokens_1 = initial_tokens_0
    cosimulation_1 = csnet, step_sizes_1, rate_converters, initial_tokens_1
    results_1 = cs.execute(cosimulation_1, end_time)
    ts_F_1, vals_F_1 = cs.get_signal_samples(
        cosimulation_1, results_1, 'middle_oscillator', 'F_right'
    )
    ts_v_1, vals_v_1 = cs.get_signal_samples(cosimulation_1, results_1, 'right_oscillator', 'v')

    step_sizes_2 = step_sizes_0
    initial_tokens_2 = autoconfig.null_jacobi_initial_tokens(connections, step_sizes_2)
    cosimulation_2 = csnet, step_sizes_2, rate_converters, initial_tokens_2
    results_2 = cs.execute(cosimulation_2, end_time)
    ts_F_2, vals_F_2 = cs.get_signal_samples(
        cosimulation_2, results_2, 'middle_oscillator', 'F_right'
    )
    ts_v_2, vals_v_2 = cs.get_signal_samples(cosimulation_2, results_2, 'right_oscillator', 'v')

    defects_0 = cs.evaluate(cosimulation_0, end_time)
    defects_1 = cs.evaluate(cosimulation_1, end_time)
    defects_2 = cs.evaluate(cosimulation_2, end_time)

    print(max(max(defects_0.output.values()), max(defects_0.connection.values())))
    print(max(max(defects_1.output.values()), max(defects_1.connection.values())))
    print(max(max(defects_2.output.values()), max(defects_2.connection.values())))

    results = monolithic_solution(parameters, end_time, Fraction(1, 4))
    vms = -results.y[1] - results.y[4]
    fs = parameters.middle.spring * results.y[2] + parameters.middle.damping * vms

    fig, axs = plt.subplots(2, 1, sharex=True)
    axf, axv = axs
    axf.set_title('f')
    axv.set_title('v')

    axf.plot(ts_F_0, vals_F_0, label='ref')
    axf.plot(ts_F_1, vals_F_1, label='h / 2')
    axf.plot(ts_F_2, vals_F_2, label='0 init')
    axf.plot(results.t, fs, 'r', label='monolithic')

    axv.plot(ts_v_0, vals_v_0, label='ref')
    axv.plot(ts_v_1, vals_v_1, label='h / 2')
    axv.plot(ts_v_2, vals_v_2, label='0 init')
    axv.plot(results.t, results.y[4], 'r', label='monolithic')

    axf.legend()
    axv.legend()

    show_figure(fig, fig_file)


if __name__ == '__main__':
    # automatic_configuration()
    three_cosimulations_comparison()
