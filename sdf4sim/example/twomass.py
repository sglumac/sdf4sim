"""An example showing the automatic configuration of a two mass oscillator"""

from typing import NamedTuple, Dict
from fractions import Fraction
from math import sqrt, exp, sin, cos
import matplotlib.pyplot as plt  # pylint: disable=import-error
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
        delta_v = v_right - v_left
        self._dx += self._step_size * delta_v
        force = self._d * delta_v + self._c * self._dx
        return {'F_left': [-force], 'F_right': [force]}


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


if __name__ == '__main__':
    automatic_configuration()
