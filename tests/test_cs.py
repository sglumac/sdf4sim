"""Tests for sdf4sim.cs"""

import logging
from fractions import Fraction
from os import path
from sympy import lcm, gcd  # pylint: disable=import-error
import matplotlib  # pylint: disable=import-error
import pytest
from sdf4sim import example, sdf, cs


logging.basicConfig(
    format='%(asctime)-15s %(message)s',
    filename='test_cs.log',
    filemode='a'
)


def test_control_cs_valid():
    """Check if conversion to cs gives a valid graph"""
    sdf_graph = cs.convert_to_sdf(example.control.gauss_seidel(1., 5., 1.))
    assert sdf.validate_graph(sdf_graph)


def test_control_loop_example():
    """A test whether the control loop example runs"""
    example.control.print_error_measurement()
    matplotlib.use('agg')
    example.control.visualise_error_measurement(fig_file='cs_compare.pdf')
    assert path.isfile('cs_compare.pdf')
    example.control.sil_comparison()


def test_repetition_vector():
    """Test to see check the cs repetition vector"""
    cosimulations = [
        example.control.gauss_seidel(1., 5., 1.),
        example.control.gauss_seidel(1., 5., 1., True),
        example.control.gauss_seidel(1., 5., 1., True, True),
        example.control.gauss_seidel(1., 5., 1., False, True),
        example.control.gauss_jacobi(1., 5., 1.),
        example.control.multi_rate(1., 5., 1.),
    ]
    for cosimulation in cosimulations:
        sdfg = cs.convert_to_sdf(cosimulation)
        schedule = sdf.calculate_schedule(sdfg)
        network, hs, _, _ = cosimulation
        _, connections = network
        repetitions = cs.repetition_vector(connections, hs)
        for agent in sdfg[0]:
            assert sum(agent == executed for executed in schedule) == repetitions[agent]


def test_defect_calculation_control():
    """Tests whether defect calculation give sane results"""
    csnet = example.control.cs_network()
    slaves, connections = csnet
    step_sizes = {name: Fraction(1, 2) for name in slaves}
    make_zoh: cs.ConverterConstructor = cs.Zoh
    rate_converters = {cs.Connection(src, dst): make_zoh for dst, src in connections.items()}
    initial_tokens = {sdf.Dst('PI', 'u'): [0.], sdf.Dst('PT2', 'u'): [0.]}
    cosim = csnet, step_sizes, rate_converters, initial_tokens
    defect = cs.evaluate(cosim, Fraction(20.))
    for val in defect.connection.values():
        assert val < float('inf')
    for val in defect.output.values():
        assert val < float('inf')


def _semiconnected_ramps(slope1=1., slope2=1.) -> cs.Network:
    """Example network used to test the defect calculation"""

    class _Ramp(cs.Simulator):
        """Used in tests to mock a simulator"""
        def __init__(self, slope: Fraction, step_size: Fraction):
            self._step_size = step_size
            self._x = Fraction(0)
            self._slope = slope

        @property
        def inputs(self):
            return {'u': sdf.InputPort(float, 1)}

        @property
        def outputs(self):
            return {'y': sdf.OutputPort(float, 1)}

        def calculate(self, input_tokens):
            self._x += self._slope * self._step_size
            return {'y': [float(self._x)]}

    def construct_ramp1(step_size):
        return _Ramp(slope1, step_size)

    def construct_ramp2(step_size):
        return _Ramp(slope2, step_size)

    slaves = {'Ramp1': construct_ramp1, 'Ramp2': construct_ramp2}

    connections = {
        cs.Dst('Ramp1', 'u'): cs.Src('Ramp2', 'y'),
        cs.Dst('Ramp2', 'u'): cs.Src('Ramp1', 'y'),
    }
    return slaves, connections


def ramp_cosimulation(slope1=2., slope2=3., step1=Fraction(5), step2=Fraction(7)):
    """Used for testing the defect calculation"""
    csnet = _semiconnected_ramps(slope1, slope2)
    _, connections = csnet
    step_sizes = {'Ramp1': step1, 'Ramp2': step2}
    make_zoh: cs.ConverterConstructor = cs.Zoh
    rate_converters = {cs.Connection(src, dst): make_zoh for dst, src in connections.items()}
    alpha = Fraction(int(lcm(step1.numerator, step2.numerator)),
                     int(gcd(step1.denominator, step2.denominator)))
    num1, num2 = tuple(map(int, [alpha / step for step in (step1, step2)]))
    initial_tokens = {
        sdf.Dst('Ramp1', 'u'): [(i - num1 + 1) * step2 * slope2 for i in range(num1)],
        sdf.Dst('Ramp2', 'u'): [(i - num2 + 1) * step1 * slope1 for i in range(num2)]
    }
    return csnet, step_sizes, rate_converters, initial_tokens


def test_defect_calculation():
    """Tests defect calculation on the cosimulation made of ramps"""
    slope1, slope2 = 2., 3.
    step1, step2 = Fraction(5), Fraction(7)
    cosim = ramp_cosimulation(slope1, slope2, step1, step2)
    t_end = Fraction(20)
    defect = cs.evaluate(cosim, t_end)

    alpha = Fraction(int(lcm(step1.numerator, step2.numerator)),
                     int(gcd(step1.denominator, step2.denominator)))
    num1, num2 = tuple(map(int, [alpha / step for step in (step1, step2)]))
    big = max(num1, num2) + 1
    small = min(num1, num2) - 1
    assert defect.connection['Ramp1', 'u'] > small * slope2 * step2
    assert defect.connection['Ramp1', 'u'] < big * slope2 * step2
    assert defect.connection['Ramp2', 'u'] > small * slope1 * step1
    assert defect.connection['Ramp2', 'u'] < big * slope1 * step1

    assert defect.output['Ramp1', 'y'] == pytest.approx(slope1 * step1)
    assert defect.output['Ramp2', 'y'] == pytest.approx(slope2 * step2)


def test_csw_control():
    """Tests to see if the image was created"""
    example.control.gauss_jacobi_csw_run('csw_control.pdf')
    assert path.isfile('csw_control.pdf')
