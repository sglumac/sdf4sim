"""Tests for sdf4sim.cs"""

import logging
from fractions import Fraction
from os import path
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
    import matplotlib
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
    csnet = example.control.cs_network()
    slaves, connections = csnet
    step_sizes = {name: Fraction(1, 2) for name in slaves.keys()}
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
        def __init__(self, slope: float, step_size: Fraction):
            self._step_size = step_size
            self._x = 0.
            self._slope = slope

        @property
        def inputs(self):
            return {'u': sdf.InputPort(float, 1)}

        @property
        def outputs(self):
            return {'y': sdf.OutputPort(float, 1)}

        def calculate(self, input_tokens):
            self._x += self._slope * self._step_size
            return {'y': [self._x]}

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


def test_defect_calculation():
    csnet = _semiconnected_ramps()
    slaves, connections = csnet
    step_sizes = {name: Fraction(1) for name in slaves.keys()}
    make_zoh: cs.ConverterConstructor = cs.Zoh
    rate_converters = {cs.Connection(src, dst): make_zoh for dst, src in connections.items()}
    initial_tokens = {sdf.Dst('Ramp1', 'u'): [0.], sdf.Dst('Ramp2', 'u'): [0.]}
    cosim = csnet, step_sizes, rate_converters, initial_tokens
    defect = cs.evaluate(cosim, Fraction(2.))

    assert defect.connection['Ramp1', 'u'] == pytest.approx(1.)
    assert defect.connection['Ramp2', 'u'] == pytest.approx(1.)

    assert defect.output['Ramp1', 'y'] == pytest.approx(1.)
    assert defect.output['Ramp2', 'y'] == pytest.approx(1.)
