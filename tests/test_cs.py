"""Tests for sdf4sim.cs"""

import logging
from fractions import Fraction
from os import path
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


def test_defect_calculation():
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
