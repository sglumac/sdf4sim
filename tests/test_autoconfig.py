"""Tests the automatic configuration functionality"""

import logging
from fractions import Fraction
from os import path
from sdf4sim.autoconfig import find_initial_tokens
from sdf4sim import cs, example


logging.basicConfig(
    format='%(asctime)-15s %(message)s',
    filename='test_autoconfig.log',
    filemode='a'
)


def test_autoconfig_example():
    """Runs the example and checks if it crashes"""
    example.control.automatic_configuration(tolerance=0.1, fig_file='autoconfig.pdf')
    assert path.isfile('autoconfig.pdf')


def test_initial_tokens():
    """Checks the procedure for finding the initial tokens"""
    csnet = example.control.cs_network()
    initial_step = Fraction(1, 5)
    slaves, connections = csnet
    step_sizes: cs.StepSizes = {
        name: (idx + 5) * initial_step for idx, name in enumerate(slaves.keys())
    }
    make_zoh: cs.ConverterConstructor = cs.Zoh
    rate_converters = {cs.Connection(src, dst): make_zoh for dst, src in connections.items()}
    initial_tokens = find_initial_tokens(csnet, step_sizes, rate_converters)
    rpv = cs.repetition_vector(connections, step_sizes)
    for (name, _), buffer in initial_tokens.items():
        if name in slaves:
            assert rpv[name] == len(buffer)


def test_twomass_autoconfig():
    """Another test on an example"""
    example.twomass.automatic_configuration(end_time=Fraction(20), fig_file='twomass.pdf')
    assert path.isfile('twomass.pdf')


def test_another_twomass_autoconfig():
    """Yet another test on an example"""
    example.twomass.three_tolerances_auto(fig_file='autos.pdf')
    assert path.isfile('autos.pdf')