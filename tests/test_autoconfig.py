"""Tests the automatic configuration functionality"""

import logging
from fractions import Fraction
from os import path
from sdf4sim.autoconfig import find_initial_tokens
from sdf4sim.example.autoconfig import example
from sdf4sim.example.control import cs_network
from sdf4sim import cs


logging.basicConfig(
    format='%(asctime)-15s %(message)s',
    filename='test_autoconfig.log',
    filemode='a'
)


def test_autoconfig_example():
    """Runs the example and checks if it crashes"""
    example(max_iter=3, fig_file='autoconfig.pdf')
    assert path.isfile('autoconfig.pdf')



def test_initial_tokens():
    """Checks the procedure for finding the initial tokens"""
    K, T1, Ts = (1., 5., 1.)
    csnet = cs_network(K, T1, Ts)
    initial_step = Fraction(1, 5)
    slaves, connections = csnet
    step_sizes: cs.StepSizes = {
        name: (idx + 5) * initial_step for idx, name in enumerate(slaves.keys())
    }
    make_zoh: cs.ConverterConstructor = cs.Zoh
    rate_converters = {cs.Connection(src, dst): make_zoh for dst, src in connections.items()}
    initial_tokens = find_initial_tokens(csnet, step_sizes, rate_converters, 1e-3)
    rpv = cs.repetition_vector(connections, step_sizes)
    for (name, port), buffer in initial_tokens.items():
        if name in slaves:
            assert rpv[name] == len(buffer)
