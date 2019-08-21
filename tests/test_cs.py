"""Tests for sdf4sim.cs"""

import logging

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
    example.control.visualise_error_measurement(fig_file='cs_compare.pdf')
    assert path.isfile('cs_compare.pdf')
    example.control.sil_comparison()
