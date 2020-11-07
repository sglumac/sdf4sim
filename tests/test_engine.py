"""Test the examples"""

from os import path
from sdf4sim.example import engine


def test_engine_demo():
    """analytic"""
    engine.demo(fig_file="engine.pdf")
    assert path.isfile("engine.pdf")
