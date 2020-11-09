"""Test the examples"""

from os import path
from sdf4sim.example import engine


def test_engine_demo():
    """analytic"""
    engine.demo(fig_file="engine.pdf")
    assert path.isfile("engine.pdf")


def test_engine_errors():
    """error plot"""
    engine.plot_quality_evaluation(fig_file="engerrs.pdf")
    assert path.isfile("engerrs.pdf")
