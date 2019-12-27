"""Test the examples"""

import numpy as np  # pylint: disable=import-error
import pytest
from sdf4sim.example.twomass import second_order_response

def test_oscillatory_response():
    """Test to see if SideOscillators are calculating correctly"""
    response = second_order_response(2, 12, 68)
    ts = np.linspace(0, 10)
    x_init, v_init, const = 7., 11., 13.
    x_part = 13. / 68.
    for t in ts:
        expected_x = (x_init - x_part) * np.exp(-3. * t) * np.cos(5. * t) \
            + (1. / 5.) * (v_init + (x_init - x_part) * 3.) * np.exp(-3. * t) * np.sin(5. * t) \
            + x_part
        expected_v = (x_init - x_part) * (-3.) * np.exp(-3. * t) * np.cos(5. * t) \
            + (x_init - x_part) * (-5.) * np.exp(-3. * t) * np.sin(5. * t) \
            + (-3. / 5.) * (v_init + (x_init - x_part) * 3.) * np.exp(-3. * t) * np.sin(5. * t) \
            + (v_init + (x_init - x_part) * 3.) * np.exp(-3. * t) * np.cos(5. * t)
        next_x, next_v = response(x_init, v_init, const, t)
        print(t)
        assert next_x == pytest.approx(expected_x)
        assert next_v == pytest.approx(expected_v)
