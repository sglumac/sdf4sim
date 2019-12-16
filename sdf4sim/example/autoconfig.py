"""An example showing the automatic configuration of simple control loop"""

from fractions import Fraction
import matplotlib.pyplot as plt  # pylint: disable=import-error
from sdf4sim import autoconfig, cs
from sdf4sim.example import control


def example(max_iter=100, end_time=20., tolerance=1e-3, fig_file=None):
    """The demo function"""
    K, T1, Ts = (1., 5., 1.)
    end_time = 20.
    csnet = control.cs_network(K, T1, Ts)
    initial_step = Fraction(12, 5)
    cosimulation = autoconfig.find_configuration(
        csnet, end_time, initial_step, tolerance, max_iter
    )
    results = cs.execute(cosimulation, end_time)
    fig, axs = plt.subplots(1, 2, sharex=True)
    fig.set_size_inches(10, 5)
    control.plot_cs_output(cosimulation, results, axs)
    control.show_figure(fig, fig_file)


if __name__ == '__main__':
    example()
