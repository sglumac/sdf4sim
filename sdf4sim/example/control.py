"""An example showing a simple control loop"""

from os import path
from fractions import Fraction
import numpy as np  # pylint: disable=import-error
import matplotlib.pyplot as plt  # pylint: disable=import-error
from matplotlib import rc  # pylint: disable=import-error
from sdf4sim import cs, sdf


rc('text', usetex=True)


def controller_parameters(K, T1, Ts):
    """The controller parameters used"""
    TI = T1
    KR = TI / (2 * K * Ts)
    return KR, TI


def slaves(K, T1, Ts) -> cs.Slaves:
    """The FMUs used in the example"""
    cur_dir = path.dirname(path.abspath(__file__))
    pi_path = path.join(cur_dir, 'PI.fmu')
    pt2_path = path.join(cur_dir, 'PT2.fmu')
    KR, TI = controller_parameters(K, T1, Ts)
    pi = cs.prepare_slave('PI', pi_path, False)
    pi.fmu.enterInitializationMode()
    pivrs = [
        next(var.valueReference for var in pi.description.modelVariables
             if var.name == name)
        for name in ["KP", "KI"]
    ]
    pi.fmu.setReal(pivrs, [KR, KR / TI])
    pi.fmu.exitInitializationMode()
    pt2 = cs.prepare_slave('PT2', pt2_path, False)
    pt2.fmu.enterInitializationMode()
    pt2vrs = [
        next(var.valueReference for var in pt2.description.modelVariables
             if var.name == name)
        for name in ["K", "T1", "Ts"]
    ]
    pt2.fmu.setReal(pt2vrs, [K, T1, Ts])
    pt2.fmu.exitInitializationMode()
    return [pi, pt2]


def cs_network(K, T1, Ts):
    """The network is formed from slaves"""
    connections = {
        ('PI', 'u'): ('PT2', 'y'),
        ('PT2', 'u'): ('PI', 'y'),
    }
    return slaves(K, T1, Ts), connections


def rate_converters():
    """Two ZOH converters"""
    return {
        cs.Connection(cs.Src('PI', 'y'), cs.Dst('PT2', 'u')): cs.Zoh,
        cs.Connection(cs.Src('PT2', 'y'), cs.Dst('PI', 'u')): cs.Zoh,
    }


def analytic_solution(K, T1, Ts):
    """The analytic solution of the system"""
    KR, TI = controller_parameters(K, T1, Ts)

    omegan = np.sqrt(KR * K / (TI * Ts))
    zeta = TI * omegan / (2 * KR * K)

    omegad = omegan * np.sqrt(1 - zeta * zeta)
    phiy2 = np.arctan(np.sqrt(1 - zeta * zeta) / zeta)
    cy2 = 1 / np.sin(phiy2)
    phiy1 = np.arctan(
        (TI * (1 - KR) * omegad) / (TI * (1 - KR) * omegan * zeta - KR)
    )
    cy1 = (1 - KR) / np.sin(phiy1)

    @np.vectorize
    def y1(t):
        return 1 - cy1 * np.exp(-omegan * zeta * t) * np.sin(omegad * t + phiy1)

    @np.vectorize
    def y2(t):
        return 1 - cy2 * np.exp(-omegan * zeta * t) * np.sin(omegad * t + phiy2)

    return y1, y2


def gauss_seidel(K, T1, Ts, inverse=False, init=False) -> cs.Cosimulation:
    """The SDF representation of a co-simulation master"""
    h = Fraction(3, 4)
    y1, y2 = analytic_solution(K, T1, Ts)
    step_sizes = {'PI': h, 'PT2': h}
    tokens = {
        sdf.Dst('PT2', 'u'): [] if inverse else [y1(h if init else 0)],
        sdf.Dst('PI', 'u'): [y2(h if init else 0)] if inverse else [],
    }
    return cs_network(K, T1, Ts), step_sizes, rate_converters(), tokens


def gauss_jacobi(K, T1, Ts) -> cs.Cosimulation:
    """The SDF representation of a co-simulation master"""
    h = Fraction(1, 2)
    y1, y2 = analytic_solution(K, T1, Ts)
    step_sizes = {'PI': h, 'PT2': h}
    tokens = {
        sdf.Dst('PT2', 'u'): [y1(0.5)],
        sdf.Dst('PI', 'u'): [y2(0.5)],
    }
    return cs_network(K, T1, Ts), step_sizes, rate_converters(), tokens


def multi_rate(K, T1, Ts) -> cs.Cosimulation:
    """The SDF representation of a co-simulation master"""
    y1, _ = analytic_solution(K, T1, Ts)
    step_sizes = {'PI': Fraction(3, 8), 'PT2': Fraction(3, 4)}
    tokens = {
        sdf.Dst('PT2', 'u'): [y1(0)],
        sdf.Dst('PI', 'u'): [],
    }
    return cs_network(K, T1, Ts), step_sizes, rate_converters(), tokens


def _plot_cs_output(cosimulation, results, axs):
    signals = [('PI', 'y'), ('PT2', 'y')]
    for signal, ax in zip(signals, axs):
        instance, port = signal
        ts, vals = cs.get_signal_samples(cosimulation, results, instance, port)
        ax.stem(ts, vals, label=r'Gauss-Seidel 21 \{$y_{11}(0)$\}',
                markerfmt='ks', basefmt='C7--', linefmt='C7--')


def _plot_error_lines(cosimulation, results, y2, ax2):
    ts, vals = cs.get_signal_samples(cosimulation, results, 'PT2', 'y')
    for t, val in zip(ts, vals):
        ax2.plot([t, t], [val, y2(t)], 'r-')
    t, val = ts[-1], vals[-1]
    ax2.plot([t, t], [val, y2(t)], 'r-', label=r'$|\gamma_{21}[k_2]- y_{21}(k_2h_2)|$')


def _plot_analytic(axs, ys, end_time):
    anlbls = ['$y_{11}(t)$', '$y_{21}(t)$']
    for ax, y, anlbl in zip(axs, ys, anlbls):
        ts = np.linspace(0, end_time)
        ax.plot(ts, y(ts), 'k--', label=anlbl)
        ax.set_xlim([0, end_time])
        ax.legend()


def visualise_error_measurement(K=1., T1=5., Ts=1., end_time=20, fig_file=None):
    """Visualizes the error measurement"""
    fig, axs = plt.subplots(1, 2, sharex=True)
    fig.set_size_inches(10, 5)
    ys = analytic_solution(K, T1, Ts)
    cosimulation = gauss_seidel(K, T1, Ts)
    results = cs.execute(cosimulation, end_time)

    _plot_cs_output(cosimulation, results, axs)
    _plot_error_lines(cosimulation, results, ys[1], axs[1])
    _plot_analytic(axs, ys, end_time)

    fig.tight_layout()
    if not fig_file:
        plt.show()
    else:
        plt.savefig(fig_file)


def print_error_measurement(K=1., T1=5., Ts=1., end_time=20):
    """Runs the example"""

    experiments = [
        (gauss_seidel(K, T1, Ts), 'Gauss-Seidel 21 [y11(0)]'),
        (gauss_seidel(K, T1, Ts, inverse=True), 'Gauss-Seidel 12 [y21(0)]'),
        (gauss_seidel(K, T1, Ts, init=True), 'Gauss-Seidel 21 [y11(h)]'),
        (gauss_jacobi(K, T1, Ts), 'Gauss-Jacobi'),
        (multi_rate(K, T1, Ts), 'Multi-rate 211'),
    ]

    _, y2 = analytic_solution(K, T1, Ts)

    print('Mean absolute error of')
    for cosimulation, lbl in experiments:
        results = cs.execute(cosimulation, end_time)
        ts, vals = cs.get_signal_samples(cosimulation, results, 'PT2', 'y')
        y2s = np.array(y2(ts))
        mean_abs_err = np.sum(np.abs(y2s - vals)) / len(vals)
        print(f' - {lbl} is equal to {mean_abs_err:.4f}')


if __name__ == '__main__':
    print_error_measurement()
    visualise_error_measurement(fig_file=None)
