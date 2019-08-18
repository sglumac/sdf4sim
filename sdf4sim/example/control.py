"""An example showing a simple control loop"""

from fractions import Fraction
import numpy as np  # pylint: disable=import-error
import matplotlib.pyplot as plt  # pylint: disable=import-error
from sdf4sim import cs, sdf


def controller_parameters(K, T1, Ts):
    """The controller parameters used"""
    TI = T1
    KR = TI / (2 * K * Ts)
    return KR, TI


def slaves(K, T1, Ts) -> cs.Slaves:
    """The FMUs used in the example"""
    KR, TI = controller_parameters(K, T1, Ts)
    pi = cs.prepare_slave('PI', r'C:\BenchmarkFMUs\PI.fmu', False)
    pi.fmu.enterInitializationMode()
    pivrs = [
        next(var.valueReference for var in pi.description.modelVariables
             if var.name == name)
        for name in ["KP", "KI"]
    ]
    pi.fmu.setReal(pivrs, [KR, KR / TI])
    pi.fmu.exitInitializationMode()
    pt2 = cs.prepare_slave('PT2', r'C:\BenchmarkFMUs\PT2.fmu', False)
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


def gauss_seidel(K, T1, Ts) -> cs.Cosimulation:
    """The SDF representation of a co-simulation master"""
    h = Fraction(3, 4)
    y1, y2 = analytic_solution(K, T1, Ts)
    step_sizes = {'PI': h, 'PT2': h}
    tokens = {
        sdf.Dst('PT2', 'u'): [y1(0)],
        sdf.Dst('PI', 'u'): [],
    }
    return cs_network(K, T1, Ts), step_sizes, rate_converters(), tokens


def gauss_jacobi(K, T1, Ts) -> cs.Cosimulation:
    """The SDF representation of a co-simulation master"""
    h = Fraction(1, 2)
    y1, y2 = analytic_solution(K, T1, Ts)
    step_sizes = {'PI': h, 'PT2': h}
    tokens = {
        sdf.Dst('PT2', 'u'): [y1(0)],
        sdf.Dst('PI', 'u'): [y2(0)],
    }
    return cs_network(K, T1, Ts), step_sizes, rate_converters(), tokens


def multi_rate(K, T1, Ts) -> cs.Cosimulation:
    """The SDF representation of a co-simulation master"""
    y1, y2 = analytic_solution(K, T1, Ts)
    step_sizes = {'PI': Fraction(1, 2), 'PT2': Fraction(1)}
    tokens = {
        sdf.Dst('PT2', 'u'): [y1(0)],
        sdf.Dst('PI', 'u'): [],
    }
    return cs_network(K, T1, Ts), step_sizes, rate_converters(), tokens


def run(K=1., T1=5., Ts=1., fig_file='cs_compare.png'):
    """Runs the example"""
    # end_time = 15 seconds
    ys = analytic_solution(K, T1, Ts)
    ts = np.linspace(0, 15)
    _, axs = plt.subplots(2, 1)
    for y, ax in zip(ys, axs):
        ax.plot(ts, y(ts), 'r--', label='analytic')

    simulators = ['PI', 'PT2']
    buffers = [sdf.Dst('PI_y_PT2_u', 'u'), sdf.Dst('PT2_y_PI_u', 'u')]
    fmts = ['ko', 'mo', 'go']
    lbls = ['Gauss-Seidel', 'Gauss-Jacobi', 'Multi-rate']
    cosimulations = [
        gauss_seidel(K, T1, Ts),
        gauss_jacobi(K, T1, Ts),
        multi_rate(K, T1, Ts),
    ]
    for cosimulation, fmt, lbl in zip(cosimulations, fmts, lbls):
        sdfg = cs.convert_to_sdf(cosimulation)
        hs = [cosimulation[1][sim] for sim in simulators]
        termination = cs.time_expired(cosimulation, 15)
        results = sdf.sequential_run(sdfg, termination)
        for buffer, h, ax in zip(buffers, hs, axs):
            ts = np.arange(h, 15 + 0.1 * h, h)
            ax.plot(ts, results[buffer], fmt, markerfacecolor='None', label=lbl)
    axs[0].legend()
    plt.savefig(fig_file)
