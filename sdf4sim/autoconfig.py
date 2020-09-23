"""Module used for automatic configuration of co-simulation"""

from fractions import Fraction
from itertools import chain
import functools as fcn
from scipy.optimize import minimize  # pylint: disable=import-error
from sdf4sim import cs, sdf


def null_jacobi_initial_tokens(
        connections: cs.Connections, step_sizes: cs.StepSizes
) -> cs.InitialTokens:
    """The initial tokens for fully parallel execution"""
    rpv = cs.repetition_vector(connections, step_sizes)
    return {
        sdf.Dst(dst.slave, dst.port): [0.] * rpv[dst.slave]
        for dst in connections.keys()
    }


def _next_tokens(connections, step_sizes, results) -> cs.InitialTokens:
    """Calculates the next iteration of the tokens"""
    rpv = cs.repetition_vector(connections, step_sizes)
    dsts = (
        (src.slave,
         sdf.Dst('_'.join([src.slave, src.port, dst.slave, dst.port]), 'u'),
         sdf.Dst(dst.slave, dst.port))
        for dst, src in connections.items()
    )

    def resample_tokens(buffer, num_src, num_dst):
        return [buffer[(i * num_src) // num_dst] for i in range(num_dst)]

    next_tokens = {
        dst: resample_tokens(results.tokens[src], rpv[src_slave], rpv[dst.agent])
        for src_slave, src, dst in dsts
    }
    return next_tokens


def tokens_to_vector(tokens):
    """Prerequisite for minimization defect minimization"""
    return list(chain.from_iterable(tokens[port] for port in sorted(tokens.keys())))


def vector_to_tokens(model_tokens, vector):
    """Prerequisite for minimization defect minimization"""
    tokens = dict()
    idx = 0
    for port in sorted(model_tokens.keys()):
        num = len(model_tokens[port])
        tokens[port] = vector[idx:idx + num]
        idx += num
    return tokens


def calculate_simulator_defects(slaves, connections, defect: cs.CommunicationDefect):
    """Calculates max of output and connection defect for each simulator"""
    return {
        name: max(
            max(
                value for port, value in defect.connection.items()
                if connections[port].slave == name
            ),
            max(value for port, value in defect.connection.items() if port.slave == name),
            max(value for port, value in defect.output.items() if port.slave == name),
        )
        for name in slaves
    }


def token_evaluation(csnet, step_sizes, rate_converters, model_tokens, vector):
    """Evaluates tokens with two iterations"""
    tokens = vector_to_tokens(model_tokens, vector)
    cosim = csnet, step_sizes, rate_converters, tokens
    slaves, connections = csnet
    simulator_defects = calculate_simulator_defects(
        slaves, connections, cs.evaluate_until(cosim, sdf.iterations_expired(1))
    )
    return max(simulator_defects.values())


def find_initial_tokens(
        csnet: cs.Network, step_sizes: cs.StepSizes, rate_converters: cs.RateConverters
) -> cs.InitialTokens:
    """Find the initial tokens based on fixed point iteration"""
    slaves, connections = csnet
    tokens = null_jacobi_initial_tokens(connections, step_sizes)

    num_slaves = len(slaves)
    for _ in range(num_slaves * num_slaves):
        sdfg = cs.convert_to_sdf(
            (csnet, step_sizes, rate_converters, tokens)
        )
        results = sdf.sequential_run(sdfg, sdf.iterations_expired(1))
        tokens = _next_tokens(connections, step_sizes, results)

    minimization_criterion = fcn.partial(
        token_evaluation, csnet, step_sizes, rate_converters, tokens
    )
    res = minimize(
        minimization_criterion, tokens_to_vector(tokens),
        method='Nelder-Mead', options={'adaptive': True}
    )

    return vector_to_tokens(tokens, res.x)


def _step_reduction_factor(defect: float, tolerance: float) -> Fraction:
    """Calculate the reduction factor to achieve the desired tolerance"""
    factor = Fraction(1)
    while defect * factor > tolerance:
        factor /= 2
    return factor


def find_configuration(
        csnet: cs.Network, end_time: Fraction, tolerance: float, max_iter: int = 10
) -> cs.Cosimulation:
    """A method for finding a working configuration for the given co-simulation network"""
    slaves, connections = csnet
    step_sizes: cs.StepSizes = {name: end_time / 10 for name in slaves.keys()}
    make_zoh: cs.ConverterConstructor = cs.Zoh
    rate_converters = {cs.Connection(src, dst): make_zoh for dst, src in connections.items()}

    num_iter = 0
    while True:
        initial_tokens = find_initial_tokens(csnet, step_sizes, rate_converters)
        cosim = csnet, step_sizes, rate_converters, initial_tokens
        simulator_defects = calculate_simulator_defects(
            slaves, connections, cs.evaluate(cosim, end_time)
        )
        tolerance_satisfied = all(defect < tolerance for defect in simulator_defects.values())
        num_iter += 1
        if not tolerance_satisfied and num_iter < max_iter:
            step_sizes = {
                name: step_size * _step_reduction_factor(simulator_defects[name], tolerance)
                for name, step_size in step_sizes.items()
            }
        else:
            defect = cs.evaluate(cosim, end_time)
            assert max(max(defect.output.values()), max(defect.connection.values())) < tolerance
            break

    return csnet, step_sizes, rate_converters, initial_tokens
