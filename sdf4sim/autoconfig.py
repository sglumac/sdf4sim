"""Module used for automatic configuration of co-simulation"""

from fractions import Fraction
from sdf4sim import cs, sdf


def _null_jacobi_initial_tokens(
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


def find_initial_tokens(
        csnet: cs.Network, step_sizes: cs.StepSizes, rate_converters: cs.RateConverters,
        tolerance: float
) -> cs.InitialTokens:
    """Find the initial tokens based on fixed point iteration"""
    _, connections = csnet
    tokens = _null_jacobi_initial_tokens(connections, step_sizes)

    tolerance_satisfied = False
    while not tolerance_satisfied:
        sdfg = cs.convert_to_sdf(
            (csnet, step_sizes, rate_converters, tokens)
        )
        results = sdf.sequential_run(sdfg, sdf.iterations_expired(1))
        next_tokens = _next_tokens(connections, step_sizes, results)
        initial_defect = max(
            max(abs(next_token - token)
                for next_token, token
                in zip(next_tokens[dst], tokens[dst]))
            for dst in tokens.keys()
        )
        tolerance_satisfied = initial_defect < tolerance
        tokens = next_tokens

    return tokens


def _step_reduction_factor(defect: float, tolerance: float) -> Fraction:
    """Calculate the reduction factor to achieve the desired tolerance"""
    factor = Fraction(1)
    while defect * factor > tolerance:
        factor /= 2
    return factor


def find_configuration(
        csnet: cs.Network, end_time: Fraction, tolerance: float, max_iter: int
) -> cs.Cosimulation:
    """A method for finding a working configuration for the given co-simulation network"""
    slaves, connections = csnet
    step_sizes: cs.StepSizes = {name: end_time / 10 for name in slaves.keys()}
    make_zoh: cs.ConverterConstructor = cs.Zoh
    rate_converters = {cs.Connection(src, dst): make_zoh for dst, src in connections.items()}

    num_iter = 0
    while True:
        initial_tokens = find_initial_tokens(csnet, step_sizes, rate_converters, tolerance)
        cosim = csnet, step_sizes, rate_converters, initial_tokens
        defect = cs.evaluate(cosim, end_time)
        simulator_defects = {
            name: max(
                max(value for port, value in defect.connection.items() if port.slave == name),
                max(value for port, value in defect.output.items() if port.slave == name),

            )
            for name in slaves
        }
        tolerance_satisfied = all(defect < tolerance for defect in simulator_defects.values())
        num_iter += 1
        if not tolerance_satisfied and num_iter < max_iter:
            step_sizes = {
                name: step_size * _step_reduction_factor(simulator_defects[name], tolerance)
                for name, step_size in step_sizes.items()
            }
        else:
            break

    return csnet, step_sizes, rate_converters, initial_tokens
