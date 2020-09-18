"""
The module used to generate a SDF graph as a model of computation
for a non-iterative co-simulation.
"""

from typing import Tuple, Dict, List, NamedTuple, Any, Callable, Optional, Union
from fractions import Fraction
from collections import deque
import numpy as np  # pylint: disable=import-error
import fmpy  # pylint: disable=import-error
from fmpy.fmi2 import FMU2Slave  # pylint: disable=import-error
from fmpy.model_description import ModelDescription  # pylint: disable=import-error
from . import sdf


Slave = NamedTuple('Slave', [
    ('description', ModelDescription), ('fmu', FMU2Slave)
])
InstanceName = str
PortLabel = str
Src = NamedTuple('Src', [
    ('slave', InstanceName), ('port', PortLabel)
])
Dst = NamedTuple('Dst', [
    ('slave', InstanceName), ('port', PortLabel)
])
Connection = NamedTuple('Connection', [
    ('src', Src), ('dst', Dst)
])
Connections = Dict[Dst, Src]
InitialTokens = Dict[sdf.Dst, List[Any]]

SimulatorContructor = Callable[[Fraction], sdf.Agent]
SimulatorContructors = Dict[InstanceName, SimulatorContructor]
StepSizes = Dict[InstanceName, Fraction]
ConverterConstructor = Callable[[int, int], sdf.Agent]
RateConverters = Dict[Connection, ConverterConstructor]
Network = Tuple[SimulatorContructors, Connections]
Cosimulation = Tuple[Network, StepSizes, RateConverters, InitialTokens]

TimeStamps = List[Fraction]
Values = List[Any]

OutputDefect = Dict[Src, float]
ConnectionDefect = Dict[Dst, float]
CommunicationDefect = NamedTuple('CommunicationDefect', [
    ('connection', ConnectionDefect), ('output', OutputDefect)
])
IntermediateValues = Dict[InstanceName, Dict[PortLabel, List[float]]]


def _filter_mvs(mvs, causality):
    return {v.name: v.valueReference for v in mvs if v.causality == causality}


used_slaves: Dict[str, Tuple[ModelDescription, str]] = dict()


def prepare_slave(
        instance_name: InstanceName, path: str
) -> Slave:
    """A function to read an FMU and go through default initialization"""
    if path in used_slaves:
        mdl_desc, unzip_dir = used_slaves[path]
    else:
        mdl_desc = fmpy.read_model_description(path)
        unzip_dir = fmpy.extract(path)
        used_slaves[path] = mdl_desc, unzip_dir
    fmu = FMU2Slave(
        instanceName=instance_name,
        guid=mdl_desc.guid,
        modelIdentifier=mdl_desc.coSimulation.modelIdentifier,
        unzipDirectory=unzip_dir
    )
    fmu.instantiate()
    fmu.setupExperiment(startTime=0.)
    fmu.enterInitializationMode()
    fmu.exitInitializationMode()
    return Slave(mdl_desc, fmu)


class Simulator(sdf.Agent):
    """A simulator wraps a co-simulation FMU to an SDF agent"""
    def __init__(self, slave: Slave, step_size: Fraction):
        self._inputs = _filter_mvs(
            slave.description.modelVariables, 'input')
        self._outputs = _filter_mvs(
            slave.description.modelVariables, 'output')
        self._time = 0.
        self._slave = slave.fmu
        self._dt = float(step_size)

    @property
    def inputs(self):
        return {u: sdf.InputPort(float, 1) for u in self._inputs}

    @property
    def outputs(self):
        return {y: sdf.OutputPort(float, 1) for y in self._outputs}

    def calculate(self, input_tokens):
        if input_tokens:
            in_vrs, in_vals = zip(*[
                (self._inputs[u], tokens[0])
                for u, tokens in input_tokens.items()
            ])
            # limitation of the simulator to fmi2Reals
            self._slave.setReal(in_vrs, in_vals)
        out_vars = list(self._outputs)
        out_vrs = list(self._outputs[y] for y in out_vars)

        self._slave.doStep(self._time, self._dt)
        out_vals = [[val] for val in self._slave.getReal(out_vrs)]

        self._time += self._dt
        return dict(zip(out_vars, out_vals))


class Zoh(sdf.Agent):
    """The rate converter which simply holds the last value"""
    def __init__(self, consumption: int, production: int):
        self._consumption = consumption
        self._production = production

    @property
    def inputs(self):
        return {'u': sdf.InputPort(float, self._consumption)}

    @property
    def outputs(self):
        return {'y': sdf.OutputPort(float, self._production)}

    def calculate(self, input_tokens):
        tokens = [
            input_tokens['u'][int(
                np.ceil((tidx + 1) * self._consumption / float(self._production)) - 1
            )]
            for tidx in range(self._production)
        ]
        return {'y': tokens}


def _construct_rate_converter(h_src, h_dst, construct):
    dennum_src = h_src.denominator * h_dst.numerator
    dennum_dst = h_dst.denominator * h_src.numerator
    lcm = np.lcm(dennum_src, dennum_dst)  # pylint: disable=no-member
    consumption = int(lcm / dennum_dst)
    production = int(lcm / dennum_src)
    return construct(consumption, production)


def convert_to_sdf(cosimulation: Cosimulation) -> sdf.Graph:
    """
    The function which converts a non-iteratove co-simulation to a SDF graph.
    """
    network, step_sizes, rate_converters, tokens = cosimulation
    agents: sdf.Agents = dict()
    buffers = list()
    # simulators
    for name, create_simulator in network[0].items():
        agents[name] = create_simulator(step_sizes[name])
    # rate converters and buffers
    for connection, construct in rate_converters.items():
        src, dst = connection
        agent_label = '_'.join([src.slave, src.port, dst.slave, dst.port])
        agents[agent_label] = _construct_rate_converter(
            step_sizes[src.slave], step_sizes[dst.slave], construct
        )
        # hardcoded ports of rate converters
        # assumed each converter has ports 'u' and 'y'
        buffers.append(sdf.Buffer(
            sdf.Src(src.slave, src.port),
            sdf.Dst(agent_label, 'u'),
            deque()
        ))
        buffers.append(sdf.Buffer(
            sdf.Src(agent_label, 'y'),
            sdf.Dst(dst.slave, dst.port),
            deque(tokens[sdf.Dst(dst.slave, dst.port)])
        ))
    return agents, buffers


def time_expired(cosimulation: Cosimulation, end_time: Fraction) -> sdf.Termination:
    """The termination based on the number of iterations"""

    network, step_sizes, _, _ = cosimulation
    _, connections = network
    connection = list(connections.items())[0]
    dst, src = connection
    converter = f'{src.slave}_{src.port}_{dst.slave}_{dst.port}'
    buffer = sdf.Dst(converter, 'u')
    step = step_sizes[src.slave]

    # pylint: disable=unused-argument
    def terminate(sdf_graph: sdf.Graph, results: sdf.Results) -> bool:
        nonlocal end_time, buffer, step
        return end_time <= len(results.tokens[buffer]) * step

    return terminate


def execute(cosimulation: Cosimulation, end_time: Fraction) -> sdf.Results:
    """Execution of specified cosimulation until end_time"""
    termination = time_expired(cosimulation, end_time)
    sdfg = convert_to_sdf(cosimulation)
    return sdf.sequential_run(sdfg, termination)


def get_signal_samples(
        cosimulation: Cosimulation, results: sdf.Results,
        simulator: InstanceName, output: PortLabel
) -> Tuple[TimeStamps, Values]:
    """Gets the time and value samples of the signal obtained at the specified port"""
    _, hs, _, _ = cosimulation
    h = hs[simulator]
    _, buffers = convert_to_sdf(cosimulation)
    buf_lbl = next(buffer.dst for buffer in buffers if buffer.src == sdf.Src(simulator, output))
    vals = results.tokens[buf_lbl]
    ts = [h * (i + 1) for i in range(len(vals))]
    return ts, vals


def repetition_vector(connections: Connections, hs: StepSizes) -> Dict[InstanceName, int]:
    """The expression for the repetion vector of the non-iterative co-simulation"""
    inv_hs = {simulator_name: 1 / h for simulator_name, h in hs.items()}
    for connection in connections.items():
        dst, src = connection
        agent_label = '_'.join([src.slave, src.port, dst.slave, dst.port])
        h_src = hs[src.slave]
        h_dst = hs[dst.slave]
        inv_hs[agent_label] = Fraction(
            h_src.denominator * h_dst.denominator,
            np.lcm(h_dst.denominator * h_src.numerator,  # pylint: disable=no-member
                   h_src.denominator * h_dst.numerator)
        )
    lcm_nums = np.lcm.reduce(  # pylint: disable=no-member
        [inv_h.denominator for inv_h in inv_hs.values()]
    )
    gcd_dens = np.gcd.reduce(  # pylint: disable=no-member
        [inv_h.numerator for inv_h in inv_hs.values()]
    )
    return {name: int(lcm_nums * inv_h / gcd_dens) for name, inv_h in inv_hs.items()}


Interval = NamedTuple('Interval', [
    ('start', Fraction), ('end', Fraction)
])
Derivatives = Union[float, Tuple[float, ...]]
Sample = NamedTuple('Sample', [
    ('interval', Interval), ('derivatives', Derivatives)
])
Samples = List[Sample]
OptionalSamples = List[Optional[Sample]]


def _to_taylor(ders: Derivatives):
    if isinstance(ders, float):
        ders = (ders,)
    return tuple(
        coef / np.math.factorial(k)
        for k, coef in enumerate(ders)
    )


def _polyval(poly, t):
    """Evaluate polynomial in the format given in the code"""
    return sum(coef * t ** k for k, coef in enumerate(poly))


def _polyder(poly):
    """Derivative of the polynomial"""
    return tuple((k + 1) * coef for k, coef in enumerate(poly[1:]))


def _max_abs_derdiff(
        der1: Derivatives, der2: Derivatives
) -> float:
    """Return the maximum absolute difference of two polynomials"""
    if isinstance(der1, float):
        if isinstance(der2, float):
            return abs(der1 - der2)

        return abs(der1 - der2[0])

    if isinstance(der2, float):
        return abs(der1[0] - der2)

    return abs(der1[0] - der2[0])


def _next_sample(from_it) -> Optional[Sample]:
    """Utility function to make mypy stop complaining"""
    return next(from_it, None)


def _max_abs_difference(x1s, hx1, x2s, hx2) -> float:
    """Used to calculate the connection defect"""
    idx1, idx2 = 0, 0
    max_abs_diff = 0.
    while idx1 < len(x1s) and idx2 < len(x2s):
        max_abs_diff = max(
            max_abs_diff,
            abs(x1s[idx1] - x2s[idx2])
        )
        if (idx1 + 1) * hx1 == (idx2 + 1) * hx2:
            idx1, idx2 = idx1 + 1, idx2 + 1
        elif (idx1 + 1) * hx1 < (idx2 + 1) * hx2:
            idx1 += 1
        else:
            idx2 += 1

    return max_abs_diff


def _calculate_output_defect(
        cosimulation: Cosimulation, intermediate_values: IntermediateValues, results: sdf.Results
) -> OutputDefect:
    """
    The function calculates the output defect under the assumption
    the co-simulation enables the output defect calculation.
    """
    _, connections = cosimulation[0]
    defect: OutputDefect = dict()
    for dst, src in connections.items():
        defect[src] = 0.
    for dst, src in connections.items():
        rate_u = sdf.Dst('_'.join([src.slave, src.port, dst.slave, dst.port]), 'u')
        end_vals = results.tokens[rate_u]
        half_vals = intermediate_values[src.slave][src.port]
        for half, end in zip(half_vals, end_vals):
            defect[src] = max(defect[src], abs((end - half) * 2))
    return defect


def _calculate_connection_defect(
        cosimulation: Cosimulation, results: sdf.Results
) -> ConnectionDefect:
    """The function calculates the connection defect of the co-simulation."""
    (_, connections), step_sizes, _, _ = cosimulation

    defect: ConnectionDefect = dict()
    for dst, src in connections.items():
        rate_u = sdf.Dst('_'.join([src.slave, src.port, dst.slave, dst.port]), 'u')
        out_step = step_sizes[src.slave]
        out_results = results.tokens[rate_u]
        in_step = step_sizes[dst.slave]
        in_results = results.tokens[sdf.Dst(dst.slave, dst.port)]
        defect[dst] = _max_abs_difference(
            in_results, in_step, out_results, out_step
        )

    return defect


class _OutputDefectMonitor(sdf.Agent):
    """Used for output defect calculation"""
    def __init__(
            self, create_simulator: SimulatorContructor,
            step_size: Fraction, intermediate_values
    ):
        self._simulator = create_simulator(step_size * Fraction(1, 2))
        self._intermediate_values = intermediate_values
        for port in self._simulator.outputs:
            self._intermediate_values[port] = []

    @property
    def inputs(self) -> Dict[str, sdf.InputPort]:
        """Input ports of the agent"""
        return self._simulator.inputs

    @property
    def outputs(self) -> Dict[str, sdf.OutputPort]:
        """Output ports of the agent"""
        return self._simulator.outputs

    def calculate(self, input_tokens):
        """The calculation method of the agent"""
        intermediate_tokens = self._simulator.calculate(input_tokens)
        for port, values in intermediate_tokens.items():
            self._intermediate_values[port].extend(values)
        return self._simulator.calculate(input_tokens)


def _sdf_output_monitor(
        cosimulation: Cosimulation,
        intermediate_values: IntermediateValues
) -> sdf.Graph:
    """Creates the SDF graph which monitors the output defect"""
    agents, buffers = convert_to_sdf(cosimulation)
    (simulator_constructors, _), step_sizes, _, _ = cosimulation
    for name, constructor in simulator_constructors.items():
        intermediate_values[name] = dict()
        agents[name] = _OutputDefectMonitor(
            constructor, step_sizes[name], intermediate_values[name]
        )
    return agents, buffers


def evaluate(cosimulation: Cosimulation, end_time: Fraction) -> CommunicationDefect:
    """Evaluates the co-simulation and returns the defects"""
    termination = time_expired(cosimulation, end_time)
    intermediate_values: IntermediateValues = dict()
    sdfg = _sdf_output_monitor(cosimulation, intermediate_values)
    results = sdf.sequential_run(sdfg, termination)
    output_defect = _calculate_output_defect(cosimulation, intermediate_values, results)
    connection_defect = _calculate_connection_defect(cosimulation, results)
    return CommunicationDefect(connection_defect, output_defect)
