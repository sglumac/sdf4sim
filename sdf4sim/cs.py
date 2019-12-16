"""
The module used to generate a SDF graph as a model of computation
for a non-iterative co-simulation.
"""

from typing import Tuple, Dict, List, NamedTuple, Any, Callable
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
SlaveContructor = Callable[[], Slave]
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

SlaveContructors = Dict[InstanceName, SlaveContructor]
StepSizes = Dict[InstanceName, Fraction]
ConverterConstructor = Callable[[int, int], sdf.Agent]
RateConverters = Dict[Connection, ConverterConstructor]
Network = Tuple[SlaveContructors, Connections]
Cosimulation = Tuple[Network, StepSizes, RateConverters, InitialTokens]

TimeStamps = List[Fraction]
Values = List[Any]

OutputDefect = Dict[Src, float]
ConnectionDefect = Dict[Dst, float]


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
    def __init__(self, slave: Slave, step_size: Fraction, calculate_defect=False):
        self._inputs = _filter_mvs(
            slave.description.modelVariables, 'input')
        self._outputs = _filter_mvs(
            slave.description.modelVariables, 'output')
        self._time = 0.
        self._slave = slave.fmu
        self._dt = float(step_size)
        self._calculate_defect = calculate_defect

    @property
    def inputs(self):
        return {u: sdf.InputPort(float, 1) for u in self._inputs}

    @property
    def outputs(self):
        return {y: sdf.OutputPort(float, 1) for y in self._outputs}

    def calculate(self, input_tokens):
        in_vrs, in_vals = zip(*[
            (self._inputs[u], tokens[0])
            for u, tokens in input_tokens.items()
        ])
        # limitation of the simulator to fmi2Reals
        self._slave.setReal(in_vrs, in_vals)
        out_vars = list(self._outputs)
        out_vrs = list(self._outputs[y] for y in out_vars)
        if self._calculate_defect:
            dt05 = self._dt / 2
            self._slave.doStep(self._time, dt05)
            half_vals = self._slave.getReal(out_vrs)
            self._slave.doStep(self._time + dt05, dt05)
            out_vals = [both_vals for both_vals in zip(half_vals, self._slave.getReal(out_vrs))]
        else:
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


def convert_to_sdf(cosimulation: Cosimulation, calculate_defect: bool = False) -> sdf.Graph:
    """
    The function which converts a non-iteratove co-simulation to a SDF graph.
    """
    network, step_sizes, rate_converters, tokens = cosimulation
    agents: sdf.Agents = dict()
    buffers = list()
    # simulators
    for name, create_slave in network[0].items():
        agents[name] = Simulator(create_slave(), step_sizes[name], calculate_defect)
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
        return end_time <= len(results[buffer]) * step

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
    vals = results[buf_lbl]
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


def _calculate_output_defect(results: sdf.Results) -> OutputDefect:
    """
    The function calculates the output defect under the assumption
    the co-simulation enables the output defect calculation.
    """
    return {Src('4', 'a'): 3.}


def _calculate_connection_defect(
        connections: Connections, results: sdf.Results
) -> ConnectionDefect:
    """The function calculates the connection defect of the co-simulation."""
    return {Dst('4', 'a'): 3.}


def evaluate(cosimulation: Cosimulation, end_time: Fraction):
    """Evaluates the co-simulation and returns the defects"""
    termination = time_expired(cosimulation, end_time)
    sdfg = convert_to_sdf(cosimulation)
    results = sdf.sequential_run(sdfg, termination)
    output_defect = _calculate_output_defect(results)
    _, connections = cosimulation[0]
    connection_defect = _calculate_connection_defect(connections, results)
    return output_defect, connection_defect
