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
InitialTokens = Dict[sdf.Dst, List[Any]]
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

Slaves = List[Slave]
StepSizes = Dict[InstanceName, Fraction]
ConverterConstructor = Callable[[int, int], sdf.Agent]
RateConverters = Dict[Connection, ConverterConstructor]
Network = Tuple[Slaves, Connections]
Cosimulation = Tuple[Network, StepSizes, RateConverters, InitialTokens]


def _filter_mvs(mvs, causality):
    return {v.name: v.valueReference for v in mvs if v.causality == causality}


def prepare_slave(
        instance_name: InstanceName, path: str,
        default_init=True
) -> Slave:
    """A function to read an FMU and go through default initialization"""
    mdl_desc = fmpy.read_model_description(path)
    unzip_dir = fmpy.extract(path)
    fmu = FMU2Slave(
        instanceName=instance_name,
        guid=mdl_desc.guid,
        modelIdentifier=mdl_desc.coSimulation.modelIdentifier,
        unzipDirectory=unzip_dir
    )
    fmu.instantiate()
    fmu.setupExperiment(startTime=0.)
    if default_init:
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
        in_vrs, in_vals = zip(*[
            (self._inputs[u], tokens[0])
            for u, tokens in input_tokens.items()
        ])
        # limitation of the simulator to fmi2Reals
        self._slave.setReal(in_vrs, in_vals)
        self._slave.doStep(self._time, self._dt)
        self._time += self._dt
        out_vars = list(self._outputs)
        out_vrs = list(self._outputs[y] for y in out_vars)
        out_vals = [[val] for val in self._slave.getReal(out_vrs)]
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
            input_tokens['u'][tidx * self._consumption // self._production]
            for tidx in range(self._production)
        ]
        return {'y': tokens}


def _construct_rate_converter(h_src, h_dst, construct):
    dennum_src = h_src.denominator * h_dst.numerator
    dennum_dst = h_dst.denominator * h_src.numerator
    lcm = np.lcm(dennum_src, dennum_dst)
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
    for slave in network[0]:
        name = slave.fmu.instanceName
        agents[name] = Simulator(slave, step_sizes[name])
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
    slaves, _ = network
    sim1, sim2 = [slaves[i].fmu.instanceName for i in (0, 1)]
    converter = f'{sim1}_y_{sim2}_u'
    buffer = sdf.Dst(converter, 'u')
    step = step_sizes[sim1]

    # pylint: disable=unused-argument
    def terminate(sdf_graph: sdf.Graph, results: sdf.Results) -> bool:
        nonlocal end_time, buffer, step
        return end_time <= len(results[buffer]) * step

    return terminate
