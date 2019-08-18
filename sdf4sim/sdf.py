"""
The module is used for working with synchronous data flow graphs.
It implements a sequential execution algorithm.
"""

from typing import Tuple, Dict, List, Any, NamedTuple, Deque, Callable
from abc import ABC, abstractmethod
from functools import reduce
import logging
from sympy import Matrix, lcm, gcd, fraction  # pylint: disable=import-error
import numpy as np  # pylint: disable=import-error


InputPort = NamedTuple('InputPort', [
    ('type', type), ('consumption', int)
])
OutputPort = NamedTuple('OutputPort', [
    ('type', type), ('production', int)
])


class Agent(ABC):
    """All SDF agents should preferably derive from this class"""
    @property
    @abstractmethod
    def inputs(self) -> Dict[str, InputPort]:
        """Input ports of the agent"""

    @property
    @abstractmethod
    def outputs(self) -> Dict[str, OutputPort]:
        """Output ports of the agent"""

    @abstractmethod
    def calculate(self, input_tokens):
        """The calculation method of the agent"""


PortLabel = str
AgentLabel = str
Dst = NamedTuple('Dst', [
    ('agent', AgentLabel), ('port', PortLabel)
])
Src = NamedTuple('Src', [
    ('agent', AgentLabel), ('port', PortLabel)
])
Buffer = NamedTuple('Buffer', [
    ('src', Src), ('dst', Dst), ('tokens', Deque[Any])
])
Results = Dict[Dst, List[Any]]

Agents = Dict[str, Agent]
Buffers = List[Buffer]

Graph = Tuple[Agents, Buffers]


class Gain(Agent):
    """An SDF agents which multiplies its input"""
    def __init__(self, gain):
        self._gain = gain

    @property
    def inputs(self) -> Dict[str, InputPort]:
        return {'u': InputPort(float, 1)}

    @property
    def outputs(self) -> Dict[str, OutputPort]:
        return {'y': OutputPort(float, 1)}

    def calculate(self, input_tokens):
        return {'y': [input_tokens['u'][0] * self._gain]}


TopologyMatrix = Tuple[List[AgentLabel], List[List[int]]]
Schedule = List[AgentLabel]

Termination = Callable[[Graph, Results], bool]


def get_src_buffers(
        agent_name: AgentLabel,
        buffers: Buffers
) -> Buffers:
    """ List all sources for the actor """
    return [buffer for buffer in buffers
            if buffer.dst.agent == agent_name]


def get_dst_buffers(
        agent_name: AgentLabel,
        buffers: Buffers
) -> Buffers:
    """ List all destinations for the actor """
    return [buffer for buffer in buffers
            if buffer.src.agent == agent_name]


def can_fire(
        agent_name: AgentLabel,
        sdf_graph: Graph,
        modified_tokens
) -> bool:
    """ Check whether an agent in an SDF graph can fire. """
    agents, buffers = sdf_graph
    agent = agents[agent_name]
    return all(agent.inputs[dst.port].consumption <= modified_tokens[src, dst]
               for src, dst, _ in buffers
               if dst.agent == agent_name)


def _validate_connections(sdf_graph):
    agents, buffers = sdf_graph
    for src, dst, _ in buffers:
        if dst.agent not in agents:
            logging.error(f'''
            {src}-{dst} has no valid destination of a connection!
            ''')
            return False
        if src.agent not in agents:
            logging.error(f'''
            {src}-{dst} has no valid source of a connection!
            ''')
            return False
    return True


def validate_graph(sdf_graph: Graph) -> bool:
    """ It validates a synchronous data flow graph """

    validators = (
        _validate_connections, calculate_schedule
    )

    for validate in validators:
        if not validate(sdf_graph):
            return False
    return True


def calculate_topology_matrix(sdf_graph: Graph) -> TopologyMatrix:
    """ The topology matrix """
    agents, buffers = sdf_graph

    kernel_labels = list(agents.keys())
    agent_indices = {agent: aidx
                     for aidx, agent in enumerate(kernel_labels)}
    values = [[0 for _ in kernel_labels] for _ in buffers]
    for bidx, buffer in enumerate(buffers):
        src, dst, _ = buffer
        srcidx = agent_indices[src.agent]
        dstidx = agent_indices[dst.agent]
        src_port = agents[src.agent].outputs[src.port]
        dst_port = agents[dst.agent].inputs[dst.port]
        values[bidx][srcidx] = src_port.production
        values[bidx][dstidx] = -dst_port.consumption

    return (kernel_labels, values)


def is_consistent(
        topology_matrix: TopologyMatrix
) -> bool:
    """ Checks whether an SDF graph has consistent firing rates """
    _, matrix = topology_matrix
    num_agents = len(matrix[0])
    return np.linalg.matrix_rank(matrix) == num_agents - 1


def repetition_vector(
        topology_matrix: TopologyMatrix
) -> List[int]:
    """ Find a repetition vector for as SDF graph """
    _, top_matrix = topology_matrix
    matrix = Matrix(top_matrix)
    fractional = list(map(fraction, matrix.nullspace()[0]))
    denominators = [f[1] for f in fractional]
    lcm_null = reduce(lcm, denominators)
    integer = list(n * lcm_null / d for n, d in fractional)
    gcd_integer = reduce(gcd, integer)
    shortest = [x / gcd_integer for x in integer]
    return shortest


def calculate_schedule(sdf_graph: Graph) -> Schedule:
    """
    Function which calculates PASS.
    Returns an empty list of no PASS exists.
    """
    matrix = calculate_topology_matrix(sdf_graph)
    if not is_consistent(matrix):
        logging.error('''
        The SDF graph has inconsistent production and consumption rates!
        ''')
        return []
    repetitions: Dict[AgentLabel, int] = {
        matrix[0][idx]: val
        for idx, val in enumerate(repetition_vector(matrix))
    }

    agents, buffers = sdf_graph
    modified_tokens = {
        (buffer.src, buffer.dst): len(buffer.tokens)
        for buffer in buffers
    }

    schedule = []
    while True:
        agent_scheduled = False
        for agent_name in agents:
            if (can_fire(agent_name, sdf_graph, modified_tokens)
                    and repetitions[agent_name] > 0):
                schedule.append(agent_name)
                agent_scheduled = True
                repetitions[agent_name] -= 1
                agent = agents[agent_name]
                for buffer in get_src_buffers(agent_name, buffers):
                    consumption = agent.inputs[buffer.dst.port].consumption
                    modified_tokens[buffer.src, buffer.dst] -= consumption
                for buffer in get_dst_buffers(agent_name, buffers):
                    production = agent.outputs[buffer.src.port].production
                    modified_tokens[buffer.src, buffer.dst] += production
                break
        if not agent_scheduled:
            logging.error('''
            The given graph is in a deadlock! Please, modify the tokens to
            avoid the deadlock.
            ''')
            return []
        if all(repetitions[agent_name] == 0 for agent_name in agents):
            return schedule


def sequential_run(
        sdf_graph: Graph,
        terminate: Termination
) -> Results:
    """Runs the SDF graph sequentially"""
    agents, buffers = sdf_graph
    schedule = calculate_schedule(sdf_graph)
    results = {
        buffer.dst: list(buffer.tokens)
        for buffer in buffers
    }
    while not terminate(sdf_graph, results):
        for agent_name in schedule:
            agent = agents[agent_name]

            input_tokens = {
                buffer.dst.port: [
                    buffer.tokens.popleft() for _ in
                    range(agent.inputs[buffer.dst.port].consumption)
                ]
                for buffer in get_src_buffers(agent_name, buffers)
            }

            output_tokens = agent.calculate(input_tokens)

            for buffer in get_dst_buffers(agent_name, buffers):
                tokens = output_tokens[buffer.src.port]
                buffer.tokens.extend(tokens)
                results[buffer.dst].extend(tokens)

    return results


def iterations_expired(iterations: int) -> Termination:
    """The termination based on the number of iterations"""

    # pylint: disable=unused-argument
    def terminate(sdf_graph: Graph, results: Results) -> bool:
        nonlocal iterations
        iterations -= 1
        return iterations < 0

    return terminate
