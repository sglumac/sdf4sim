"""Tests for sdf4sim.sdf"""

import logging
from collections import deque
import pytest
from sdf4sim import sdf


logging.basicConfig(
    format='%(asctime)-15s %(message)s',
    filename='test_sdf.log',
    filemode='a'
)


def get_two_unit_graph(tokens12=13):
    """ Two units in a loop """
    agents = {
        'one': sdf.Gain(1),
        'two': sdf.Gain(1)
    }
    buffers = [
        sdf.Buffer(src=sdf.Src('one', 'y'),
                   dst=sdf.Dst('two', 'u'),
                   tokens=deque([tokens12])),
        sdf.Buffer(src=sdf.Src('two', 'y'),
                   dst=sdf.Dst('one', 'u'),
                   tokens=deque())
    ]
    return (agents, buffers)


def test_invalid_connection1():
    """ check for invalid tokens """
    sdf_graph = get_two_unit_graph()
    agents, buffers = sdf_graph
    buffers.append(sdf.Buffer(
        src=sdf.Src('nonexisting_actor', 'nonexisting_port'),
        dst=sdf.Dst('two', 'u'), tokens=deque([4])
    ))
    assert not sdf.validate_graph(sdf_graph)


def test_invalid_connection2():
    """Actors do not match the connections!"""
    sdf_graph = get_two_unit_graph()
    actors, buffers = sdf_graph
    buffers[0] = sdf.Buffer(
        src=sdf.Src('nonexisting_actor1', 'nonexisting_port1'),
        dst=sdf.Dst('nonexisting_actor2', 'nonexisting_port2'),
        tokens=buffers[0].tokens
    )
    assert not sdf.validate_graph(sdf_graph)


class GainRate(sdf.Gain):
    """A test agent which modifies its rate to cause an error"""
    @property
    def outputs(self):
        return {'y': sdf.OutputPort(float, 2)}


def test_inconsistent_graph():
    """ as the name says """
    sdf_graph = get_two_unit_graph()
    agents, _ = sdf_graph
    agents['one'] = GainRate(1)
    assert not sdf.validate_graph(sdf_graph)


def test_deadlock():
    """ as the name says """
    sdf_graph = get_two_unit_graph()
    _, buffers = sdf_graph
    buffers[0].tokens.popleft()
    assert not sdf.validate_graph(sdf_graph)


def test_valid_graph():
    """ as the name says """
    sdf_graph = get_two_unit_graph()
    assert sdf.validate_graph(sdf_graph)
    topology_matrix = sdf.calculate_topology_matrix(sdf_graph)
    assert sdf.is_consistent(topology_matrix)
    repetitions = sdf.repetition_vector(topology_matrix)
    assert repetitions[0] == 1
    assert repetitions[1] == 1
    sequence = sdf.calculate_schedule(sdf_graph)
    assert sequence == ['two', 'one']

    iterations = 10

    def terminate(sdf_graph: sdf.Graph, results: sdf.Results) -> bool:
        nonlocal iterations
        iterations -= 1
        return iterations <= 0

    results = sdf.sequential_run(sdf_graph, terminate)
    for tokens in results.tokens.values():
        assert len(tokens) > 5
        for token in tokens:
            assert token == pytest.approx(13)
