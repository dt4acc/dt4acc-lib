import datetime
from importlib import resources

import numpy as np
import pytest

from dt4acc.custom_facility.bessyii.provide_lattice import bessyii_pyat_lattice
from dt4acc_lib.model.output.track import ParticleState
from dt4acc_lib.pyat_simulator.accelerator_simulator import PyATAcceleratorSimulator
from dt4acc_lib.pyat_simulator.simulator_backend import SimulatorBackend


@pytest.fixture(scope="module")
def bessyii_simulator():
    filename = resources.files("dt4acc").joinpath(
        "custom_facility/bessyii/resources/storage_ring/input/bessy2_storage_ring_reflat.json"
    )

    acc = bessyii_pyat_lattice(filename=filename)
    acc = PyATAcceleratorSimulator(at_lattice=acc)
    return acc


@pytest.fixture(scope="module")
def bessyii_backend(bessyii_simulator):
    r = SimulatorBackend(
        name="Facility specific PYAT",
        acc=bessyii_simulator
    )
    return r

def test_tracking(bessyii_backend) -> None:
    p0 = ParticleState.from_sequence(
        [0] * 6
    )
    p1 = ParticleState.from_sequence([0] * 6)
    p1.delta = 1e-6
    p2 = ParticleState.from_sequence([0] * 6)
    p2.ct = 1e-8
    p3 = ParticleState.from_sequence([0] * 6)
    p3.x = 1e-8
    p4 = ParticleState.from_sequence([0] * 6)
    p4.dx = 1e-8
    p5 = ParticleState.from_sequence([0] * 6)
    p5.y = 1e-8
    p6 = ParticleState.from_sequence([0] * 6)
    p6.dy = 1e-8
    p7 = ParticleState.from_sequence([0] * 6)
    p7.x = -1e-8
    p8 = ParticleState.from_sequence([0] * 6)
    p8.dx = -1e-8
    p9 = ParticleState.from_sequence([0] * 6)
    p9.y = 1e-8


    elm_names = bessyii_backend.get_element_names()
    data_needed_at = [name for name in elm_names if name.startswith("BPM")]

    n_turns = 256
    start = datetime.datetime.now()
    data = bessyii_backend.compute_track(
        [p0, p1, p2, p3, p4, p5, p6, p7, p8], n_turns=n_turns,
        data_needed_at=data_needed_at
    )
    end = datetime.datetime.now()
    dt = end - start
    print(
        f"Computing {n_turns} turns required {dt.total_seconds()} seconds"
        f", thus {dt.total_seconds() / n_turns * 1000} ms per turn"
    )

    # Check that look up works
    for_element = data.for_element("BPMZ7D1R")
    pcol = for_element.for_turn(1)
    p = pcol.particle_per_id(0)

    # And that particles can be printed
    print(p)
    print(pcol)
    print(for_element)

    x_view = for_element.get_x()
    y_view = for_element.get_y()
    print(x_view)
    print(repr(x_view))
    print(y_view)
    print(repr(y_view))

    # just check that the call does not fail for now
    x_view.mean_per_turn()
    x_view.std_per_turn()

    pass
