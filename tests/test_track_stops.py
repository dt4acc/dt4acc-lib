"""Turn by turn tracking: the beam ends at a stop (dt4acc#75)"""
import asyncio

import at
import numpy as np
import pytest

from dt4acc_lib.model.output.track import ParticleState
from dt4acc_lib.pyat_simulator.accelerator_simulator import PyATAcceleratorSimulator
from dt4acc_lib.pyat_simulator.simulator_backend import SimulatorBackend, first_element_reached

BPMS = [f"BPM{i}" for i in range(6)]
P0 = [ParticleState(1e-3, 0.0, -5e-4, 0.0, 0.0, 0.0), ParticleState(2e-3, 0.0, 5e-4, 0.0, 0.0, 0.0)]


def make_backend() -> SimulatorBackend:
    cells = [at.Marker("START")]
    for i in range(6):
        cells += [at.Monitor(f"BPM{i}"), at.Quadrupole(f"QF{i}", 0.3, 1.0), at.Drift("D", 0.8),
                  at.Dipole(f"B{i}", 1.0, 2 * np.pi / 6), at.Quadrupole(f"QD{i}", 0.3, -1.0), at.Drift("D", 0.8)]
    lattice = at.Lattice(cells, name="ring", energy=1.5e9, periodicity=1)
    lattice.disable_6d()
    return SimulatorBackend(acc=PyATAcceleratorSimulator(at_lattice=lattice), name="ring")


@pytest.fixture
def backend():
    return make_backend()


def data(turns):
    """{uid: [coordinate, particle, turn]}"""
    return {e.uid: e.track_data for e in turns.per_element}


@pytest.mark.parametrize("start_at, stop_at, seen", [
    (None, "QF2", ["BPM0", "BPM1", "BPM2"]),
    (None, "BPM2", ["BPM0", "BPM1", "BPM2"]),             # seen at the stop itself
    ("QF3", "QF2", ["BPM4", "BPM5", "BPM0", "BPM1", "BPM2"]),  # around the lattice's end
    ("QF3", "QF3", []),                                    # at the start
    (None, "START", []),
    (None, "QD5", BPMS),
])
def test_beam_ends_at_the_stop(backend, start_at, stop_at, seen):
    free = data(backend.compute_track(P0, 3, BPMS, start_at=start_at))
    stopped = backend.compute_track(P0, 3, BPMS, start_at=start_at, stop_at=stop_at)
    for uid, values in data(stopped).items():
        assert values.shape == free[uid].shape, "as many turns as asked for"
        if uid in seen:
            np.testing.assert_array_equal(values[:, :, 0], free[uid][:, :, 0])
        else:
            assert np.isnan(values[:, :, 0]).all(), uid
        assert np.isnan(values[:, :, 1:]).all(), "lost in all later turns"
    for e in stopped.per_element:
        assert list(e.get_survived().get_n_particles_survived_per_turn()) == [2 if e.uid in seen else 0, 0, 0]


@pytest.mark.parametrize("stop_at", ["NOPE", "D"])
def test_stop_at_one_element(backend, stop_at):
    """as start_at: exactly one element of that id ("D": several drifts)"""
    with pytest.raises(ValueError, match="expected one element"):
        backend.compute_track(P0, 1, BPMS, stop_at=stop_at)


def test_stop_at_set_through_the_backend(backend):
    async def run(stop_at):
        for prop, value in (("p0", P0), ("n_turns", 2), ("data_needed_at", BPMS), ("stop_at", stop_at)):
            await backend.set("turn_by_turn_start", prop, value)
        assert await backend.read("turn_by_turn_start", "stop_at") == (stop_at or None)
        return data(await backend.read("turn_by_turn", "pos"))

    assert np.isnan(asyncio.run(run("QF2"))["BPM3"]).all()
    assert np.isfinite(asyncio.run(run(""))["BPM3"]).all(), "removed again"
    assert np.isfinite(asyncio.run(run(None))["BPM3"]).all()
    with pytest.raises(TypeError):
        asyncio.run(run(["QF2"]))


def test_stop_leaves_the_lattice_alone(backend):
    n = len(backend.acc.acc)
    before = data(backend.compute_track(P0, 2, BPMS))
    backend.compute_track(P0, 2, BPMS, stop_at="QF2")
    after = data(backend.compute_track(P0, 2, BPMS))
    assert len(backend.acc.acc) == n
    for uid in BPMS:
        np.testing.assert_array_equal(before[uid], after[uid])


@pytest.mark.parametrize("start_at, candidates, first", [
    (None, ["B4", "QF2"], "QF2"),
    ("QF3", ["QF2", "B4"], "B4"),
    ("QF3", ["QF2", "QF3"], "QF3"),    # at the start
    (None, ["NOPE"], None),
    (None, [], None),
])
def test_first_element_reached(backend, start_at, candidates, first):
    uids = backend.get_element_uids()[: len(backend.acc.acc)]
    assert first_element_reached(uids, candidates, start_at=start_at) == first
