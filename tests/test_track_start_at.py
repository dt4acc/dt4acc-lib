"""Tracking from an element of the ring (e.g. an injection point), and
calibrations given at a reference brho"""
import asyncio

import at
import numpy as np
import pytest

from dt4acc_lib.bl.translator_service import create_translation_object_for_polynom_coefficients
from dt4acc_lib.model.output.track import ParticleState
from dt4acc_lib.model.utils.translator_manager_lookup_table import PolynomCoefficients
from dt4acc_lib.pyat_simulator.accelerator_simulator import PyATAcceleratorSimulator
from dt4acc_lib.pyat_simulator.simulator_backend import SimulatorBackend


def ring() -> at.Lattice:
    cells = [at.Marker("START"), at.Drift("D0", 0.5), at.Marker("INJ")]
    for i in range(8):
        cells += [
            at.Monitor(f"BPM{i}"),
            at.Drift("D", 0.4),
            at.Quadrupole(f"QF{i}", 0.3, 1.0),
            at.Drift("D", 0.4),
            at.Dipole(f"B{i}", 1.0, 2 * np.pi / 8),
            at.Drift("D", 0.4),
            at.Quadrupole(f"QD{i}", 0.3, -1.0),
            at.Drift("D", 0.4),
        ]
    lattice = at.Lattice(cells, name="ring", energy=1.5e9, periodicity=1)
    lattice.disable_6d()
    return lattice


def track(backend, p0, n_turns, monitors, start_at=None):
    asyncio.run(backend.set("turn_by_turn_start", "p0", [ParticleState.from_sequence(p0)]))
    asyncio.run(backend.set("turn_by_turn_start", "n_turns", n_turns))
    asyncio.run(backend.set("turn_by_turn_start", "data_needed_at", monitors))
    asyncio.run(backend.set("turn_by_turn_start", "start_at", start_at))
    data = asyncio.run(backend.read("turn_by_turn", "pos"))
    return {e.uid: (e.get_x().nanmean_per_turn(), e.get_y().nanmean_per_turn()) for e in data.per_element}, data


@pytest.mark.parametrize("start_at", ["INJ", "QF3"])
def test_tracking_starts_at_the_element(start_at) -> None:
    lattice = ring()
    backend = SimulatorBackend(acc=PyATAcceleratorSimulator(at_lattice=lattice), name="ring")
    monitors = [e.FamName for e in lattice if isinstance(e, at.Monitor)]
    p0 = np.array([1e-3, 2e-4, -5e-4, 1e-4, 0.0, 0.0])
    got, data = track(backend, p0, 3, monitors, start_at=start_at)

    (idx,) = [i for i, e in enumerate(lattice) if e.FamName == start_at]
    rotated = lattice.rotate(idx)
    refpts = [i for i, e in enumerate(rotated) if isinstance(e, at.Monitor)]
    out, *_ = rotated.track(p0.reshape(6, 1).copy(), refpts=refpts, nturns=3)
    order = [rotated[i].FamName for i in refpts]
    # the data come in the order the beam passes the monitors
    assert [e.uid for e in data.per_element] == order
    for k, name in enumerate(order):
        np.testing.assert_allclose(got[name][0], out[0, 0, k, :], atol=1e-15)
        np.testing.assert_allclose(got[name][1], out[2, 0, k, :], atol=1e-15)


def test_start_at_none_is_the_lattice_start() -> None:
    lattice = ring()
    backend = SimulatorBackend(acc=PyATAcceleratorSimulator(at_lattice=lattice), name="ring")
    monitors = ["BPM0", "BPM5"]
    p0 = np.array([1e-3, 0.0, 0.0, 0.0, 0.0, 0.0])
    a, _ = track(backend, p0, 2, monitors, start_at=None)
    b, _ = track(backend, p0, 2, monitors, start_at="START")
    for name in monitors:
        np.testing.assert_allclose(a[name][0], b[name][0], atol=1e-15)


def test_start_at_sees_magnet_changes() -> None:
    """The rotated lattice shares the elements: a set reaches the tracking"""
    lattice = ring()
    backend = SimulatorBackend(acc=PyATAcceleratorSimulator(at_lattice=lattice), name="ring")
    p0 = np.zeros(6)
    before, _ = track(backend, p0, 1, ["BPM5"], start_at="INJ")
    asyncio.run(backend.set("QF2", "main_strength", 1.05))
    asyncio.run(backend.set("QF2", "dx", 1e-4))
    after, _ = track(backend, p0, 1, ["BPM5"], start_at="INJ")
    assert abs(after["BPM5"][0][0] - before["BPM5"][0][0]) > 1e-7


def test_unknown_start_element_raises() -> None:
    backend = SimulatorBackend(acc=PyATAcceleratorSimulator(at_lattice=ring()), name="ring")
    with pytest.raises(ValueError):
        track(backend, np.zeros(6), 1, ["BPM0"], start_at="NOPE")


def test_calibration_at_reference_brho() -> None:
    """kick per current given at the design energy: the kick scales with 1 / brho"""
    kick_per_amp, ref_brho = 6.6e-4, 5.73
    data = PolynomCoefficients([0.0, 1.0 / kick_per_amp], energy_dependent=True, reference_brho=ref_brho)
    at_design = create_translation_object_for_polynom_coefficients(data, brho=ref_brho)
    assert at_design.inverse(2.0) == pytest.approx(2.0 * kick_per_amp)
    higher = create_translation_object_for_polynom_coefficients(data, brho=1.1 * ref_brho)
    assert higher.inverse(2.0) == pytest.approx(2.0 * kick_per_amp / 1.1)
    assert higher.forward(higher.inverse(2.0)) == pytest.approx(2.0)
    with pytest.raises(ValueError):
        create_translation_object_for_polynom_coefficients(
            PolynomCoefficients([0.0, 1.0], energy_dependent=False, reference_brho=ref_brho), brho=ref_brho)
