"""Beam survival: particles around the closed orbit tracked for some turns"""
import asyncio

import at
import numpy as np
import pytest

from dt4acc_lib.model.output.beam_survival import BeamSurvival, BeamSurvivalSettings
from dt4acc_lib.pyat_simulator.accelerator_simulator import PyATAcceleratorSimulator
from dt4acc_lib.pyat_simulator.simulator_backend import SimulatorBackend


def ring(k: float = 0.5, aperture=None) -> at.Lattice:
    cells = [at.Marker("START"), at.Drift("D0", 0.5)]
    for i in range(8):
        cells += [
            at.Monitor(f"BPM{i}"),
            at.Drift("D", 0.4),
            at.Quadrupole(f"QF{i}", 0.3, k),
            at.Drift("D", 0.4),
            at.Dipole(f"B{i}", 1.0, 2 * np.pi / 8),
            at.Drift("D", 0.4),
            at.Quadrupole(f"QD{i}", 0.3, -k),
            at.Drift("D", 0.4),
        ]
    if aperture is not None:
        cells[1].EApertures = np.asarray(aperture, dtype=float)
    lattice = at.Lattice(cells, name="ring", energy=1.5e9, periodicity=1)
    lattice.disable_6d()
    return lattice


def survival(lattice, **settings) -> BeamSurvival:
    backend = SimulatorBackend(acc=PyATAcceleratorSimulator(at_lattice=lattice), name="ring",
                               beam_survival_settings=BeamSurvivalSettings(n_turns=50, **settings))
    return asyncio.run(backend.read("beam_survival", "stored")), backend


def test_stored():
    r, backend = survival(ring())
    assert r.stored and r.closed_orbit
    assert len(r.offsets) == 5 and not any(r.lost)
    sx, sy = r.sigma
    assert sx > 0 and sy > 0
    assert r.offsets[1][0] == pytest.approx(5 * sx) and r.offsets[3][1] == pytest.approx(5 * sy)
    # kept until the lattice changes
    assert asyncio.run(backend.read("beam_survival", "stored")) is r


def test_lost_at_an_aperture():
    # apertures in the lattice count as they are: the particles at +- n sigma hit it, the closed orbit does not
    r, _ = survival(ring(), n_sigma=5.0)
    a = 2 * r.offsets[1][0]
    r, _ = survival(ring(aperture=[a, 1.0]), n_sigma=5.0)
    assert r.stored
    r, _ = survival(ring(aperture=[a, 1.0]), n_sigma=20.0)
    assert not r.stored and r.closed_orbit
    assert r.lost == [False, True, True, False, False]


def test_no_closed_orbit():
    # an unstable ring (pyAT: AtError "Unstable ring")
    r, _ = survival(ring(k=1.2))
    assert not r.stored and not r.closed_orbit and not r.lost


def test_emittance_and_energy_spread():
    ex, se = PyATAcceleratorSimulator(at_lattice=ring()).emittance_and_energy_spread()
    rp = ring().radiation_parameters()
    assert ex == pytest.approx(rp.emittances[0]) and se == pytest.approx(rp.sigma_e)
