"""Transfer line mode: optics and trajectory propagated from the entering beam"""
import asyncio
import math

import at
import numpy as np
import pytest
from dt4acc_lib.model.input.twiss_in import TwissIn
from dt4acc_lib.pyat_simulator.accelerator_simulator import PyATAcceleratorSimulator
from dt4acc_lib.pyat_simulator.simulator_backend import SimulatorBackend

ENERGY = 1.7185e9
TWISS_IN = TwissIn(beta=(8.0, 3.0), alpha=(-0.5, 1.2), dispersion=(0.1, 0.01, 0.0, 0.0))


def line_lattice() -> at.Lattice:
    """A short open line: a bend, two quadrupoles, a steerer, three monitors"""
    return at.Lattice(
        [
            at.Monitor("BPM1"),
            at.Drift("D1", 1.0),
            at.Dipole("B1", 1.0, 0.1),
            at.Drift("D2", 0.5),
            at.Quadrupole("QF", 0.3, 1.2),
            at.Drift("D3", 0.5),
            at.Corrector("HV1", 0.1, [0.0, 0.0]),
            at.Monitor("BPM2"),
            at.Drift("D4", 0.5),
            at.Quadrupole("QD", 0.3, -1.1),
            at.Drift("D5", 2.0),
            at.Monitor("BPM3"),
        ],
        name="test line",
        energy=ENERGY,
    )


def fodo_ring() -> at.Lattice:
    cells = []
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
    ring = at.Lattice(cells, name="test ring", energy=ENERGY, periodicity=1)
    ring.disable_6d()
    return ring


def line_backend(twiss_in: TwissIn = TWISS_IN) -> SimulatorBackend:
    acc = PyATAcceleratorSimulator(at_lattice=line_lattice(), twiss_in=twiss_in)
    return SimulatorBackend(acc=acc, name="line")


def monitor_positions(backend: SimulatorBackend) -> dict:
    track = asyncio.run(backend.read("track", "pos"))
    return {p.fam_name: (p.x, p.y) for p in track.track if p.fam_name.startswith("BPM")}


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(beta=(1.0,)),
        dict(beta=(1.0, 0.0)),
        dict(beta=(1.0, -2.0)),
        dict(beta=(1.0, 1.0), alpha=(0.0, math.nan)),
        dict(beta=(1.0, 1.0), closed_orbit=(0.0,) * 4),
    ],
)
def test_twiss_in_rejects_invalid(kwargs) -> None:
    with pytest.raises(ValueError):
        TwissIn(**kwargs)


def test_ring_is_default() -> None:
    acc = PyATAcceleratorSimulator(at_lattice=fodo_ring())
    assert acc.is_ring
    assert not PyATAcceleratorSimulator(at_lattice=line_lattice(), twiss_in=TWISS_IN).is_ring


def test_line_optics_match_pyat() -> None:
    """Line mode gives what pyAT gives for the same entering beam"""
    acc = PyATAcceleratorSimulator(at_lattice=line_lattice(), twiss_in=TWISS_IN)
    _, _, elem_data = acc.get_optics_parameters()
    _, _, expected = line_lattice().get_optics(at.All, twiss_in=TWISS_IN.as_at())
    np.testing.assert_allclose(elem_data.beta, expected.beta)
    np.testing.assert_allclose(elem_data.alpha, expected.alpha)
    np.testing.assert_allclose(elem_data.dispersion, expected.dispersion)
    # starts where the entering beam says
    np.testing.assert_allclose(elem_data.beta[0], TWISS_IN.beta)
    np.testing.assert_allclose(elem_data.alpha[0], TWISS_IN.alpha)


def test_line_twiss_starts_at_twiss_in() -> None:
    twiss = asyncio.run(line_backend().read("twiss", "parameters"))
    first = twiss.twiss[0]
    assert first.fam_name == "BPM1"
    assert first.x.beta == pytest.approx(TWISS_IN.beta[0])
    assert first.y.beta == pytest.approx(TWISS_IN.beta[1])


def test_line_trajectory_is_tracked_centroid() -> None:
    """Monitors read the trajectory of the entering centroid, checked by tracking it"""
    x0 = (1e-3, -2e-4, -5e-4, 1e-4, 0.0, 0.0)
    backend = line_backend(TwissIn(beta=(8.0, 3.0), closed_orbit=x0))
    readings = monitor_positions(backend)

    lattice = line_lattice()
    refpts = lattice.get_uint32_index(at.Monitor)
    rout, *_ = lattice.track(np.array(x0, dtype=float).reshape(6, 1), refpts=refpts, nturns=1)
    for k, idx in enumerate(refpts):
        name = lattice[idx].FamName
        assert readings[name][0] == pytest.approx(rout[0, 0, k, 0], abs=1e-12)
        assert readings[name][1] == pytest.approx(rout[2, 0, k, 0], abs=1e-12)
    assert readings["BPM1"] == pytest.approx((x0[0], x0[2]))


def test_line_end_point_is_exit_not_first_element() -> None:
    """The extra point at the end of a line is its exit, not the first element again"""
    track = asyncio.run(line_backend().read("track", "pos"))
    names = [p.fam_name for p in track.track]
    uids = [p.uid for p in track.track]
    assert names[0] == "BPM1"
    assert names[-1] == "BPM3:exit"
    assert uids.count("BPM1") == 1
    assert len(set(uids)) == len(uids)


def test_line_steerer_moves_downstream_only() -> None:
    backend = line_backend()
    before = monitor_positions(backend)
    assert all(v == pytest.approx((0.0, 0.0), abs=1e-15) for v in before.values())

    asyncio.run(backend.set("HV1", "x_kick", 1e-4))
    after = monitor_positions(backend)
    # BPM2 sits right after the steerer: a thin kick there moves the angle, the
    # 0.1 m long corrector a bit of the position too
    assert after["BPM1"] == pytest.approx((0.0, 0.0), abs=1e-15)
    assert abs(after["BPM3"][0]) > 1e-5
    assert after["BPM3"][1] == pytest.approx(0.0, abs=1e-15)


def test_line_has_no_tune_and_no_chromaticity() -> None:
    backend = line_backend()
    tune = asyncio.run(backend.read("tune", "transversal"))
    chroma = asyncio.run(backend.read("chromaticity", "transversal"))
    assert math.isnan(tune.x) and math.isnan(tune.y)
    assert math.isnan(chroma.x) and math.isnan(chroma.y)


def test_ring_still_periodic() -> None:
    backend = SimulatorBackend(acc=PyATAcceleratorSimulator(at_lattice=fodo_ring()), name="ring")
    tune = asyncio.run(backend.read("tune", "transversal"))
    _, ring_data, _ = fodo_ring().get_optics(at.All)
    assert tune.x == pytest.approx(ring_data.tune[0])
    assert tune.y == pytest.approx(ring_data.tune[1])
    assert 0 < tune.x < 1 and 0 < tune.y < 1
