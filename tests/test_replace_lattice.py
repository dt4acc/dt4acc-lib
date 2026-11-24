"""A lattice with small differences (e.g. an insertion device added) replaces
the simulated one at run time (dt4acc#81)"""
import asyncio

import at
import numpy as np
import pytest

from dt4acc_lib.pyat_simulator.accelerator_simulator import (
    LatticeMismatch,
    PyATAcceleratorSimulator,
    check_replacement,
)
from dt4acc_lib.pyat_simulator.simulator_backend import SimulatorBackend


def ring(with_insertion: bool = False, kf: float = 0.9) -> at.Lattice:
    """8 FODO cells; ``with_insertion``: the first straight's drift split
    around a thin vertical kick (an insertion device in the model) and a marker"""
    if with_insertion:
        straight = [
            at.Drift("D0", 0.25),
            at.Marker("ID_START"),
            at.Corrector("ID", 0.0, [0.0, 1e-4]),
            at.Marker("ID_END"),
            at.Drift("D0", 0.25),
        ]
    else:
        straight = [at.Drift("D0", 0.5)]
    cells = [at.Marker("START"), *straight, at.Marker("INJ")]
    for i in range(8):
        cells += [
            at.Monitor(f"BPM{i}"),
            at.Drift("D", 0.4),
            at.Quadrupole(f"QF{i}", 0.3, kf),
            at.Drift("D", 0.4),
            at.Dipole(f"B{i}", 1.0, 2 * np.pi / 8),
            at.Drift("D", 0.4),
            at.Quadrupole(f"QD{i}", 0.3, -1.0),
            at.Drift("D", 0.4),
        ]
    lattice = at.Lattice(cells, name="ring", energy=1.5e9, periodicity=1)
    lattice.disable_6d()
    return lattice


def devices(lattice: at.Lattice) -> list:
    """what the twin refers to: magnets and monitors"""
    return [e.FamName for e in lattice if isinstance(e, (at.Quadrupole, at.Dipole, at.Monitor))]


def tune(backend):
    t = asyncio.run(backend.read("tune", "transversal"))
    return t.x, t.y


def test_lattice_with_an_insertion_replaces_the_present_one():
    backend = SimulatorBackend(acc=PyATAcceleratorSimulator(at_lattice=ring()), name="ring")
    names_before = backend.get_element_names()
    qx0, qy0 = tune(backend)

    new = ring(with_insertion=True, kf=0.91)
    asyncio.run(backend.load_lattice(new))

    assert backend.get_element_names() == [e.FamName for e in new] + [new[0].FamName]
    assert len(backend.get_element_names()) == len(names_before) + 4
    qx1, qy1 = tune(backend)
    reference = new.get_tune()
    np.testing.assert_allclose([qx1, qy1], reference[:2], atol=1e-9)
    assert abs(qx1 - qx0) > 1e-4
    # the simulated lattice is the new one: the insertion's kick is in it
    (inserted,) = [e for e in backend.acc.acc if e.FamName == "ID"]
    np.testing.assert_allclose(inserted.KickAngle, [0.0, 1e-4])


def test_reinit_returns_to_the_replacing_lattice():
    simulator = PyATAcceleratorSimulator(at_lattice=ring())
    new = ring(with_insertion=True)
    simulator.replace_lattice(new)
    simulator.acc[3].KickAngle = np.array([0.0, 5e-4])  # a change made in the twin
    simulator.reinit()
    assert len(simulator.acc) == len(new)
    np.testing.assert_allclose(simulator.acc[3].KickAngle, [0.0, 1e-4])


def test_a_missing_element_is_refused():
    present = ring()
    new = ring(with_insertion=True)
    del new[[e.FamName for e in new].index("QF3")]
    with pytest.raises(LatticeMismatch, match="QF3"):
        check_replacement(present, new, devices(present))


def test_replacing_does_not_check():
    """the check is the application's: without it any lattice is taken"""
    simulator = PyATAcceleratorSimulator(at_lattice=ring())
    new = ring()
    del new[[e.FamName for e in new].index("QF3")]
    simulator.replace_lattice(new)
    assert [e.FamName for e in simulator.acc] == [e.FamName for e in new]


def test_reordered_elements_are_refused():
    present = ring()
    new = ring()
    i, j = (next(k for k, e in enumerate(new) if e.FamName == name) for name in ("BPM1", "BPM2"))
    new[i], new[j] = new[j], new[i]
    with pytest.raises(LatticeMismatch, match="order"):
        check_replacement(present, new, devices(present))


def test_elements_nobody_refers_to_may_change():
    present = ring(with_insertion=True)
    new = ring()  # the insertion and its markers removed
    check_replacement(present, new, devices(present))


def test_required_elements_unknown_to_the_present_lattice_are_ignored():
    check_replacement(ring(), ring(), devices(ring()) + ["NOT_IN_ANY"])
