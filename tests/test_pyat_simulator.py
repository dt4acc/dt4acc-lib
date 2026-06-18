from importlib import resources

import numpy as np
import pytest

from dt4acc.custom_facility.bessyii.run_bessyii_twin import bessyii_pyat_lattice
from dt4acc_lib.pyat_simulator.accelerator_simulator import PyATAcceleratorSimulator


@pytest.fixture(scope="module")
def bessyii_simulator():
    filename = resources.files("dt4acc").joinpath(
        "custom_facility/bessyii/resources/storage_ring/input/bessy2_storage_ring_reflat.json"
    )

    acc = bessyii_pyat_lattice(filename=filename)
    acc = PyATAcceleratorSimulator(at_lattice=acc)
    return acc

def test_get_survey(bessyii_simulator) -> None:
    data = bessyii_simulator.get_survey()
    data

def test_tracking(bessyii_simulator) -> None:
    p0 = np.zeros(6)
    data = bessyii_simulator.track(p0, n_turns=3)

    data


