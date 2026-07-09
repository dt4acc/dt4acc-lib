from importlib import resources
import pytest

from dt4acc_lib.pyat_simulator.accelerator_simulator import PyATAcceleratorSimulator
from dt4acc_lib.pyat_simulator.simulator_backend import SimulatorBackend


@pytest.fixture(scope="module")
def bessyii_lattice():
    bessyii_pyat_lattice = pytest.importorskip(
        "dt4acc.custom_facility.bessyii.run_bessyii_twin.bessyii_pyat_lattice"
    )
    filename = resources.files("dt4acc").joinpath(
        "custom_facility/bessyii/resources/storage_ring/input/bessy2_storage_ring_reflat.json"
    )
    return bessyii_pyat_lattice(filename=filename)


def test_get_survey(bessyii_lattice) -> None:

    acc = PyATAcceleratorSimulator(at_lattice=bessyii_lattice)
    data = acc.get_survey()
    data
