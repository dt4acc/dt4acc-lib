from importlib import resources

from dt4acc.custom_facility.bessyii.run_bessyii_twin import bessyii_pyat_lattice
from dt4acc_lib.pyat_simulator.accelerator_simulator import PyATAcceleratorSimulator
from dt4acc_lib.pyat_simulator.simulator_backend import SimulatorBackend


def test_get_survey() -> None:
    filename = resources.files("dt4acc").joinpath(
        "custom_facility/bessyii/resources/storage_ring/input/bessy2_storage_ring_reflat.json"
    )

    acc = bessyii_pyat_lattice(filename=filename)
    acc = PyATAcceleratorSimulator(at_lattice=acc)
    data = acc.get_survey()
    data
