# Digital twin for particle accelerators: library part (dt4acc-lib) 

## Rings and transfer lines

`PyATAcceleratorSimulator` computes the optics of a ring by default: the
periodic solution, the closed orbit at the monitors, tune and chromaticity.

A transfer line has no periodic solution. Give the beam entering it as
`twiss_in` and the simulator works as a line: optics are propagated from that
beam, the monitors read the trajectory of its centroid, tune and chromaticity
are NaN.

```python
from dt4acc_lib.model.input import TwissIn
from dt4acc_lib.pyat_simulator.accelerator_simulator import PyATAcceleratorSimulator

twiss_in = TwissIn(
    beta=(8.0, 3.0),                  # m
    alpha=(-0.5, 1.2),
    dispersion=(0.1, 0.01, 0.0, 0.0),  # eta_x, eta_x', eta_y, eta_y'
    closed_orbit=(1e-3, 0, 0, 0, 0, 0),  # entering centroid x, x', y, y', delta, ct
)
acc = PyATAcceleratorSimulator(at_lattice=line, twiss_in=twiss_in)
```

The optics of a line have one point more than it has elements, as for a ring;
it is labelled `<last element>:exit` (in a ring it repeats the first element).
