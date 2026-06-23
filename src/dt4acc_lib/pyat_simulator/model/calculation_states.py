import enum


class CalculationStates(enum.Enum):
    pending = "pending"
    """Changes were made, need to calculate to get orbit"""

    executing = "executing"
    """Optics parameters are being computed"""

    finished = "finished"
    """Optics parameters have been computed"""

    error = "error"
    """Optics calculation failed"""

    acknowledged = "acknowledged"
    """Optics error calculation failed: user has acknowledged this error

    This means:  all element parameters can now be reset. But no optics
                 calculation will be performed by itself

    This state is current here, so that one can always see, why there are
    no optics parameters
    """
