"""Info on which device belongs to which other

Todo:
    provide enum of standard family names?
"""
from abc import abstractmethod, ABCMeta
from typing import Sequence


class YellowPagesBase(metaclass=ABCMeta):
    """Handling lattice element and the families they belong to

    Two ways to see it:
    * family as seen by a lattice:
      typically some magnets that are split all over the place
    * family sees as functionality:
      e.g. a quadrupole can belong to the focusing family
      and the tune correction quadrupoles

    Imagine for some lattice e.g. some quadrupoles could be
    identically as they are produced. But they could still
    belong to different families.

    Todo:
        need to define standard family names (as a Enum)
    """

    @abstractmethod
    def get(self, family_name: str) -> Sequence[str]:
        """Return a sequence with all identifiers belonging to base class"""
        raise NotImplementedError("use derived class instead")
