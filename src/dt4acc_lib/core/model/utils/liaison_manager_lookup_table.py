from collections import defaultdict
from dataclasses import dataclass
from functools import cached_property
from typing import Sequence, Hashable, Callable, TypeVar

from .identifiers import DevicePropertyID, LatticeElementPropertyID


@dataclass
class LiaisonManagerForwardLookupElement:
    lat_id : LatticeElementPropertyID
    dev_ids : Sequence[DevicePropertyID]


@dataclass
class LiaisonManagerInverseLookupElement:
    dev_id : DevicePropertyID
    lat_ids : Sequence[LatticeElementPropertyID]


@dataclass
class LiaisonManagerForwardLookupTable:
    """
    Todo:
        see what can be made common with LiaisonManagerInverseLookupTable
    """
    lut: Sequence[LiaisonManagerForwardLookupElement]

    def get(self, item: LatticeElementPropertyID):
        return self._dict.get(item)

    def non_unique_entries(self):
        return check_unique_entries(self.lut, lambda elem: elem.lat_id)

    def verify(self):
        tmp = self.non_unique_entries()
        assert tmp == {}, f"Following entries are not unique {tmp}"

    def keys(self):
        return self._dict.keys()

    @cached_property
    def _dict(self):
        return {entry.lat_id: entry.dev_ids for entry in self.lut}


@dataclass
class LiaisonManagerInverseLookupTable:
    lut: Sequence[LiaisonManagerInverseLookupElement]

    def non_unique_entries(self):
        return check_unique_entries(self.lut, lambda elem: elem.dev_id)

    def verify(self):
        tmp = self.non_unique_entries()
        assert tmp == {}, f"Following entries are not unique {tmp}"

    def get(self, item: DevicePropertyID):
        return self._dict.get(item)

    def keys(self):
        return self._dict.keys()

    @cached_property
    def _dict(self):
        return {entry.dev_id: entry.lat_ids for entry in self.lut}


T = TypeVar("T")


def check_unique_entries(
    elems: Sequence[T], extract_id: Callable[[T], Hashable]
) -> Sequence[T]:
    d = defaultdict(list)
    for elem in elems:
        d[extract_id(elem)].append(elem)
    return {k : v for k,v in d.items() if len(v) > 1}


