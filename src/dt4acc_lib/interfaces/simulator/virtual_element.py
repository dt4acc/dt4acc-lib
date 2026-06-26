from abc import ABCMeta, abstractmethod
from typing import Sequence, Union


class VirtualElementInterface(metaclass=ABCMeta):
    """A storage for some calculation start data

    The back engine does not now when it will be asked
    to track some data through some accelerator. For that
    it needs start info. This info can arrive any time and
    gets updated as it

    Todo:
        should one name the methods peek and update instead?
        the interface is too similar to element.ElementInterface

        Is that rather a virtual element?
    """
    @abstractmethod
    def get(self, prop_id: str) -> object:
        raise NotImplementedError("use derived class instead")

    @abstractmethod
    def set(self, prop_id: str, value: Union[int, float, Sequence[str]]) -> object:
        raise NotImplementedError("use derived class instead")

