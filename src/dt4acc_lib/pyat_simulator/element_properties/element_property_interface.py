from abc import ABCMeta, abstractmethod


class ElementPropertyInterface(metaclass=ABCMeta):
    """
    Todo:
        Shall these be acting as filters?

        Update and peek could be passed the object to work on

        shall all these methods be async ?
        These are typically fast
        Review after MAX IV poles strips have been implemented
    """

    @abstractmethod
    def handles_property(self) -> str:
        raise NotImplementedError("use derived class instead")

    @abstractmethod
    async def update(self, obj, value: object):
        """Updates value

        Todo: need to narrow type description down
        """
        raise NotImplementedError("use derived class instead")

    @abstractmethod
    def peek(self, obj) -> object:
        """returns value

        Todo: need to narrow type description down
        """
        raise NotImplementedError("use derived class instead")


__all__ = ["ElementPropertyInterface"]
