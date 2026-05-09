from abc import ABCMeta, abstractmethod


from .state_conversion import StateConversion
from dt4acc_lib.model.utils.identifiers import ConversionID


class TranslatorServiceBase(metaclass=ABCMeta):
    """Provide objects that can translate recalculate the value

    Actor says:

    * I know:
        * I want to change property "A" of lattice element "B"
        * I know that device "C" needs to change property "D"
    * please give me the translation object that converts between these
    """

    @abstractmethod
    def get(self, id_: ConversionID) -> StateConversion:
        pass
