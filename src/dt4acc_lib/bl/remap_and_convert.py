import logging

from ..interfaces.utils.translator_service import StateConversion
from ..model.output.track_as_np_wrapper import (
    NPStatePerElementPerTurn,
    NPStatesForTurns,
)

logger = logging.getLogger("dt4acc_lib")


class RemapIdentifiersAndConvert(StateConversion):
    def __init__(self, name_map, convert) -> None:
        self.name_map = name_map
        self.convert = convert

    def forward(self, state: NPStatesForTurns) -> NPStatesForTurns:
        r = []
        for data in state.per_element:
            try:
                uid = self.name_map[data.uid]
            except KeyError:
                logger.info("%s: Ignoring uid %s as no map provided", self.__class__.__name__, data.uid)
                continue
            # Todo: need to implement conversion too!
            for_elem = NPStatePerElementPerTurn(track_data=data.track_data, uid=uid)
            r.append(for_elem)
        return NPStatesForTurns(r)

    def inverse(self, state: object) -> object:
        raise AttributeError("inverse method should not be called")
