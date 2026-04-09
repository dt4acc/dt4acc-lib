"""convert from lattice element change to device property

Please note:
    here we have to map (lattice_name, property) -> (device_name, property)
"""

from typing import Sequence

from dt4acc_lib.model.utils.command import ReadCommand
from dt4acc_lib.interfaces.utils.command_rewritter import CommandRewriterBase
from dt4acc_lib.interfaces.utils.liaison_manager import LiaisonManagerBase
from dt4acc_lib.interfaces.utils.translator_service import TranslatorServiceBase
from dt4acc_lib.model.utils.command import Command
from dt4acc_lib.model.utils.identifiers import (
    DevicePropertyID,
    LatticeElementPropertyID,
    ConversionID,
)


class CommandRewriter(CommandRewriterBase):
    """Command rewriter.

    This module translates commands between two state spaces:
    from a lattice element specification to a device property specification (forward)
    and vice versa (inverse).

    It uses a liaison manager to map identifiers and a translator service
    to convert the command values between representations.
    """

    def __init__(
        self,
        liaison_manager: LiaisonManagerBase,
        translation_service: TranslatorServiceBase,
    ):
        """
        Initialize the CommandRewriter with the required services.

        Args:
            liaison_manager: The manager responsible for mapping identifiers.
            translator_service: The service providing value conversions.
        """
        self.translator_service = translation_service
        self.liaison_manager = liaison_manager

    def inverse(self, cmd: Command) -> Sequence[Command]:
        """
        Convert a command from a device property space back to the lattice space.

        Args:
            :param cmd: The command to be transformed.

        Returns:
            A sequence of commands corresponding to the inverse translations.
        """
        dev_prop_id = DevicePropertyID(device_name=cmd.id, property=cmd.property)
        rcmd = self.inverse_read_command(cmd)
        lat_prop_ids = [
            LatticeElementPropertyID(element_name=r.id, property=r.property)
            for r in rcmd
        ]

        return [
            self.inverse_translate_one(cmd, dev_prop_id, lat_prop_id)
            for lat_prop_id in lat_prop_ids
        ]

    def inverse_translate_one(
        self,
        cmd: Command,
        dev_prop_id: DevicePropertyID,
        lat_prop_id: LatticeElementPropertyID,
    ) -> Command:
        """
        Perform a single inverse translation.

        Args:
            :param dev_prop_id: The source device property identifier
            :param cmd: The original command
            :param lat_prop_id: The target lattice property identifier

        Returns:
            A new Command with the value converted to the lattice state.
        """
        translation_object = self.translator_service.get(
            ConversionID(
                lattice_property_id=lat_prop_id, device_property_id=dev_prop_id
            )
        )

        if dev_prop_id.device_name is None:
            raise ValueError(
                "Device name cannot be None in device property identifier."
            )

        ncmd = Command(
            id=lat_prop_id.element_name,
            property=lat_prop_id.property,
            value=translation_object.inverse(cmd.value),
            behaviour_on_error=cmd.behaviour_on_error,
        )
        return ncmd

    def forward(self, cmd: Command) -> Sequence[Command]:
        rcmds = self.forward_read_command(cmd)
        dev_prop_ids = [
            DevicePropertyID(device_name=rcmd.id, property=rcmd.property)
            for rcmd in rcmds
        ]
        lat_prop_id = LatticeElementPropertyID(
            element_name=cmd.id, property=cmd.property
        )

        return [
            self.forward_translate_one(cmd, lat_prop_id, dev_prop_id)
            for dev_prop_id in dev_prop_ids
        ]

    def forward_translate_one(
            self,
            cmd: Command,
            lat_prop_id : LatticeElementPropertyID,
            dev_prop_id: DevicePropertyID):
        translation_object = self.translator_service.get(
            ConversionID(
                lattice_property_id=lat_prop_id, device_property_id=dev_prop_id
            )
        )
        ncmd = Command(
            id=dev_prop_id.device_name,
            property=dev_prop_id.property,
            value=translation_object.forward(cmd.value),
            behaviour_on_error=cmd.behaviour_on_error,
        )
        return ncmd



    def forward_read_command(self, command: ReadCommand) -> Sequence[ReadCommand]:
        """
        Todo:
            is there a use case where one lattice read command
            goes to more than one device ?
        """
        lat_prop_id = LatticeElementPropertyID(
            element_name=command.id, property=command.property
        )
        dev_prop_ids = self.liaison_manager.forward(lat_prop_id)
        return [
            ReadCommand(id=dev_prop_id.device_name, property=dev_prop_id.property)
            for dev_prop_id in dev_prop_ids
        ]

    def inverse_read_command(self, command: ReadCommand) -> Sequence[ReadCommand]:
        dev_prop_id = DevicePropertyID(
            device_name=command.id, property=command.property
        )
        lat_prop_ids = self.liaison_manager.inverse(dev_prop_id)
        return [
            ReadCommand(id=lp.element_name, property=lp.property) for lp in lat_prop_ids
        ]


__all__ = ["CommandRewriter"]
