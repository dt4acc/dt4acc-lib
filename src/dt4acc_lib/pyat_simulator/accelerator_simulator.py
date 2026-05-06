from .element_proxies import ElementProxy
# Todo: revisit should that not rather be a storage
from .element_proxies import ADDON_PROXY_REGISTRY
from dt4acc_lib.interfaces.simulator.accelerator_simulator import AcceleratorSimulatorInterface
import at


class PyATAcceleratorSimulator(AcceleratorSimulatorInterface):
    """
    Factory class for creating proxies for accelerator elements.

    This class provides an interface for retrieving accelerator element proxies
    from a given lattice structure.

    Warning:
        Currently, this implementation uses an `at_lattice` directly,
        which should be revised when a proper lattice model becomes available.

    Todo:
        Revisit name: still a proxy? The element proxy currently not necessary`?

        Leave addon element proxy e.g. for handling combined function magnets

        Do we only need to calculate optics? How much one needs to optimise
        calculation speed.


    """

    def __init__(self, *, at_lattice):
        """
        Initialize the proxy factory.

        Args:
            at_lattice: The actual AT lattice used to retrieve elements.
        """
        self.acc = at_lattice

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(at_lattice={self.acc})"

    def get_optics_parameters(self):
        x0, ring_pars, elem_data = self.acc.get_optics(at.All)
        return x0, ring_pars, elem_data

    # def get_twiss(self) -> Twiss:
    #     pass
    #
    # def get_track(self) -> CalculatedTrack:
    #     pass

    # def get_tune(self) -> Tune:
    #    """
    #    Todo:
    #        review who else needs these data
    #    """
    #    _, ring_pars, elem_data = self._get_optics_parameter()
    #
    #    tune = ring_pars["tune"]
    #    return Tune(x=tune[0], y=tune[1])

    def _find_by_uuid(self, uuid: str):
        """
        Find element(s) in the lattice by UUID or FamName.

        Strategy (in order):
        1. UUID attribute match: element.UUID == uuid  (SOLEIL .m lattice)
        2. FamName match: element.FamName == uuid      (MAX IV JSON lattice)

        Returns [element] or None if not found.
        """
        # 1. UUID attribute search (SOLEIL)
        matches = [
            elem for elem in self.acc
            if getattr(elem, "UUID", None) == uuid
        ]
        if matches:
            return [matches[0]]

        # 2. FamName search (MAX IV JSON lattice)
        matches = [
            elem for elem in self.acc
            if getattr(elem, "FamName", None) == uuid
        ]
        if matches:
            return [matches[0]]

        return None

    def _find_by_uuids(self, uuids: list):
        """
        Find multiple elements by a list of FamName/UUID strings.
        Returns a list of elements in the same order as uuids.
        Used for multi-element devices (e.g. SQFI: 4 AT elements per magnet).
        Elements not found are silently skipped.
        """
        result = []
        for uuid in uuids:
            matches = self._find_by_uuid(uuid)
            if matches:
                result.extend(matches)
        return result if result else None

    def get(self, element_id):
        """
        Retrieve an element proxy based on the given element ID.

        element_id can be:
          - A single UUID string (SOLEIL): matched via element.UUID attribute
          - A single FamName string (MAX IV): matched via element.FamName
          - A list of FamName strings (MAX IV multi-element): all matched,
            returned as a group proxy so writes apply to all elements
          - A compound string "<prefix>:<host>" (CQLN/skew): resolved via
            ADDON_PROXY_REGISTRY

        Compound ids "<type_prefix>:<host_uuid>" (e.g. "CQLN:OH2_001",
        "skew:69") are resolved via ADDON_PROXY_REGISTRY populated by
        facility-specific setup.
        """
        # 0. Array of uuids — multi-element device (MAX IV SQFI/SXDI/etc.)
        if isinstance(element_id, list):
            elements = self._find_by_uuids(element_id)
            if elements is None:
                raise ValueError(
                    f"No elements found for uuid list {element_id!r}. "
                    f"Check FamNames match the AT JSON lattice."
                )
            return ElementProxy(elements, element_id=element_id[0])

        # 1. Compound id: "<type_prefix>:<host>" → ADDON_PROXY_REGISTRY
        if ":" in element_id:
            type_prefix, host_uuid = element_id.split(":", 1)
            factory = ADDON_PROXY_REGISTRY.get(type_prefix)
            if factory is None:
                raise ValueError(
                    f"No proxy registered for type prefix {type_prefix!r}. "
                    f"Register it in element_proxies.ADDON_PROXY_REGISTRY before use."
                )
            matches = self._find_by_uuid(host_uuid)
            if matches is None:
                raise ValueError(
                    f"Host element with UUID/FamName {host_uuid!r} not found in lattice."
                )
            (element,) = matches
            return factory(element, element_id, host_uuid)

        # 2. Single UUID/FamName search
        matches = self._find_by_uuid(element_id)
        if matches is not None:
            return ElementProxy(matches, element_id=element_id)

        raise ValueError(
            f"Element with UUID/FamName {element_id!r} not found in lattice. "
            f"Check that the uuid field in the JSON setup matches either "
            f"element.UUID (SOLEIL) or element.FamName (MAX IV) in the lattice."
        )

