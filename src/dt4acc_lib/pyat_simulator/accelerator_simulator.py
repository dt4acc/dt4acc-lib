from .element_proxies import ElementProxy, ADDON_PROXY_REGISTRY
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
        Find a single element in the lattice by its UUID attribute.
        Returns [element] or raises ValueError if not found or ambiguous.
        UUID is unique per physical magnet — multiple JSON entries
        (main magnet + steerers) share the same UUID because they are
        coils on the same AT element.
        """
        matches = [
            elem for elem in self.acc
            if getattr(elem, "UUID", None) == uuid
        ]
        if len(matches) == 0:
            return None
        # Return the first match — multiple elements can share a UUID
        # (e.g. split elements) but they are physically the same magnet
        return [matches[0]]

    def get(self, element_id):
        """
        Retrieve an element proxy based on the given element ID.

        element_id is always a UUID from the JSON database (design view).
        The AT lattice element is found by matching element.UUID == element_id.

        FamName is NOT used — it is not unique across the ring
        (many sextupoles share the same FamName across different sectors).

        Compound ids "<type_prefix>:<host_uuid>" (e.g. "CQLN:OH2_001") are
        resolved via ADDON_PROXY_REGISTRY populated by facility-specific setup.
        """
        # 1. Compound id: "<type_prefix>:<host_uuid>" → registry
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
                    f"Host element with UUID {host_uuid!r} not found in lattice."
                )
            (element,) = matches
            return factory(element, element_id, host_uuid)

        # 2. UUID attribute search — the only correct lookup method
        matches = self._find_by_uuid(element_id)
        if matches is not None:
            return ElementProxy(matches, element_id=element_id)

        raise ValueError(
            f"Element with UUID {element_id!r} not found in lattice. "
            f"Check that element.UUID in the AT .m file matches the JSON uuid field."
        )

    @staticmethod
    def get_element_id_of_host(element_id: str) -> str:
        """
        Derives the host element ID from the provided element ID.
        Used by the EPICS path (H/V prefix convention).

        Args:
            element_id (str): The ID of the element.

        Returns:
            str: The ID of the host element.
        """
        if element_id.startswith("H") or element_id.startswith("V"):
            return element_id[1:]
        raise ValueError(f"Unknown element id: {element_id}")

    def instantiate_addon_proxy(self, sub_lattice, *, element_id, host_element_id):
        """
        Instantiates the correct proxy for the given sub lattice and element ID.
        Used by the EPICS path (H/V prefix convention).

        Args:
            sub_lattice: The AT sub lattice containing the element.
            element_id: The ID of the element.
            host_element_id: The ID of the host element.

        Returns:
            KickAngleCorrectorProxy: The proxy instance for the element.
        """
        from .element_proxies import KickAngleCorrectorProxy

        if not host_element_id.startswith("S"):
            raise ValueError(f"Unsupported host element ID: {host_element_id}")

        correction_plane = (
            "horizontal" if element_id.startswith("H")
            else "vertical" if element_id.startswith("V")
            else None
        )
        if correction_plane is None:
            raise ValueError(f"Unknown correction plane for element ID: {element_id}")

        return KickAngleCorrectorProxy(
            sub_lattice,
            element_id=element_id,
            host_element_id=host_element_id,
        )