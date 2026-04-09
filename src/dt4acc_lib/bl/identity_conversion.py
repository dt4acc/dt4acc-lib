from dt4acc_lib.interfaces.utils.state_conversion import StateConversion


class IdentityConversion(StateConversion):
    def inverse(self, state: object) -> object:
        return state

    def forward(self, state: object) -> object:
        return state


# as a singleton
identity_conversion = IdentityConversion()


__all__ = ["identity_conversion", "IdentityConversion"]