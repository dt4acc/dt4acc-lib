"""Support of Tango Resource Locator

Tango resource locators seem to need a bit extra treatment
e.g. Bluesky expects them to be compatible with json names

accml needs a bit further investigation to fully support
TRL with all available tools without treating TRL specially
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class TangoResourceLocator:
    """ Lightweight Tango Resource Locator (TRL) for device names only.

    This class intentionally represents *only* the core Tango device name
    consisting of exactly three components:

        domain / family / member

    Todo:
       review whole stack to find where trl can not be used as
       string

       or to say differently that TRL can be used as strings.
       These are only made json compatible where imposed by
       (external) modules

    Limitations and design choices
    -------------------------------
    * Only the device name part of a TRL is supported.
      Full TRL features such as protocol (tango://), host:port,
      attributes, properties (->), database selectors (#dbase),
      or wildcards are deliberately NOT handled here.

    * The class is designed to be a small, explicit value object and
      not a full TRL parser.

    * Input tokens are assumed to be already valid Tango name tokens.
      No automatic escaping, sanitization, or normalization is performed
      except where explicitly documented.

    * Users are responsible for providing JSON-compatible tokens
      when serialization is required. The helper method
      `json_compatible()` exists for this purpose but is not lossless.

    * Tango device names are case-insensitive. Equality comparisons
      are therefore performed in a case-insensitive manner.

    Rationale
    ---------
    This restricted scope avoids implicit behavior, keeps the object
    predictable, and makes failures explicit. Any logic related to full
    TRL parsing or environment-specific resolution should live at a
    higher level in the application stack.
    """

    domain: str
    family: str
    member: str

    @classmethod
    def from_trl(cls, trl: str):
        tmp = trl.split("/")
        assert len(tmp) == 3, (
                "Only simple device TRLs of the form 'domain/family/member'"
                f' are supported. I received "{trl}" which does not split in three'
            )
        for cnt, token in enumerate(tmp):
            assert token != "", f'trl "{trl}" split in "{tmp}", but token {cnt} is empty'
        domain, family, member = tmp
        return cls(domain=domain, family=family, member=member)

    def as_trl(self) -> str:
        r = "/".join([self.domain, self.family, self.member])
        return r

    def json_compatible(self) -> str:
        return "__".join(map(clear_token, [self.domain, self.family, self.member]))

    def __str__(self):
        return self.as_trl()

    def __eq__(self, other):
        if not isinstance(other, TangoResourceLocator):
            return NotImplemented
        return (
            self.domain.lower(),
            self.family.lower(),
            self.member.lower(),
        ) == (
            other.domain.lower(),
            other.family.lower(),
            other.member.lower(),
        )

    def __hash__(self):
        return hash((
            self.domain.lower(),
            self.family.lower(),
            self.member.lower(),
        ))


def clear_token(token):
    """
    Todo:
        jsons complains on this name
    """
    return token.replace(".", "_")


def name_from_trl(trl):
    """
    Todo:
        not used ... remove me
    """
    prefix, middle, suffix = map(clear_token, trl.split("/"))
    return "__".join([prefix, middle, suffix])
