# tests/test_tango_resource_locator.py
import pytest
from dataclasses import FrozenInstanceError

from dt4acc_lib.model.utils.tango_resource_locator import TangoResourceLocator, clear_token


def test_from_trl_valid_simple():
    trl = "DOMAIN/Fam/Member1"
    obj = TangoResourceLocator.from_trl(trl)
    # stored exactly as provided by your class (no normalization in your current impl)
    assert obj.domain == "DOMAIN"
    assert obj.family == "Fam"
    assert obj.member == "Member1"
    # as_trl and __str__ should return the same canonical TRL
    assert obj.as_trl() == "DOMAIN/Fam/Member1"
    assert str(obj) == obj.as_trl()


def test_from_trl_invalid_parts_raises():
    bad_values = [
        "",                 # empty
        "too/many/parts/x", # 4 parts
        "onlyonepart",      # 1 part
        "two/parts",        # 2 parts
        "/leading/slash",   # 2 meaningful parts (empty first)
    ]
    for bad in bad_values:
        with pytest.raises(AssertionError):
            TangoResourceLocator.from_trl(bad)


def test_json_compatible_and_clear_token():
    # tokens contain dots which clear_token should convert to underscores
    domain = "a.b"
    family = "c.d"
    member = "e.f"
    obj = TangoResourceLocator(domain=domain, family=family, member=member)
    j = obj.json_compatible()
    # Each dot replaced by underscore, parts joined by "__"
    assert j == "a_b__c_d__e_f"
    # direct clear_token check
    assert clear_token("one.two.three") == "one_two_three"
    # ensure no accidental extra separators
    assert "__" in j and j.count("__") == 2


def test_equality_case_insensitive_and_hash():
    a = TangoResourceLocator(domain="Lab", family="Power", member="01")
    b = TangoResourceLocator(domain="lab", family="power", member="01")
    assert a == b
    # identical hash for equal objects
    assert hash(a) == hash(b)
    # can be used as dict keys and in sets
    s = {a: "value"}
    assert s[b] == "value"
    assert a in set([b])


def test_inequality_with_other_types_returns_false():
    trl = TangoResourceLocator(domain="D", family="F", member="M")
    # comparing to unrelated type should not raise, but be False
    assert (trl == "D/F/M") is False
    assert (trl != "D/F/M") is True


def test_frozen_immutable_dataclass():
    trl = TangoResourceLocator(domain="X", family="Y", member="Z")
    with pytest.raises(FrozenInstanceError):
        trl.domain = "new"  # cannot assign because frozen
