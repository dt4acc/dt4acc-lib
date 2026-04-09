from typing import Any, Dict, Union
import jsons


def deserialse_value(value: Dict[str, Union[str, int, float, dict]], **kwargs):
    def delegate_to_jsons(obj, **kwargs):
        return jsons.dump(obj, **kwargs)

    d = dict(int=int, float=float, dict=delegate_to_jsons)
    ins = d[value["type"]]
    r = ins(value["value"])
    return r


def serialize_value(value, **kwargs) -> dict[str, Any]:
    if isinstance(value, int):
        return dict(type="int", value=value)
    elif isinstance(value, float):
        return dict(type="float", value=value)

    d = jsons.dump(value, **kwargs)
    assert isinstance(d, dict), "don't know how to serialize {value}"
    return dict(type="dict", value=d)
