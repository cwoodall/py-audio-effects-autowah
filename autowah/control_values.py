from dataclasses import dataclass
from multiprocessing import Lock, Value
from typing import Any, List


@dataclass
class ControlValueDef:
    name: str
    typestr: str = "f"
    init_value: Any = 0.0
    min: float = 0
    max: float = 1


class ControlValue:
    def __init__(self, cvd: ControlValueDef):
        self.lock = Lock()
        self.defn = cvd
        self.val = Value(cvd.typestr, cvd.init_value)

    @property
    def value(self):
        with self.lock:
            return self.val.value

    @value.setter
    def value(self, val):
        with self.lock:
            self.val.value = val

    @property
    def name(self):
        return self.defn.name

    @property
    def typestr(self):
        return self.defn.typestr

    @property
    def init_value(self):
        return self.defn.init_value

    @property
    def min(self):
        return self.defn.min

    @property
    def max(self):
        return self.defn.max


class ControlValues:
    def __init__(self, values: List[ControlValueDef]):
        self.lock = Lock()
        self.values = {x.name: ControlValue(x) for x in values}
