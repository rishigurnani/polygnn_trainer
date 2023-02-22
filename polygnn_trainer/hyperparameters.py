import numpy as np
from .utils import sorted_attrs

passable_types = {
    int: [int, np.int64],
    float: [float],
}


class Parameter:
    def __init__(self, pass_type, value=None, options=None):
        self.pass_type = pass_type
        self.options = options
        if value == None:
            self.value = None
        else:
            self.set_value(value)

    def set_value(self, value):
        if self.pass_type == callable:
            if callable(value):
                self.value = value
            else:
                raise TypeError(f"The value passed in is not callable.")
        elif any(
            [
                isinstance(value, valid_type)
                for valid_type in passable_types[self.pass_type]
            ]
        ):
            self.value = self.pass_type(value)
        else:
            raise TypeError(
                f"The value passed in is of type {type(value)} but it should be of type {self.pass_type}"
            )

    def get_value(self):
        return self.value

    def __str__(self):
        if self.pass_type == callable:
            if self.value:
                return self.value.__name__
            else:
                return str(None)
        else:
            return str(self.value)


class ModelParameter(Parameter):
    def __init__(self, pass_type, value=None):
        super().__init__(pass_type, value=value)


class HpConfig:
    """
    A class to configure which hyperparameters to use during training
    """

    def __init__(self):
        super().__init__()
        # below are "standard" hyoerparameters that should always be set
        self.capacity = ModelParameter(int)
        self.batch_size = Parameter(int)
        self.r_learn = Parameter(float)
        self.dropout_pct = ModelParameter(float)
        self.activation = ModelParameter(callable)

    def set_values(self, dictionary):
        """
        Create attributes from "dictionary". One attribute will be
        created per key in "dictionary", only if the attribute
        already exists. Otherwise, it will be ignored.
        """
        for key, val in dictionary.items():
            if hasattr(self, key):
                param = getattr(self, key)
                param.set_value(val)

    def __str__(self):
        attrs = sorted_attrs(self)
        attrs_str = ", ".join([f"{k}: {v}" for k, v in attrs])
        return "{" + attrs_str + "}"

    def __eq__(self, __o: object) -> bool:
        return str(self) == str(__o)
