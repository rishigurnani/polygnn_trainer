import numpy as np
from polygnn_trainer import scale, utils
from torch import nn

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
        attrs = utils.sorted_attrs(self)
        attrs_str = "; ".join([f"{k}: {v}" for k, v in attrs])
        return "{" + attrs_str + "}"

    def set_values_from_string(
        self,
        string,
        extras={
            "leaky_relu": nn.functional.leaky_relu,
            "kaiming_normal_": nn.init.kaiming_normal_,
            "kaiming_uniform_": nn.init.kaiming_uniform_,
            "xavier_uniform_": nn.init.xavier_uniform_,
            "xavier_normal_": nn.init.xavier_normal_,
        },
    ):
        string = string.replace("{", "").replace("}", "")
        attrs_list = string.split("; ")
        dictionary = {}
        for attr in attrs_list:
            name, value = tuple(attr.split(": ", 1))
            pass_type = getattr(self, name).pass_type
            if pass_type == scale.SequentialScaler:
                scaler = scale.SequentialScaler()
                scaler.from_string(value)
                value = scaler
            elif pass_type == callable:
                value = extras[value]
            else:
                if value == "None":
                    value = None
                else:
                    value = pass_type(value)
            dictionary[name] = value

        self.set_values(dictionary)

    def __eq__(self, __o: object) -> bool:
        return str(self) == str(__o)
