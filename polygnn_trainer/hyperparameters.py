import numpy as np
from polygnn_trainer import scale, utils, constants
from torch import nn

passable_types = {
    int: [int, np.int64],
    float: [float],
    scale.SequentialScaler: [scale.SequentialScaler],
}


class identity(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x):
        return x


class Parameter:
    def __init__(self, pass_type, value=None, options=None):
        self.pass_type = pass_type
        self.options = options
        if value == None:
            self.value = None
        else:
            self.set_value(value)

    def set_value(self, value):
        if self.pass_type == str:
            if value in self.options:
                self.value = value
            else:
                raise TypeError(
                    f"The value passed in is '{value}' but the only valid options are {self.options}."
                )
        elif self.pass_type == scale.SequentialScaler:
            self.value = value
        elif self.pass_type == callable:
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
        self.set_value_callbacks()

    def get_value(self):
        return self.value

    def set_value_callbacks(self):
        """
        This function will be run after set_value
        """
        pass

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
        # initialize an activation
        self.activation = ModelParameter(callable)
        self.activation.set_value(nn.functional.leaky_relu)
        # initialize a normalization method
        self.norm = ModelParameter(callable)
        self.norm.set_value(identity)
        # initialize an initialization method
        self.initializer = Initializer()
        self.initializer.set_value(nn.init.xavier_uniform_)
        # initialize the optimizer
        self.optimizer = Parameter(str, options=["adam", "swa"])
        self.optimizer.set_value("adam")
        # initialize swa parameters
        self.swa_start_frac = Parameter(float)  # between 0 and 1
        self.swa_start_frac.set_value(0.0)
        self.swa_freq = Parameter(int)
        self.swa_freq.set_value(0)
        # initialize a weight decay value
        self.weight_decay = Parameter(float)
        self.weight_decay.set_value(0.0)
        # initialize a scaler for graph features
        scaler = Parameter(scale.SequentialScaler)
        scaler.set_value(scale.SequentialScaler())
        graph_scaler_name = f"{constants._F_GRAPH}_scaler"
        setattr(self, graph_scaler_name, scaler)
        # initialize a scaler for node features
        scaler = Parameter(scale.SequentialScaler)
        scaler.set_value(scale.SequentialScaler())
        node_scaler_name = f"{constants._F_NODE}_scaler"
        setattr(self, node_scaler_name, scaler)

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
            "identity": identity,
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


# #########################################
# Code related to parameter initialization
# #########################################
class Initializer(ModelParameter):
    def __init__(self):
        super().__init__(callable)

    def set_value_callbacks(self):
        # In PyTorch, functions that end in "_" denote operations that
        # modify tensors in place. For initializers, we only support these
        # type of in-place operations.
        assert self.value.__name__.endswith("_")
