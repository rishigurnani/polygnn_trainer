# These scalers are all implemented with PyTorch so that they can be used
# on a GPU, if necessary. These scalers are only written for use
# on ***labels***, not features
import ast
import torch
from .utils import sorted_attrs


ignore_attrs = ["index", "is_parent", "linear"]


class SequentialScaler:
    def __init__(self):
        """
        Keyword Arguments:
        scaler_ls - A list of objects with a transform and inverse_transform method.
        The first element should be the first transformation that occured, the second
        elemnts should be the second transformation that occured, etc.
        """
        self.scaler_ls = []  # the list of child scalers
        self.n_children = 0  # how many child scalers are there?
        self.index = 0  # the index of scaler in its parent's sequence
        self.is_parent = True

    def fit_transform(self, y):
        # fit child scalers
        sequence = sorted(self.scaler_ls, key=lambda x: x.index)
        for scaler in sequence:
            y = scaler.fit_transform(y)

        return y

    def transform(self, y):
        """
        Keyword Arguments:
        y - should be a 2D array-like
        """
        sequence = sorted(self.scaler_ls, key=lambda x: x.index)
        for scaler in sequence:
            y = scaler.transform(y)

        return y

    def inverse_transform(self, y):
        """
        Keyword Arguments:
        y - should be a 2D array-like
        """
        sequence = sorted(self.scaler_ls, key=lambda x: x.index, reverse=True)
        for scaler in sequence:
            y = scaler.inverse_transform(y)

        return y

    def append(self, scaler):
        self.scaler_ls.append(scaler)
        setattr(scaler, "index", self.n_children + 1)
        self.n_children += 1
        setattr(scaler, "is_parent", False)

    def is_linear(self):

        return all([scaler.is_linear() for scaler in self.scaler_ls])

    def __str__(self) -> str:
        string = "Forward: " + " --> ".join(str(scaler) for scaler in self.scaler_ls)

        return string

    def format_tensorlike(self, data):
        """
        Format 'data' as a torch tensor
        """
        if not isinstance(data, torch.Tensor):
            data = torch.tensor(data, dtype=torch.float)
        # do not store gradients because we do not need to compute dLoss/dy
        if data.is_leaf:
            data.requires_grad = False
        return data

    def from_string(self, string):
        # Strings prior to June 7th, 2022 contained a colon. The colon
        # was removed after this date.
        if "Forward: " in string:
            string = string.replace("Forward: ", "")
        else:
            if "Forward(" in string:
                string = string.replace("Forward(", "")
                string = string[:-1]  # remove the trailing ")"
        if string:
            str_list = string.split(" --> ")
            for individual_str in str_list:
                if "(" in individual_str:
                    left_paren_idx = individual_str.index("(")
                    scaler_name = individual_str[:left_paren_idx]
                    scaler_cls = globals()[scaler_name]
                    scaler = scaler_cls.from_string(individual_str)
                    self.append(scaler)
                else:
                    # If individual_str does not contain "(" then the
                    # corresponding scaler does not need any inputs to
                    # __init__.
                    scaler = globals()[individual_str]()
                    self.append(scaler)

    def __eq__(self, __o: object) -> bool:
        return str(self) == str(__o)


class Scaler:
    def __init__(self):
        """
        Keyword Arguments:
        scaler_ls - A list of objects with a transform and inverse_transform method.
        The first element should be the first transformation that occured, the second
        elemnts should be the second transformation that occured, etc.
        """
        self.linear = False  # should be True if the scaler is a linear
        # transformation (e.g., MinMax) and should if False if the
        # scaler is not a linear transformation (e.g., LogTen). As a
        # default, we will set this value to False.

    def fit(self, y):

        pass

    def fit_transform(self, y):
        self.fit(y)

        return self.transform(y)

    def transform(self, y):
        """
        Keyword Arguments:
        y - should be a 2D array-like
        """

        pass

    def inverse_transform(self, y):
        """
        Keyword Arguments:
        y - should be a 2D array-like
        """

        pass

    @classmethod
    def format_tensorlike(cls, data):
        """
        Format 'data' as a torch tensor.
        """
        if not isinstance(data, torch.Tensor):
            data = torch.tensor(data, dtype=torch.float)
        # do not store gradients because we do not need to compute dLoss/dy
        if data.is_leaf:
            data.requires_grad = False
        return data

    @classmethod
    def string_to_tensor(cls, string):
        """
        Format a tensor string to a tensor.
            Ex. string='tensor([1., 2.])'
        """
        return cls.format_tensorlike(
            ast.literal_eval(string.replace("tensor(", "").replace(")", ""))
        )

    def fmt_input(self, y):
        """
        Format 'y' as a torch tensor
        """
        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y, dtype=torch.float)
        # do not store gradients because we do not need to compute dLoss/dy
        if y.is_leaf:
            y.requires_grad = False
        return y

    def is_linear(self):
        return self.linear

    def __str__(self) -> str:
        attrs = sorted_attrs(self)
        attrs = [x for x in attrs if x[0] not in ignore_attrs]
        self_class_name = type(self).__name__
        if not attrs:
            return self_class_name
        else:
            attr_str = ", ".join([f"{k}: {v}" for k, v in attrs])
            return f"{self_class_name}({attr_str})"

    @classmethod
    def dim_from_string(cls, string):
        # get the "dim" value
        if "dim: " in string:
            # extract as dim as an integer-like string
            value = string.split("dim: ")[1].split(",")[0].replace(")", "")
            value = int(value)
        else:
            value = 0
        return value

    @classmethod
    def from_string(cls, string):
        """
        The default from_string method for Scaler objects. This method
        should be overwritten in child classes that contain
        arguments in their __init__ method.
        """
        return cls()


class ZeroMeanScaler(Scaler):
    def __init__(self):
        super().__init__()
        self.mean = None
        self.linear = True

    def fit(self, y):
        y = self.fmt_input(y)
        self.mean = torch.clone(y).mean()

    def transform(self, y):
        y = self.fmt_input(y)

        return y - self.mean

    def inverse_transform(self, y):
        """
        Keyword Arguments:
        y - should be a 2D array-like
        """
        y = self.fmt_input(y)
        if self.mean:
            return y + self.mean
        else:
            raise AttributeError(".fit() should be called before .inverse_transform()")

    @classmethod
    def from_string(cls, string):
        scaler = cls()
        # set "mean" attribute.
        value = string.split("mean: ")[1]  # tensor string
        value = cls.string_to_tensor(value)
        setattr(scaler, "mean", value)
        return scaler


class LogTenScaler(Scaler):
    def __init__(self):
        super().__init__()
        self.linear = False

    def transform(self, y):
        y = self.fmt_input(y)
        y = torch.log10(y)

        return y

    def inverse_transform(self, y):
        y = self.fmt_input(y)

        return 10**y


class LogTenDeltaScaler(Scaler):
    def __init__(self):
        super().__init__()
        self.linear = False

    def transform(self, y):
        y = self.fmt_input(y)
        y = torch.log10(y + 1)

        return y

    def inverse_transform(self, y):
        y = self.fmt_input(y)

        return (10**y) - 1


class MinMaxScaler(Scaler):
    def __init__(self):
        super().__init__()
        self.linear = True

    def fit(self, y):
        y = self.fmt_input(y)

        self.min = torch.clone(y).min()
        self.max = torch.clone(y).max()

    def transform(self, y):
        y = self.fmt_input(y)

        return (y - self.min) / (self.max - self.min)

    def inverse_transform(self, y):
        y = self.fmt_input(y)

        return y * (self.max - self.min) + self.min

    @classmethod
    def from_string(cls, string):
        scaler = cls()
        # set the "max" attribute
        max_value = string.split("max: ")[1].split("min: ")[0][
            :-2
        ]  # tensor-like string
        max_value = cls.string_to_tensor(max_value)
        setattr(scaler, "max", max_value)
        # set the "min" attribute
        min_value = string.split("min: ")[1]
        min_value = cls.string_to_tensor(min_value)
        setattr(scaler, "min", min_value)
        return scaler


class ProductScaler(Scaler):
    def __init__(self, multiplier):
        super().__init__()
        self.multiplier = multiplier
        self.linear = True

    def transform(self, y):
        y = self.fmt_input(y)

        return y * self.multiplier

    def inverse_transform(self, y):
        y = self.fmt_input(y)

        return y / self.multiplier

    @classmethod
    def from_string(cls, string):
        # get "multiplier" attribute
        value = string.split("multiplier: ")[1]  # tensor-like string
        value = cls.string_to_tensor(value)
        if cls == ProductScaler:
            return cls(
                multiplier=value,
            )
        elif cls == QuotientScaler:
            return cls(
                divisor=1 / value,
            )


class QuotientScaler(ProductScaler):
    def __init__(self, dividend):
        super().__init__(multiplier=(1 / dividend))


class DummyScaler:
    """
    This is a "Scaler" which just returns the object passed in.
    """

    def __init__(self):
        pass

    def transform(self, data):
        """
        Just return the data that is passed in
        """
        return data

    def inverse_transform(self, data):
        return data
