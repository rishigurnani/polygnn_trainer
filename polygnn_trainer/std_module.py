import inspect
from copy import deepcopy
from torch import nn, cat

from polygnn_trainer.constants import PACKAGE_NAME
from polygnn_trainer.hyperparameters import HpConfig, ModelParameter
from polygnn_trainer.utils import module_name


class StandardModule(nn.Module):
    """
    A standard module that extends nn.Module.
    """

    def __init__(self, hps: HpConfig):
        """
        Initialize the StandardModule.

        Args:
            hps (HpConfig): Hyperparameters configuration.
        """
        super().__init__()
        hp_copy = deepcopy(hps)

        if hp_copy:
            # Delete attributes that are not of type ModelParameter
            del_attrs = []
            for attr_name, obj in hp_copy.__dict__.items():
                if not isinstance(obj, ModelParameter):
                    # Log attributes that are not of type ModelParameter
                    # so they can be deleted later. They need to be deleted
                    # later to avoid changing the dictionary size during the loop.
                    del_attrs.append(attr_name)

            for attr in del_attrs:
                delattr(hp_copy, attr)

        # Assign the modified copy of hyperparameters to self.hps
        self.hps = hp_copy

        # Print hps when a model is instantiated
        if module_name(self) == f"{PACKAGE_NAME}.models":
            print(f"\nHyperparameters after model instantiation: {self.hps}")

            # Check if "data" is the first argument of the model's "forward" method
            named_args = inspect.getfullargspec(self.forward)[0]
            assert "data" in named_args

    def assemble_data(self, data):
        """
        Assemble data by concatenating yhat, graph_feats, and selector.

        Args:
            data: Input data.

        Returns:
            Tensor: Concatenated data tensor.
        """
        return cat((data.yhat, data.graph_feats, data.selector), dim=1)
