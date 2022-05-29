import inspect
from torch import nn, cat
from polygnn_trainer.constants import PACKAGE_NAME
from polygnn_trainer.hyperparameters import HpConfig, ModelParameter
from copy import deepcopy

from polygnn_trainer.utils import module_name


class StandardModule(nn.Module):
    def __init__(self, hps: HpConfig):
        super().__init__()
        hp_copy = deepcopy(hps)
        if hp_copy:
            # delete attributes that are not of type ModelParameter
            del_attrs = []
            for attr_name, obj in hp_copy.__dict__.items():
                if not isinstance(obj, ModelParameter):
                    # log those attributes that are not of
                    # type ModelParameter so we can delete
                    # them later. They need to be deleted
                    # later so that the dictionary size does
                    # not change during the for loop.
                    del_attrs.append(attr_name)
            for attr in del_attrs:
                delattr(hp_copy, attr)
        # assign hp_copy to self
        self.hps = hp_copy
        # We only want to print hps when a model is instantiated. So,
        # below we will check that "self" is indeed a model.
        if module_name(self) == f"{PACKAGE_NAME}.models":
            print(f"\nHyperparameters after model instantiation: {self.hps}")
            # we also want to make sure that "data" is the first argument
            # of the model's "forward" method
            named_args = inspect.getfullargspec(self.forward)[0]
            assert "data" in named_args

    def assemble_data(self, data):
        x, graph_feats, selector = data.x, data.graph_feats, data.selector
        x = cat((x, graph_feats, selector), dim=1)

        return x
