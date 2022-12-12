from torch import nn, tensor
from torch import save as torch_save
from torch import float as torch_float
from rdkit import Chem
from rdkit.Chem import AllChem
from torch_geometric.data import Data
import numpy as np

from polygnn_trainer.std_module import StandardModule

n_features = 512  # hard-coded


class MathModel1(StandardModule):
    def __init__(self, hps, dummy_attr):
        super().__init__(hps)
        self.hps = hps
        self.dummy_attr = dummy_attr

    def forward(self, data):
        x = self.assemble_data(data)

        return x[:, 0] + x[:, 1] + x[:, 2] - x[:, 3]


class MathModel2(StandardModule):
    def __init__(self, hps):
        super().__init__(hps)
        self.hps = hps

    def forward(self, data):
        x = self.assemble_data(data)

        return x[:, 0] - x[:, 1]


def trainer_MathModel(model, train_pts, val_pts, scaler_dict, tc, *args, **kwargs):
    # this function will "train" the model and save it
    if tc.model_save_path:
        torch_save(model.state_dict(), tc.model_save_path)
        print("Best model saved", flush=True)


def morgan_featurizer(smile):
    smile = smile.replace("*", "H")
    mol = Chem.MolFromSmiles(smile)
    fp = AllChem.GetMorganFingerprintAsBitVect(
        mol, radius=2, nBits=n_features, useChirality=True
    )
    fp = np.expand_dims(fp, 0)
    return Data(x=tensor(fp, dtype=torch_float))
