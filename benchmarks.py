import sklearn
import rdkit.Chem as Chem
import numpy as np
import pandas as pd
import xgboost as xgb
from rdkit.Chem import AllChem
from rdkit.Chem.AtomPairs import Pairs

class XGBoost:
    def __init__(self):


class SVM:
    def __init__(self):
        pass


class DenseNN:
    def __init__(self):
        pass

class LinearReg:
    def __init__(self):
        pass

class Bench():
    def __init__self(self, dataset):
        self.dataset = dataset
        pass

    def get_features(self, smiles, feature_types):
        """

        :param mol:
        :param feature_types:
        :return:
        """
        features = []
        mol = Chem.MolFromSmiles(smiles)
        if 'morgan_fp' in feature_types:
            features.extend(AllChem.GetMorganFingerprintAsBitVect(mol))

        return

