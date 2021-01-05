import random

import numpy as np
import pandas as pd
import torch
import yaml
from pandas.io.json._normalize import nested_to_record
from rdkit import Chem, DataStructs
import rdkit.Chem.rdMolDescriptors as rdMolDescriptors


class color:
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'


def feature_one_hot_encoding(value, allowable_set):
    """
     Create one-hot encoding for the feature. Values not in the allowable set
     are mapped to the last element.
    :param value: Feature value [any].
    :param allowable_set: Set of features seen in the train data [set].
    :return:
    """
    encoding = list(map(lambda s: value == s, allowable_set))
    if not any(encoding):
        encoding += [True]
    else:
        encoding += [False]
    return encoding


def set_deterministic(seed):
    """
    Set seeds for numpy, torch and CUDA.
    :param seed:
    :return:
    """
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def save_checkpoint(model, optimizer, history, parameters, results_test, path):
    """
    Helper function to save checkpoints.
    :param model:
    :param optimizer:
    :param history:
    :param parameters:
    :param results_test:
    :param path:
    :return:
    """
    # Save torch checkpoint
    model_state = {'epoch':                history[-1]['epoch'],
                   'model_state_dict':     model.state_dict(),
                   'optimizer_state_dict': optimizer.state_dict(),
                   'train_history':        history,
                   'train_loss':           history[-1]['train_loss'],
                   'test_loss':            history[-1]['val_loss']}

    # Creates a new folder numbered depending on existing [path/number/name]
    torch.save({'parameters': parameters,
                'model':      model_state},
               path / 'best_model.pth')

    # Save YAML file with parameters (without objects)
    training_state = {'timestamp':  path.name,
                      'epoch':      history[-1]['epoch'],
                      'train_loss': history[-1]['train_loss'],
                      'val_loss':   history[-1]['val_loss'],
                      'test_MAE':   float(results_test.MAE.mean()),
                      'test_MAPE':  float(results_test.MAPE.mean())}
    yaml.dump({'parameters':     parameters,
               'training_state': training_state},
              open(path / 'parameters.yaml', 'w'))

    # Save csv with history
    results_test.to_csv(path / 'results_test.csv', index=True)
    return


def previous_models_summary(models_path):
    """
    Create dataframe with previous models test accuracy and parameters.
    :param models_path:
    :param task:
    :return:
    """
    params = []
    for path in models_path.glob('**'):
        path = path / 'parameters.yaml'
        if path.exists():
            params.append(yaml.safe_load(open(path, 'r')))

    records = [nested_to_record(p) for p in params]
    records = pd.DataFrame(records)
    records = records.loc[:, records.apply(pd.Series.nunique) > 1]
    records = records.sort_values(by='training_state.test_MAPE')
    records = records.set_index('training_state.timestamp')
    records.to_csv(models_path / 'models_summary.csv', index_label='timestamp')
    return


def get_mol_similarity(mol1, mol2):
    """
    Helper function to check outliers.
    :param mol1: RDKit mol [
    :param mol2:
    :return:
    """
    fps = [Chem.RDKFingerprint(x) for x in [mol1, mol2]]
    return DataStructs.FingerprintSimilarity(fps[0], fps[1])


def get_most_similar(dataset, smiles, n=10):
    """
    Get the n most similar molecules in the dataset to check outliers.
    :param dataset:
    :param smiles:
    :param n:
    :return:
    """
    mol = Chem.MolFromSmiles(smiles)
    temp = dataset.copy()
    temp['sim'] = temp.mol.apply(lambda x: get_mol_similarity(mol, x))
    temp.sort_values('sim', ascending=False, inplace=True)
    return [{'idCAS':  idx,
             'SMILES': row.SMILES,
             'value':  row.y_true,
             'y_pred': row.y_pred,
             'sim':    row.sim} for idx, row in temp[:n].iterrows()]


def is_aromatic(mol):
    for atom in mol.GetAtoms():
        if atom.GetIsAromatic():
            return True
        else:
            return False


def match_substructure(mol, sub):
    matches = mol.GetSubstructMatches(Chem.MolFromSmarts(sub))
    if matches:
        return True
    else:
        return False


def is_organic(mol):
    for atom in mol.GetAtoms():
        if atom.GetSymbol() == 'C':
            return True
    return False


