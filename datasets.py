import pickle
from pathlib import Path

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import InMemoryDataset
from torch.utils.data import sampler
import numpy as np

from preprocessing import Preprocessing


class ThermophysicalPropertyRegression(InMemoryDataset):
    """
    Main dataset class for thermophysical properties.
    :param root: [str].
    :param preprocessing_args: [dict].
    """

    def __init__(self, task_args, preprocessing_args, split_args):
        # Args for preprocessing [needs to be called before super to be used in overwritten methods]
        self.task_args = task_args
        self.preprocessing_args = preprocessing_args
        self.split_args = split_args
        super().__init__(f'data/{task_args["name"]}/{task_args["type"]}')

        # Called after process
        self.mol_info = pickle.load(open(self.processed_paths[1], 'rb'))
        self.data, self.slices = torch.load(self.processed_paths[0])

        # Splits index
        self.train_idx, self.val_idx, self.test_idx= None, None, None

        # Propred results
        if self.split_args['split_mode'] == 'pp_bench':
            self.pp_results = None
        self.split(self.split_args['split_mode'],
                   self.split_args['bench_threshold'],
                   self.split_args['fold_n'],
                   self.split_args['test_size'],
                   self.split_args['random_state'])

        # Normalization
        self.standarizerX = StandardScaler()
        self.standarizerY = StandardScaler()
        self.normalize_inputs()
        self.normalize_labels()

    @property
    def raw_file_names(self):
        return [f'{self.task_args["type"]}_{self.task_args["name"]}.csv']

    @property
    def processed_file_names(self):
        return ['graphs.pkl', 'mol_info.pkl']

    def download(self):
        pass

    def process(self):
        # get path and calculate graphs
        labels_path = self.raw_paths[0]
        prep = Preprocessing(labels_path,
                             self.preprocessing_args['smiles_path'],
                             self.preprocessing_args['inference_file'],
                             self.preprocessing_args['use_chirality'],
                             self.preprocessing_args['implicit_h'],
                             self.preprocessing_args['n_jobs'],
                             self.preprocessing_args['root_path'])

        # save graphs and mol info
        self.mol_info = prep.dataset
        self.mol_info.to_pickle(self.processed_paths[1])
        torch.save(self.collate(prep.graphs), self.processed_paths[0])

    def split(self, mode, bench_threshold=5, fold_n=5, test_size=0.2, random_state=0):
        """
        Splits data indices. Saves indices on self.train_idx and self.test_idx.
        pp_bench -> Splits dataset depending on propred error.
        CV -> cross validation.
        :param mode: Way of splitting between {'pp_bench', 'CV'} [str].
        :param bench_threshold: Propred benchmark threshold [float].
        :param cv_n: Number of folds for cross-validation [int].
        :param test_size:
        :param random_state:
        :return:
        """
        assert mode in {'pp_bench', 'random', 'CV', 'uncertainty'}, 'Mode has to be one of {"pp_bench"}'
        if mode == 'pp_bench':
            # propred benchmark only for experimental data.
            assert 'model' not in self.raw_paths[0], 'Propred benchmark cannot be used for model data.'

            # propred pre-calculated results to split.
            pp_results_path = Path(f"{self.raw_dir}/{self.task_name.replace('experimental', 'propred')}.csv")

            assert pp_results_path.exists(), 'ProPred files do not exists. Files needed for splitting'
            self.pp_results = pd.read_csv(pp_results_path, index_col='idCAS')

            idcas_train = self.pp_results[self.pp_results.MAPE < bench_threshold].index.tolist()
            idcas_test = self.pp_results[self.pp_results.MAPE >= bench_threshold].index.tolist()
            self.train_idx = torch.tensor([idx for idx in range(len(self)) if int(self[idx].idcas) in idcas_train])
            self.test_idx = torch.tensor([idx for idx in range(len(self)) if int(self[idx].idcas) in idcas_test])

        if mode == 'random':
            train_idx, test_idx = train_test_split(range(len(self)), test_size=test_size, random_state=random_state)
            self.train_idx = torch.tensor(train_idx)
            self.test_idx = torch.tensor(test_idx)

        if mode == 'uncertainty':
            pass
        return

    def normalize_inputs(self):
        """
        Normalize non-binary features of node vectors(mean-0 / std-1).
        Uses only training data, self.idx_train.
        :return:
        """
        # checking non-binary variables
        non_binary_variables = []
        for i in range(self.data.x.shape[1]):
            # only non-binary variables

            if torch.unique(self.data.x[:, i]).size()[0] > 2:
                # stack feature column vectors to standarize
                non_binary_variables.append(i)

        # features to transform
        self.standarizerX.fit(self[self.train_idx].data.x[:, non_binary_variables])

        # transform non-binary variables
        x = self.standarizerX.transform(self.data.x[:, non_binary_variables])
        self.data.x[:, non_binary_variables] = torch.tensor(x, dtype=torch.float)
        return

    def normalize_labels(self):
        """
        Normalize task labels (mean-0 / std-1).
        Uses only training data, self.idx_train.
        :return:
        """
        train_labels = self[self.train_idx].data.y.unsqueeze(1).numpy()
        self.standarizerY.fit(train_labels)
        self.data.y = torch.tensor(self.standarizerY.transform(self.data.y.unsqueeze(1))).squeeze()
        return


class InferenceDataset(InMemoryDataset):
    """
    Main dataset class for thermophysical properties.
    :param root: [str].
    :param preprocessing_args: [dict].
    """

    def __init__(self, path, cas_file, preprocessing_args, standarizer):
        # Args for preprocessing [needs to be called before super to be used in overwritten methods]
        self.preprocessing_args = preprocessing_args
        self.cas_file = cas_file
        super().__init__(path)

        # Called after process
        self.mol_info = pickle.load(open(self.processed_paths[1], 'rb'))
        self.data, self.slices = torch.load(self.processed_paths[0])

        # Normalization
        self.standarizerX = standarizer
        self.normalize_inputs()

    @property
    def raw_file_names(self):
        return [self.cas_file]

    @property
    def processed_file_names(self):
        return ['graphs.pkl', 'mol_info.pkl']

    def download(self):
        pass

    def process(self):
        # get path and calculate graphs
        cas_path = self.raw_paths[0]
        prep = Preprocessing(cas_path,
                             self.preprocessing_args['smiles_path'],
                             self.preprocessing_args['use_chirality'],
                             self.preprocessing_args['implicit_h'],
                             self.preprocessing_args['n_jobs'],
                             self.preprocessing_args['root_path'])

        # save graphs and mol info
        self.mol_info = prep.dataset
        self.mol_info.to_pickle(self.processed_paths[1])
        torch.save(self.collate(prep.graphs), self.processed_paths[0])

    def normalize_inputs(self):
        """
        Normalize non-binary features of node vectors(mean-0 / std-1).
        Uses only training data, self.idx_train.
        :return:
        """
        # checking non-binary variables
        non_binary_variables = []
        for i in range(self.data.x.shape[1]):
            # only non-binary variables

            if torch.unique(self.data.x[:, i]).size()[0] > 2:
                # stack feature column vectors to standarize
                non_binary_variables.append(i)

        # transform non-binary variables
        x = self.standarizerX.transform(self.data.x[:, non_binary_variables])
        self.data.x[:, non_binary_variables] = torch.tensor(x, dtype=torch.float)
        return


if __name__ == '__main__':
    preprocessing_args = {'smiles_path':   './data/CAS_SMILES.csv',
                          'use_chirality': True,
                          'implicit_h':    True,
                          'n_jobs':        1,
                          'root_path':     './'}

    split_args = {'split_mode':      'pp_bench',
                  'bench_threshold': 0.05,
                  'fold_n':          None,
                  'test_size':       0.2,
                  'random_state':    0}

    dataset = ThermophysicalPropertyRegression('data/experimental/critical_temperature',
                                               preprocessing_args,
                                               split_args)