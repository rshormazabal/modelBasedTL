from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch_geometric
from sklearn.model_selection import train_test_split
from torch.utils.data import sampler
from torch_geometric.data import DataLoader
from tqdm import trange

import utils


class GCNN_S2S(torch.nn.Module):
    """
    Graph convolutional network for the regression task. Uses a GRU as update
    function for the node states and a  Set2Set layer for the Readout phase.
    :param node_features_dim: Dimension of the nodes feature vector [int].
    :param edge_features_dim: Dimension of the edges feature vector [int].
    :param hidden_dim: Dimension of the nodes hidden representations [int].
    :param aggr: Aggregation method [str]. {'mean', 'add, 'max'}
    :param s2s_processing_steps: Processing steps in Set2Set layer [int].
    :param message_passing_steps: Message passing steeps [int].
    :param dropout: Dropout for regression layers [float].
    """

    def __init__(self,
                 node_features_dim,
                 edge_features_dim,
                 hidden_dim,
                 aggr='mean',
                 s2s_processing_steps=3,
                 message_passing_steps=3,
                 dropout=0.2):
        super(GCNN_S2S, self).__init__()

        assert aggr in {'add', 'mean', 'max'}, 'Aggration method must be one of {"add", "mean", "max"}'
        assert 0 <= dropout < 1, 'Dropout percentage must be between [0, 1['

        # message passing steps
        self.message_passing_steps = message_passing_steps

        # main projection of node features h^{t}_{v}
        self.dense0 = torch.nn.Sequential(torch.nn.Linear(node_features_dim, hidden_dim),
                                          torch.nn.ReLU())

        # projection for edges to matrix A_{e_vw}
        nn_edges = torch.nn.Sequential(torch.nn.Linear(edge_features_dim, 2 * hidden_dim),
                                       torch.nn.ReLU(),
                                       torch.nn.Linear(2 * hidden_dim, hidden_dim ** 2))
        # gcnn calculating messages
        self.nn_conv = torch_geometric.nn.NNConv(hidden_dim, hidden_dim, nn=nn_edges, aggr=aggr)

        # GRU for node update
        self.gru = torch.nn.GRU(hidden_dim, hidden_dim)

        # Set2Set layer for Readout phase
        self.set2set = torch_geometric.nn.Set2Set(hidden_dim, processing_steps=s2s_processing_steps)

        # Final regression layers
        self.nn_reg = torch.nn.Sequential(torch.nn.Linear(2 * hidden_dim, hidden_dim),
                                          torch.nn.BatchNorm1d(hidden_dim),
                                          torch.nn.ReLU(),
                                          torch.nn.Dropout(dropout),
                                          torch.nn.Linear(hidden_dim, 1))

    def forward(self, data, get_representations=False):
        """
        Forward pass of the network.
        :param data: Graph data [pyGeometric graph object].
        :param get_representations: If set to True forward pass return hidden
               representations without going through the Readout phase [bool].
        :return:
        """
        out = self.dense0(data.x)
        # Unsqueeze to match GRU input dimensions.
        h = out.unsqueeze(0)

        for i in range(self.message_passing_steps):
            # Message passing step. Gives nodes hidden representations, edge indexes and edge attributes.
            m = torch.nn.functional.relu(self.nn_conv(out, data.edge_index, data.edge_attr))
            # Unsqueeze to match GRU input dimensions.
            m = m.unsqueeze(0)
            out, h = self.gru(m, h)
            # Squeeze to match nn_conv dimensions.
            out = out.squeeze(0)

        # Readout phase.
        out = self.set2set(out, data.batch)

        # if get_representations, returns graph hidden representation after pooling.
        if get_representations:
            return out

        # Fully connected layers for regression task.
        out = self.nn_reg(out)
        return out.view(-1)


class GCNN_global_add_pool(torch.nn.Module):
    """
    Graph convolutional network for the regression task. Uses a GRU as update
    function for the node states and a Global Attention layer for the Readout phase.
    :param node_features_dim: Dimension of the nodes feature vector [int].
    :param edge_features_dim: Dimension of the edges feature vector [int].
    :param hidden_dim: Dimension of the nodes hidden representations [int].
    :param aggr: Aggregation method [str]. {'mean', 'add, 'max'}
    :param message_passing_steps: Message passing steeps [int].
    :param dropout: Dropout for regression layers [float].
    """

    def __init__(self,
                 node_features_dim,
                 edge_features_dim,
                 hidden_dim,
                 aggr='mean',
                 message_passing_steps=3,
                 dropout=0.2):
        super(GCNN_global_add_pool, self).__init__()

        assert aggr in {'add', 'mean', 'max'}, 'Aggration method must be one of {"add", "mean", "max"}'
        assert 0 <= dropout < 1, 'Dropout percentage must be between [0, 1['

        # message passing steps
        self.message_passing_steps = message_passing_steps

        # main projection of node features h^{t}_{v}
        self.dense0 = torch.nn.Sequential(torch.nn.Linear(node_features_dim, hidden_dim),
                                          torch.nn.ReLU())

        # projection for edges to matrix A_{e_vw}
        nn_edges = torch.nn.Sequential(torch.nn.Linear(edge_features_dim, 2 * hidden_dim),
                                       torch.nn.ReLU(),
                                       torch.nn.Linear(2 * hidden_dim, hidden_dim ** 2))
        # gcnn calculating messages
        self.nn_conv = torch_geometric.nn.NNConv(hidden_dim, hidden_dim, nn=nn_edges, aggr=aggr)

        # GRU for node update
        self.gru = torch.nn.GRU(hidden_dim, hidden_dim)

        # Final regression layers
        self.nn_reg = torch.nn.Sequential(torch.nn.BatchNorm1d(hidden_dim),
                                          torch.nn.Linear(hidden_dim, hidden_dim),
                                          torch.nn.BatchNorm1d(hidden_dim),
                                          torch.nn.ReLU(),
                                          torch.nn.Dropout(dropout),
                                          torch.nn.Linear(hidden_dim, 1))

    def forward(self, data, get_representations=False):
        """
        Forward pass of the network.
        :param data: Graph data [pyGeometric graph object].
        :param get_representations: If set to True forward pass return hidden
               representations without going through the Readout phase [bool].
        :return:
        """
        out = self.dense0(data.x)
        # Unsqueeze to match GRU input dimensions.
        h = out.unsqueeze(0)

        for i in range(self.message_passing_steps):
            # Message passing step. Gives nodes hidden representations, edge indexes and edge attributes.
            m = torch.nn.functional.relu(self.nn_conv(out, data.edge_index, data.edge_attr))
            # Unsqueeze to match GRU input dimensions.
            m = m.unsqueeze(0)
            out, h = self.gru(m, h)
            # Squeeze to match nn_conv dimensions.
            out = out.squeeze(0)

        # Readout phase.
        out = torch_geometric.nn.global_add_pool(out, data.batch)

        # if get_representations, returns graph hidden representation after pooling.
        if get_representations:
            return out

        # Fully connected layers for regression task.
        out = self.nn_reg(out)
        return out.view(-1)


class Net:
    """
    Main model class. Trains, validates
    :param dataset: Takes dataset of type ThermophysicalPropertyRegression. Check dataset docs.
    :param model_args: Parameters to create GCNN model. Check model docs.
    :param optimizer_args: Parameters for Adam optimizer.
    :param training_args: Parameters for training.
    :param global_pooling_method: Type of global pooling {'att', 's2s'}.
    """

    def __init__(self, dataset, model_args, optimizer_args, training_args):
        # Save args in object
        self.model_args = model_args
        self.optimizer_args = optimizer_args
        self.training_args = training_args

        # Main network
        assert self.model_args['global_pooling_method'] in {'global_add',
                                                            's2s'}, "Global pooling must be one of {'att', 's2s'}"
        if self.model_args['global_pooling_method'] == 'global_add':
            self.model = GCNN_global_add_pool(node_features_dim=model_args['node_features_dim'],
                                              edge_features_dim=model_args['edge_features_dim'],
                                              hidden_dim=model_args['hidden_dim'],
                                              aggr=model_args['aggr'],
                                              message_passing_steps=model_args['message_passing_steps'],
                                              dropout=model_args['dropout']).to(model_args['device'])

        if self.model_args['global_pooling_method'] == 's2s':
            self.model = GCNN_S2S(node_features_dim=model_args['node_features_dim'],
                                  edge_features_dim=model_args['edge_features_dim'],
                                  hidden_dim=model_args['hidden_dim'],
                                  aggr=model_args['aggr'],
                                  s2s_processing_steps=model_args['s2s_processing_steps'],
                                  message_passing_steps=model_args['message_passing_steps'],
                                  dropout=model_args['dropout']).to(model_args['device'])

        # Dataset and loaders
        self.dataset = dataset
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

        self.split_traindata()

        # Sampler. (only shuffle train data if there is no sampler)
        self.sampler = None
        if training_args['n_bins']:
            weights = self.uniform_sampler(training_args['n_bins'])
            self.sampler = sampler.WeightedRandomSampler(weights, len(weights))
            self.prepare_loaders(shuffle=False)
        else:
            self.prepare_loaders(shuffle=True)

        # Optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=self.optimizer_args['lr'],
                                          weight_decay=self.optimizer_args['weight_decay'])

        # History
        self.history = []

        # Models path
        self.models_path = Path(f'./models/{self.dataset.task_args["name"]}/{self.dataset.task_args["type"]}')

        # New checkpoint folder name
        self.checkpoint_path = self.models_path / datetime.now().strftime('%Y_%m_%d_%H:%M:%S')
        self.checkpoint_path.mkdir(parents=True, exist_ok=True)

    def split_traindata(self):
        """
        Splits train data in [train,val].
        :return:
        """
        # get validation idx
        train_idx, val_idx = train_test_split(range(len(self.dataset.train_idx)),
                                              test_size=self.training_args['val_size'],
                                              random_state=self.training_args['random_state'])

        self.dataset.val_idx = self.dataset.train_idx[val_idx]
        self.dataset.train_idx = self.dataset.train_idx[train_idx]
        return

    def prepare_loaders(self, shuffle, inference=False):
        """
        Create dataloaders.
        :return:
        """
        if inference:
            self.sampler = None
        train_dataset = self.dataset[self.dataset.train_idx]
        val_dataset = self.dataset[self.dataset.val_idx]
        test_dataset = self.dataset[self.dataset.test_idx]
        self.train_loader = DataLoader(train_dataset,
                                       batch_size=self.training_args['batch_size'],
                                       sampler=self.sampler,
                                       shuffle=shuffle)
        self.val_loader = DataLoader(val_dataset, batch_size=self.training_args['batch_size'], shuffle=False)
        self.test_loader = DataLoader(test_dataset, batch_size=self.training_args['batch_size'], shuffle=False)

        return

    def train(self):
        """
        Main training loop. Uses MSE loss.
        :return:
        """

        # parameter to save on checkpoint
        parameters = {'task_args':          self.dataset.task_args,
                      'preprocessing_args': self.dataset.preprocessing_args,
                      'split_args':         self.dataset.split_args,
                      'model_args':         self.model_args,
                      'optimizer_args':     self.optimizer_args,
                      'training_args':      self.training_args}

        device = self.model_args['device']
        lowest_val_loss = np.inf
        for epoch in trange(self.training_args['epochs']):

            # Train data
            loss_all = 0
            self.model.train()
            for bath_idx, data in enumerate(self.train_loader):
                self.optimizer.zero_grad()
                output = self.model(data.to(device))
                loss = F.mse_loss(output.double(), data.y)
                loss.backward()
                loss_all += loss.item()
                self.optimizer.step()

            # Validation data
            self.model.eval()
            with torch.no_grad():
                y_pred = torch.cat([self.model(x.to(device)) for _, x in enumerate(self.val_loader)]).to('cpu')
                y_true = torch.cat([x['y'] for _, x in enumerate(self.val_loader)])
            val_loss = F.mse_loss(y_pred.double(), y_true).item()

            # Get values on original scale and calculate MAPE
            y_pred = self.dataset.standarizerY.inverse_transform(y_pred)
            y_true = self.dataset.standarizerY.inverse_transform(y_true)
            val_mape = (abs((y_pred - y_true) / y_true) * 100).mean()

            # Save loss in history
            self.history.append({"epoch":      epoch,
                                 "train_loss": loss_all / len(self.train_loader),
                                 "val_loss":   val_loss,
                                 "val_mape":   val_mape})
            # Verbose print
            print(f'train_loss: {self.history[-1]["train_loss"]:.4f}, val_loss: {val_loss:.4f}')

            # checkpoint (avoid saving every result when training starts)
            if (epoch > 50) and (val_loss < lowest_val_loss):
                print(utils.color.PURPLE + 'Saving checkpoint - Lowest validation loss' + utils.color.END)
                print(utils.color.BLUE + f'Previous: {lowest_val_loss:.4f} | New: {val_loss:.4f}' + utils.color.END)
                lowest_val_loss = val_loss

                # Save test set results
                print(utils.color.PURPLE + 'Test results' + utils.color.END)
                results_test = self.get_results(self.test_loader)
                utils.save_checkpoint(self.model,
                                      self.optimizer,
                                      self.history,
                                      parameters,
                                      results_test,
                                      self.checkpoint_path)

        # Save final model
        model_state = {'epoch':                self.history[-1]['epoch'],
                       'model_state_dict':     self.model.state_dict(),
                       'optimizer_state_dict': self.optimizer.state_dict(),
                       'train_history':        self.history,
                       'train_loss':           self.history[-1]['train_loss'],
                       'test_loss':            self.history[-1]['val_loss']}

        torch.save({'parameters': parameters,
                    'model':      model_state},
                   self.checkpoint_path / 'final_model.pth')

        # Create models summary file
        utils.previous_models_summary(self.models_path)
        return

    def get_results(self, loader, mol_objects=False):
        """
        Get dataframe with results for loader.
        :param mol_objects:
        :param loader:
        :return:
        """
        # Get info
        device = self.model_args['device']
        self.model.eval()

        # Get predictions
        with torch.no_grad():
            y_pred = torch.cat([self.model(x.to(device)) for batch_idx, x in enumerate(loader)]).to('cpu')
            y_true = torch.cat([x['y'] for batch_idx, x in enumerate(loader)])
            idcas = torch.cat([x['idcas'] for batch_idx, x in enumerate(loader)])

        # Back to original scale
        y_pred = self.dataset.standarizerY.inverse_transform(y_pred)
        y_true = self.dataset.standarizerY.inverse_transform(y_true)

        # Results dataframe
        results = pd.DataFrame({'SMILES': self.dataset.mol_info.loc[idcas, 'SMILES'],
                                'y_pred': y_pred,
                                'y_true': y_true,
                                'MAE':    abs(y_pred - y_true),
                                'MAPE':   abs((y_pred - y_true) / y_true) * 100})

        # Propred results (only for experimental)
        if self.dataset.split_args['split_mode'] == 'pp_bench':
            results['y_pp'] = self.dataset.pp_results.loc[idcas, 'y_pp']
            results = results[['SMILES', 'y_pred', 'y_pp', 'y_true', 'MAE', 'MAPE']]

        # RDKit mol objects
        if mol_objects:
            results['mol'] = self.dataset.mol_info.loc[idcas, 'mol']

        results.index.rename('idCAS')
        results.sort_values(by='MAE', ascending=False, inplace=True)
        print(utils.color.BLUE + f'MAE: {results.MAE.mean():.4f} | '
                                 f'MAPE: {results.MAPE.mean():.4f}' + utils.color.END)
        return results

    def get_features(self, loader):
        """
        Get feature features for compounds, forward only until before fully connected
        block.
        :param loader:
        :return:
        """

        # Get info
        device = self.model_args['device']
        self.model.eval()

        # Get predictions
        with torch.no_grad():
            features = []
            cas = []
            for batch_idx, x in enumerate(loader):
                f = self.model.forward(x.to(device), get_representations=True).to('cpu')
                features.append(f)
                cas.append((x['idcas']))

        features = torch.cat(features).numpy()
        cas = torch.cat(cas).tolist()
        representations = pd.DataFrame([{'idCAS':  c,
                                         'SMILES': s} for c, s in zip(cas, self.dataset.mol_info.loc[cas, 'SMILES'])])

        representations = pd.concat([representations, pd.DataFrame(features)], axis=1)
        return representations

    def uniform_sampler(self, n_bins):
        """
        Dataset sampler to oversample from less available molecules.
        Creates sampling weights depending on the label value.
        :return:
        """
        labels = self.dataset.standarizerY.inverse_transform(self.dataset.data.y[self.dataset.train_idx])
        bins = torch.linspace(labels.min(), labels.max(), n_bins)
        assignments = np.digitize(labels, bins=bins)
        _, counts = np.unique(assignments, return_counts=True)

        weights = [1 / counts[c - 1] for c in assignments]
        return weights

    def load_checkpoint(self, model_name, task_type):
        model_path = str(self.checkpoint_path.parents[0])

        if '/experimental/' in model_path:
            model_path = model_path.replace('/experimental/', f'/{task_type}/')
        if '/model/' in model_path:
            model_path = model_path.replace('/model/', f'/{task_type}/')

        model_info = torch.load(model_path + f'/{model_name}')
        self.model.load_state_dict(model_info['model']['model_state_dict'], strict=False)
        self.model.eval()
        return


random_list = [True, 0, "cute_string", False, -23, [], 7,
[{"how many": "booleans?"}]]
counts = {"integers": 0, "booleans": 0}

for item in random_list:
    if isinstance(item, int):
        counts["integers"] += 1
    elif isinstance(item, bool):
        counts["booleans"] += 1
print(counts["booleans"]) # you sure

