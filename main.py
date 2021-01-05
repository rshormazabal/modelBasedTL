import matplotlib.pyplot as plt
import pandas as pd
import rdkit.Chem as Chem
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import rdkit.Chem.rdMolDescriptors as rdMolDescriptors
import rdkit.Chem.Descriptors as Desriptors

import datasets
import networks
from utils import is_aromatic, match_substructure, set_deterministic, is_organic

# set deterministic seed
random_state = 0
set_deterministic(random_state)

# dataset and split parameters
task_args = {'type': 'experimental',
             'name': 'critical_temperature'}
preprocessing_args = {'smiles_path':    './data/CAS_SMILES.csv',
                      'inference_file': None,
                      'use_chirality':  True,
                      'implicit_h':     True,
                      'n_jobs':         1,
                      'root_path':      './'}

split_args = {'split_mode':      'random',
              'bench_threshold': None,
              'fold_n':          None,
              'test_size':       0.1,
              'random_state':    random_state}

dataset = datasets.ThermophysicalPropertyRegression(task_args,
                                                    preprocessing_args,
                                                    split_args)

# model parameters
model_args = {'node_features_dim':     dataset.num_node_features,
              'edge_features_dim':     dataset.num_edge_features,
              'global_pooling_method': 'global_add',
              's2s_processing_steps':  3,
              'hidden_dim':            64,
              'aggr':                  'mean',
              'message_passing_steps': 3,
              'dropout':               0.3,
              'device':                'cuda'}
optimizer_args = {'lr':           5e-4,
                  'weight_decay': 1e-3}
training_args = {'batch_size':   128,
                 'epochs':       250,
                 'val_size':     0.1,
                 'n_bins':       None,
                 'random_state': random_state}

net = networks.Net(dataset, model_args, optimizer_args, training_args)
net.train()

# Results
# Set shuffle to False on loaders
net.prepare_loaders(shuffle=False, inference=True)
results_train = net.get_results(net.train_loader, mol_objects=True)
results_val = net.get_results(net.val_loader, mol_objects=True)
results_test = net.get_results(net.test_loader, mol_objects=True)

results_all = pd.concat([results_train, results_test, results_val])
results_all.sort_values('MAE', ascending=False, inplace=True)

# features
features_all = pd.concat(pd.DataFrame(net.get_features(x)) for x in [net.train_loader,
                                                                     net.val_loader,
                                                                     net.test_loader])

features_all = features_all.set_index('idCAS')
features_all['y_true'] = results_all.loc[features_all.index, 'y_true']

# Linear vs aromatic
tsne = TSNE(n_components=2)
features_all['tsne1'], features_all['tsne2'] = tsne.fit_transform(features_all.iloc[:, 1:limit_col].values).T

for plot_col in features_all.columns[limit_col-2:-2]:

    ax = sns.scatterplot(x="tsne1", y="tsne2", hue=plot_col, palette='coolwarm', data=features_all)
    if features_all[plot_col].nunique() > 6:
        norm = plt.Normalize(features_all[plot_col].min(), features_all[plot_col].max())
        sm = plt.cm.ScalarMappable(cmap="coolwarm", norm=norm)
        sm.set_array([])

        # Remove the legend and add a colorbar
        ax.get_legend().remove()
        ax.figure.colorbar(sm)

    else:
        ax.legend()

    plt.title(f'{plot_col} - nAtoms from {min_atoms} to {max_atoms}')
    plt.savefig(f'./results/images/{plot_col}_{min_atoms}_{max_atoms-1}.svg', format='svg', dpi=1200)
    plt.show()
    plt.gcf()

plt.scatter(features_all['tsne1'], features_all['tsne2'], c=features_all.n_oxygens.values)
plt.show()

plt.scatter(features_all['pca1'], features_all['pca2'], c=features_all.n_carbons.values)
plt.show()

# Ring vs aromatic

# model summary
models_summary = pd.read_csv(f'./models/{task_args["name"]}/{task_args["type"]}/models_summary.csv',
                             index_col='timestamp')

# loss graphs
history = pd.DataFrame(net.history)
sns.lineplot(data=history[['train_loss', 'val_loss']])
plt.show()
