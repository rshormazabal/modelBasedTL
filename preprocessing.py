from pathlib import Path

from pqdm.processes import pqdm
from rdkit.Chem import AllChem, PandasTools
from torch_geometric.data import Data

from utils import *


def get_node_features(atom, use_chirality, implicit_h, hybridization_allowed_set):
    """
    Calculate features for a single node.
    :param atom: Atom object [rdkit.Chem.rdchem.Atom].
    :param implicit_h: Whether hydronges are considered nodes in the graph or implicit [bool].
    :param use_chirality: Whether or not use atom chirality as a feature [bool].
    :param hybridization_allowed_set: Set of allowed values for hybridization [set].
    :return: node features [list].
    """
    features = [atom.GetAtomicNum(),
                atom.GetDegree(),
                atom.GetImplicitValence(),
                atom.GetFormalCharge(),
                atom.GetIsAromatic(),
                atom.IsInRing()]

    hybridization = feature_one_hot_encoding(atom.GetHybridization(), hybridization_allowed_set)
    features.extend(hybridization)

    # Add chirality a one-hot feature in the form [isChiral, R, S]
    if use_chirality:
        if atom.HasProp('_CIPCode'):
            cip = atom.GetProp('_CIPCode')
            cip_one_hot = [0, 1, 0] if cip == 'R' else [0, 0, 1]
            features += cip_one_hot
        else:
            features += [1, 0, 0]

    # If hydrogens are implicit, add feature for total number of hydrogens connected to the node
    if implicit_h:
        features += [atom.GetTotalNumHs()]

    return features


def get_bond_features(bond, use_chirality, bond_stereo_allowed_set):
    """
    Calculate features for a single bond.
    :param bond: Bond object [rdkit.Chem.rdchem.Bond].
    :param use_chirality: Whether or not use atom chirality as a feature [bool].
    :param bond_stereo_allowed_set: Set of allowed values for bond stereo [set].
    :return: bond features [list].
    """
    bond_type = bond.GetBondType()
    features = [bond_type == Chem.rdchem.BondType.SINGLE,
                bond_type == Chem.rdchem.BondType.DOUBLE,
                bond_type == Chem.rdchem.BondType.TRIPLE,
                bond_type == Chem.rdchem.BondType.AROMATIC,
                bond.GetIsConjugated(),
                bond.IsInRing()]

    # Add chirality as a one-hot feature.
    if use_chirality:
        features.extend(feature_one_hot_encoding(bond.GetStereo(), bond_stereo_allowed_set))

    return features


def mol_to_graph(mol, label, idcas, use_chirality, implicit_h, hybridization_allowed_set, bond_stereo_allowed_set):
    """
    Creates PyGeometric graph from RDKit mol object and label.
    :param mol: RDKit mol object [rdkit.Chem.rdchem.Mol].
    :param label: Label for the regression task [float].
    :param use_chirality: Whether or not use atom chirality as a feature [bool].
    :param implicit_h: Whether hydronges are considered nodes in the graph or implicit [bool].
    :param hybridization_allowed_set: Set of allowed values for hybridization [set].
    :param bond_stereo_allowed_set: Set of allowed values for bond stereo [set].
    :return: PyGeometric graph [torch_geometric.data.data.Data].
    """
    # Add hydrogens to calculate 3D configuration
    mol = Chem.AddHs(mol)
    conf_id = AllChem.EmbedMolecule(mol)

    if implicit_h:
        mol = Chem.RemoveHs(mol)

    # Calculate node positions. In case conf_id == -1, the geometry optimization
    # failed and pos calculation is skipped.
    pos = mol.GetConformer(conf_id).GetPositions() if conf_id != -1 else np.zeros((mol.GetNumAtoms(), 3))

    # Connectivity in COO format and edge features
    edge_index = []
    edge_attr = []
    for bond in mol.GetBonds():
        # Atoms connected in both directions [undirected graph]
        edge_index.extend([[bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()],
                           [bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()]])

        # Edge features, one time for each direction.
        bond_feats = get_bond_features(bond, use_chirality, bond_stereo_allowed_set)
        edge_attr.extend([bond_feats, bond_feats])

    # Node features
    nodes_features = [get_node_features(atom,
                                        use_chirality,
                                        implicit_h,
                                        hybridization_allowed_set) for atom in mol.GetAtoms()]

    return Data(x=torch.tensor(nodes_features, dtype=torch.float),
                edge_index=torch.tensor(edge_index, dtype=torch.long).t().contiguous(),
                edge_attr=torch.tensor(edge_attr, dtype=torch.float),
                pos=torch.tensor(pos, dtype=torch.float),
                y=torch.tensor([label], dtype=torch.float),
                idcas=torch.tensor([idcas], dtype=torch.long))


class Preprocessing():
    """
    Main preprocessing class. Takes two file paths, to the CAS-SMILES data and labels data.
    Generates PyGeometric graphs.
    :param labels_path:
    :param smiles_path:
    :param use_chirality: Whether or not use atom chirality as a feature [bool].
    :param implicit_h: Whether hydronges are considered nodes in the graph or implicit [bool].
    :param n_jobs:
    :param root_path:
    """

    def __init__(self,
                 labels_path,
                 smiles_path,
                 inference_file,
                 use_chirality,
                 implicit_h,
                 n_jobs,
                 root_path='./'):

        # get task name from file
        self.task_name = labels_path.split('.')[-2].split('/')[-1]

        # Check if raw files exist
        smiles_path = Path(root_path + smiles_path)
        labels_path = Path(root_path + labels_path)
        assert labels_path.exists() and smiles_path.exists(), 'SMILES or labels CSV does not exist.'

        # Load data [CAS ID to SMILES, regression labels]
        cas_to_smiles = pd.read_csv(smiles_path, index_col='idCAS')
        labels = pd.read_csv(labels_path, index_col='idCAS')

        # Get SMILES from compound idCAS
        self.dataset = pd.merge(labels, cas_to_smiles, left_index=True, right_index=True)
        self.dataset.drop_duplicates(keep='first', inplace=True)

        # TODO: TEMPORARY FIX
        self.inference_file = inference_file
        if self.inference_file:
            inference_idcas = pd.read_csv('./results/raw/gcnn_dataset.csv', index_col='idCAS')
            self.inf_dataset = pd.merge(inference_idcas, cas_to_smiles, left_index=True, right_index=True)
            self.inf_dataset.drop_duplicates(keep='first', inplace=True)

        # Allowable sets for featurization
        self.allowable_sets = dict()

        # Create mol objects and drop mono and diatomic molecules.
        print(color.BLUE + f"Creating mol objects for {self.task_name} dataset" + color.END)
        self.get_mol_objects()

        if self. inference_file:
            self.get_allowable_sets(self.inf_dataset)
        else:
            self.get_allowable_sets(self.dataset)

        # Parallelized calculation of graphs
        print(color.BLUE + f"Creating PyG graphs" + color.END)
        assert 'label' in self.dataset.columns, '"label" column is missing'
        args = [{'mol':                       row.mol,
                 'label':                     row.label,
                 'idcas':                     int(index),
                 'use_chirality':             use_chirality,
                 'implicit_h':                implicit_h,
                 'hybridization_allowed_set': self.allowable_sets['hybridization'],
                 'bond_stereo_allowed_set':   self.allowable_sets['bond_stereo']} for index, row in
                self.dataset.iterrows()]
        self.graphs = pqdm(args, mol_to_graph, n_jobs=n_jobs, argument_type='kwargs')

    def get_mol_objects(self):
        """
        Append mol objects to dataset.
        :return:
        """
        PandasTools.AddMoleculeColumnToFrame(self.dataset, "SMILES", molCol='mol')
        # TODO: CHECK IF WE CAN SOFTEN THIS
        self.dataset = self.dataset[self.dataset.mol.apply(lambda x: x.GetNumAtoms()) > 2]

        if self.inference_file:
            PandasTools.AddMoleculeColumnToFrame(self.inf_dataset, "SMILES", molCol='mol')
            self.inf_dataset = self.inf_dataset[self.inf_dataset.mol.apply(lambda x: x.GetNumAtoms()) > 2]
        return

    def get_allowable_sets(self, dataset):
        """
        Calculates allowable_sets for one-hot features.
        :return:
        """
        # Just to avoid verbose loops
        one_hot_features = ['hybridization', 'bond_stereo']
        instance_type = [lambda x: x.GetAtoms(),
                         lambda x: x.GetBonds()]
        prop_methods = [lambda x: x.GetHybridization(),
                        lambda x: x.GetStereo()]

        # Get allowed feature values for all atoms/bonds in the train data
        for feature, method, ins_type in zip(one_hot_features, prop_methods, instance_type):
            print(color.BLUE + f"Calculating allowable set for {feature}" + color.END)
            unique_values = set()

            for mol in dataset.mol:
                unique_values.update(map(method, ins_type(mol)))
            self.allowable_sets[feature] = unique_values
        return
