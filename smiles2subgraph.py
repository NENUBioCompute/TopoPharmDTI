from collections import defaultdict
import os
import pickle
import sys

import numpy as np

from rdkit import Chem
from rdkit.Chem import ChemicalFeatures


def create_atoms(mol):
    """Create a list of atom (e.g., hydrogen and oxygen) IDs
    considering the aromaticity."""
    atoms = [a.GetSymbol() for a in mol.GetAtoms()]

    for a in mol.GetAromaticAtoms():
        i = a.GetIdx()
        atoms[i] = (atoms[i], 'aromatic')
    # add by tangcy 20200907   added the donor and acceptor features
    # fdefName = os.path.join(RDConfig.RDDataDir,'BaseFeatures.fdef')
    # factory = ChemicalFeatures.BuildFeatureFactory(fdefName)
    factory = ChemicalFeatures.BuildFeatureFactory('BaseFeatures.fdef')


    feats = factory.GetFeaturesForMol(mol)
    # print(len(feats))  # 16


    for f in feats:
        ids = f.GetAtomIds()
        ffam = f.GetFamily()
        ftype = f.GetType()
        for d in ids:
            atoms[d] = (atoms[d], ffam)
        # print(
        #     f.GetFamily(),
        #     f.GetType(),
        #     f.GetAtomIds()
        #  )
    # end ------ add by tangcy 20200907   added the donor and acceptor features
    atoms = [atom_dict[a] for a in atoms]
    return np.array(atoms)


def create_ijbonddict(mol):
    """Create a dictionary, which each key is a node ID
    and each value is the tuples of its neighboring node
    and bond (e.g., single and double) IDs."""
    i_jbond_dict = defaultdict(lambda: [])
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        bond = bond_dict[str(b.GetBondType())]
        i_jbond_dict[i].append((j, bond))
        i_jbond_dict[j].append((i, bond))
    return i_jbond_dict


def extract_fingerprints(atoms, i_jbond_dict, radius):
    """Extract the r-radius subgraphs (i.e., fingerprints)
    from a molecular graph using Weisfeiler-Lehman algorithm."""

    if (len(atoms) == 1) or (radius == 0):
        fingerprints = [fingerprint_dict[a] for a in atoms]

    else:
        nodes = atoms
        i_jedge_dict = i_jbond_dict

        for _ in range(radius):

            """Update each node ID considering its neighboring nodes and edges
            (i.e., r-radius subgraphs or fingerprints)."""
            fingerprints = []
            for i, j_edge in i_jedge_dict.items():
                neighbors = [(nodes[j], edge) for j, edge in j_edge]
                fingerprint = (nodes[i], tuple(sorted(neighbors)))
                fingerprints.append(fingerprint_dict[fingerprint])
            nodes = fingerprints

            """Also update each edge ID considering two nodes
            on its both sides."""
            _i_jedge_dict = defaultdict(lambda: [])
            for i, j_edge in i_jedge_dict.items():
                for j, edge in j_edge:
                    both_side = tuple(sorted((nodes[i], nodes[j])))
                    edge = edge_dict[(both_side, edge)]
                    _i_jedge_dict[i].append((j, edge))
            i_jedge_dict = _i_jedge_dict

    return np.array(fingerprints)


def create_adjacency(mol):
    adjacency = Chem.GetAdjacencyMatrix(mol)
    return np.array(adjacency)


def split_sequence(sequence, ngram):
    sequence = '-' + sequence + '='
    words = [word_dict[sequence[i:i + ngram]]
             for i in range(len(sequence) - ngram + 1)]
    return np.array(words)


def dump_dictionary(dictionary, filename):
    with open(filename, 'wb') as f:
        pickle.dump(dict(dictionary), f)


if __name__ == "__main__":

    # DATASET, radius, ngram = sys.argv[1:]
    # radius, ngram = map(int, [radius, ngram])
    #
    # with open('../dataset/' + DATASET + '/original/data.txt', 'r') as f:
    #     data_list = f.read().strip().split('\n')
    #
    # """Exclude data contains '.' in the SMILES format."""
    # data_list = [d for d in data_list if '.' not in d.strip().split()[0]]
    # N = len(data_list)

    atom_dict = defaultdict(lambda: len(atom_dict))
    bond_dict = defaultdict(lambda: len(bond_dict))
    fingerprint_dict = defaultdict(lambda: len(fingerprint_dict))
    edge_dict = defaultdict(lambda: len(edge_dict))
    word_dict = defaultdict(lambda: len(word_dict))

    # Smiles, compounds, adjacencies, proteins, interactions = '', [], [], [], []
    #
    # for no, data in enumerate(data_list):
    #     print('/'.join(map(str, [no + 1, N])))
    #     print(data)
    #     smiles, sequence, interaction = data.strip().split()
    #     Smiles += smiles + '\n'
    #
    #     mol = Chem.AddHs(Chem.MolFromSmiles(smiles))  # Consider hydrogens.
    #     atoms = create_atoms(mol)
    #     i_jbond_dict = create_ijbonddict(mol)
    #
    #     fingerprints = extract_fingerprints(atoms, i_jbond_dict, radius)
    #     print("fp:")
    #     print(fingerprints)
    #     compounds.append(fingerprints)
    #
    #     adjacency = create_adjacency(mol)
    #     adjacencies.append(adjacency)
    #
    #     words = split_sequence(sequence, ngram)
    #     proteins.append(words)
    #
    #     interactions.append(np.array([float(interaction)]))
    #
    # dir_input = ('../dataset/' + DATASET + '/input/'
    #                                        'radius' + str(radius) + '_ngram' + str(ngram))
    # os.makedirs(dir_input, exist_ok=True)
    #
    # with open(dir_input + 'Smiles.txt', 'w') as f:
    #     f.write(Smiles)
    # np.save(dir_input + 'compounds', compounds)
    # np.save(dir_input + 'adjacencies', adjacencies)
    # np.save(dir_input + 'proteins', proteins)
    # np.save(dir_input + 'interactions', interactions)
    # dump_dictionary(fingerprint_dict, dir_input + 'fingerprint_dict.pickle')
    # dump_dictionary(word_dict, dir_input + 'word_dict.pickle')
    #
    # print('The preprocess of ' + DATASET + ' dataset has finished!')
    import pandas as pd
    tmp_file = pd.read_csv('../MCPI_Chembl33/chembl_2933_compound_filter.csv')
    smiles = tmp_file['compound']
    compound_feature_subgraph = []
    edge_indices = []
    success_smile = []
    loser_smile = []
    print('Start')
    compound_feature_subgraph_dic = {}
    compound_edge_indices_subgraph = {}
    index = 0
    for smile in smiles:
        try:
            adj = []
            # mol = Chem.AddHs(Chem.MolFromSmiles(smile))
            mol = Chem.MolFromSmiles(smile)
            atoms = create_atoms(mol)
            i_jbond_dict = create_ijbonddict(mol)
            fingerprints = extract_fingerprints(atoms, i_jbond_dict, 2)
            # fingerprints = np.pad(fingerprints, (0, 142 - len(fingerprints)), mode='constant')

            # if len(atoms)!=len(i_jbond_dict):
            #     loser_smile.append(smile)
            #
            # else:
            #     success_smile.append(smile)
            adjacency = create_adjacency(mol)

            ##########################
            compound_feature_subgraph_dic[index]=fingerprints

            for i in range(len(adjacency[0])):
                atom = adjacency[i]
                for j in range(len(atom)):
                    if atom[j] == 1:
                        adj.append([i, j])
            # edge_indices.append(adj)
            compound_edge_indices_subgraph[index]=adj

            index+=1

        except IndexError:
            # loser_smile.append(smile)
            print(smile)
    # compound_feature_subgraph = np.array(compound_feature_subgraph)
    # np.save('chembl_29_compound_feature_subgraph.npy', compound_feature_subgraph)
    import pickle
#     with open('chembl_29_compound_feature_subgraph.pkl', 'wb') as f:
#         pickle.dump(compound_feature_subgraph_dic, f)

#     with open('chembl_29_edge_indices_subgraph.pkl', 'wb') as f:
#         pickle.dump(compound_edge_indices_subgraph, f)
        
    regular_dict = dict(fingerprint_dict)
    with open('fingerprints_dic.pkl', 'wb') as f:
        pickle.dump(regular_dict, f)

    print(len(fingerprint_dict))
    ###筛选
    # succ_series = pd.Series(success_smile)
    # lose_series = pd.Series(loser_smile)
    # succ_series.to_csv('succ_smile.csv', index=True, header=['compound'])
    # lose_series.to_csv('lose_smile.csv', index=True, header=['compound'])
