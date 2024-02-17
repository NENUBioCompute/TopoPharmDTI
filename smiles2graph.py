from rdkit.Chem import AllChem, rdMolDescriptors
from rdkit.Chem import FunctionalGroups
from rdkit import Chem
import numpy as np
import networkx as nx
import pickle


def smi2feats(smi, max_smi_len=102):
    smi = smi.replace(' ', '')

    X = [START_TOKEN]
    for ch in smi[: max_smi_len - 2]:
        X.append(SMI_CHAR_DICT[ch])
    X.append(END_TOKEN)
    X += [PAD_TOKEN] * (max_smi_len - len(X))
    X = np.array(X).astype(np.int64)
    return X

SMI_CHAR_DICT = {"(": 1, ".": 2, "0": 3, "2": 4, "4": 5, "6": 6, "8": 7, "@": 8,
                "B": 9, "D": 10, "F": 11, "H": 12, "L": 13, "N": 14, "P": 15, "R": 16,
                "T": 17, "V": 18, "Z": 19, "\\": 20, "b": 21, "d": 22, "f": 23, "h": 24,
                "l": 25, "n": 26, "r": 27, "t": 28, "#": 29, "%": 30, ")": 31, "+": 32,
                "-": 33, "/": 34, "1": 35, "3": 36, "5": 37, "7": 38, "9": 39, "=": 40,
                "A": 41, "C": 42, "E": 43, "G": 44, "I": 45, "K": 46, "M": 47, "O": 48,
                "S": 49, "U": 50, "W": 51, "Y": 52, "[": 53, "]": 54, "a": 55, "c": 56,
                "e": 57, "g": 58, "i": 59, "m": 60, "o": 61, "s": 62, "u": 63, "y": 64,
                ":": 65, "*": 66, "|": 67,
                }
assert np.all(np.array(sorted(list(SMI_CHAR_DICT.values()))) == np.arange(1, len(SMI_CHAR_DICT) + 1))
PAD_TOKEN = 0
START_TOKEN = len(SMI_CHAR_DICT) + 1
END_TOKEN = START_TOKEN + 1
assert PAD_TOKEN not in SMI_CHAR_DICT and START_TOKEN not in SMI_CHAR_DICT.values() and END_TOKEN not in SMI_CHAR_DICT
SMI_CHAR_SET_LEN = len(SMI_CHAR_DICT) + 3  # + (PADDING, START, END)


def atom_features(atom):
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),
                                          ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As',
                                           'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se',
                                           'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr',
                                           'Pt', 'Hg', 'Pb', 'Unknown']) +
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    [atom.GetIsAromatic()])


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def smile_to_graph(smile):
    mol = Chem.MolFromSmiles(smile)
    # mol = Chem.AddHs(mol)
    c_size = mol.GetNumAtoms()

    features = []
    for atom in mol.GetAtoms():
        feature = atom_features(atom)
        features.append(feature / sum(feature))

    edges = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
    g = nx.Graph(edges).to_directed()
    edge_index = []
    for e1, e2 in g.edges:
        edge_index.append([e1, e2])

    return c_size, features, edge_index

def get_functional_groups(smiles):
    mol = Chem.MolFromSmiles(smiles)

    func_groups = FunctionalGroups.BuildFuncGroupHierarchy()
    matches = FunctionalGroups.MatchFunctionalGroups(mol, func_groups)

    results = []
    for match in matches:
        functional_group = {
            "Functional Group": match.GetProp("groupName"),
            "SMARTS": match.GetProp("SMARTS"),
            "Atoms": match.GetProp("atoms")
        }
        results.append(functional_group)

    return results

import pandas as pd
file = pd.read_csv('chembl_29_compound_filter.csv')
smiles = file['canonical_smiles']
compound_len = []
compound_feature_rdkit = []
edge_indices = []
for smile in smiles:
    c_size, features, edge_index = smile_to_graph(smile)
    compound_len.append(c_size)
max_len = max(compound_len)
compound_len=[]
print('max_len:'+str(max_len))
index = 0
compound_len_dic ={}
compound_feature_dic = {}
compound_edge_index_dic = {}
for smile in smiles:
    c_size, features, edge_index = smile_to_graph(smile)
    compound_len_dic[index]=c_size
    compound_feature_dic[index] = features
    compound_edge_index_dic[index] = edge_index
    index+=1
    # compound_len.append(c_size)
    # features = np.pad(np.array(features), [(0, max_len-c_size), (0, 0)], mode='constant')

    # compound_feature_rdkit.append(features)

    # edge_indices.append(edge_index)
compound_feature_rdkit = np.array(compound_feature_rdkit)
print(len(compound_len))
print(len(edge_indices))
print(compound_feature_rdkit.shape)

with open('chembl_29_compound_len.pkl', 'wb') as f:
    pickle.dump(compound_len_dic, f)

with open('chembl_29_compound_feature.pkl', 'wb') as f:
    pickle.dump(compound_feature_dic, f)
with open('chembl_29_edge_indices_rdkit.pkl', 'wb') as f:
    pickle.dump(compound_edge_index_dic, f)
