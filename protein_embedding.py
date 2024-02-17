import torch
import pickle
import numpy as np
import pandas as pd
from tape import ProteinBertModel, TAPETokenizer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def pad_or_cut_tensor(tensor, target_length=1024):
    current_length = tensor.size(1)
    if current_length < target_length:
        padding_length = target_length - current_length
        padding = torch.zeros((1, padding_length, tensor.size(2))).to(device)
        padded_tensor = torch.cat((tensor, padding), dim=1)
        return padded_tensor
    elif current_length > target_length:
        cut_tensor = tensor[:, :target_length, :]
        return cut_tensor
    else:
        return tensor


model = ProteinBertModel.from_pretrained('bert-base')
model = model.to(device)
tokenizer = TAPETokenizer(vocab='iupac')  # iupac is the vocab for TAPE models, use unirep for the UniRep model


def tape_bert(data):
    with torch.no_grad():
        token_ids = torch.tensor([tokenizer.encode(data)]).to(device)
        output = model(token_ids)
        sequence_output = output[0]
        sequence_output = sequence_output[:, 1:-1, :]
        sequence_output = torch.squeeze(sequence_output, 0).cpu()
    sequence_output = np.array(sequence_output)
    return sequence_output


# model = ProteinBertModel.from_pretrained('bert-base')
# tokenizer = TAPETokenizer(vocab='iupac')  # iupac is the vocab for TAPE models, use unirep for the UniRep model
embedding = {}
i = 0
with torch.no_grad():
    tmp_data = pd.read_csv('data/attention_protein_filter.csv')
    sequences = tmp_data['protein']
    for seq in sequences:
        sequence_output = tape_bert(seq)
        # sequence_output = pad_or_cut_tensor(sequence_output)
        embedding[i] = sequence_output
        i = i + 1
        print(i)
with open('data/attention_protein_filter_bert_768.pkl', 'wb') as file:
    pickle.dump(embedding, file)
