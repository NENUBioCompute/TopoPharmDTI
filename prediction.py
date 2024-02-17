
from data_loader import data_loader_chembl

from datetime import datetime
import os
import argparse
import torch
import numpy as np
import time
import torch.nn as nn
from torch.optim import Adam, RAdam
from cpi_model import CPI_model_1024, CPI_model_23, CPI_model_768
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, \
    average_precision_score,roc_curve,auc,precision_recall_curve
import pandas as pd
import csv
from torch.optim.lr_scheduler import ReduceLROnPlateau, MultiStepLR, StepLR
import random

def set_random_seed(seed, deterministic=False):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_time_str():
    now = datetime.now()
    month = now.month
    day = now.day
    hour = now.hour
    minute = now.minute

    formatted_str = f"{month:02d}_{day:02d}_{hour:02d}_{minute:02d}"
    return formatted_str

def batch_output(input_list, batch_idx):

    batched_output = []

    for i in batch_idx:
        batch = input_list[i]
        batched_output.append(batch)
    return batched_output


def predict(data_loader, model, device):
    model.eval()
    predictions = []
    true_labels = []

    logits_list = []
    for data in data_loader:
        compound_rdkit_data = data['compound_rdkit_data'].to(device)
        compound_sub_data = data['compound_sub_data'].to(device)
        compound_len_data = data['compound_len_data']


        edge_rdkit = data['edge_rdkit']
        edge_attr = data['edge_attr'].to(device)
        protein_data = data['protein_data'].to(device)
        protein_len_data = data['protein_len_data']
        labels = data['labels'].to(device)
        # Forward pass
        outputs = model(compound_rdkit_data,
                        compound_sub_data,
                        edge_rdkit,
                        edge_attr,
                        compound_len_data,
                        compound_len_data,
                        protein_data,
                        protein_len_data, protein_len_data, device)
        # Compute the loss
        # Convert logits to predictions
        preds = torch.argmax(outputs, dim=1)
        predictions.extend(preds.cpu().numpy())
        true_labels.extend(labels.cpu().numpy())
        logits_list.extend(outputs.cpu().detach().numpy())
    logits_array = np.array(logits_list)

    acc = accuracy_score(true_labels, predictions)
    precision1 = precision_score(true_labels, predictions)
    recall1 = recall_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions)
    roc_auc = roc_auc_score(true_labels, logits_array[:, 1])
    auprc = average_precision_score(true_labels, logits_array[:, 1])
    fpr, tpr, _ = roc_curve(true_labels, logits_array[:, 1])
    roc_auc = auc(fpr, tpr)

    # Calculate precision-recall curve and area
    precision, recall, _ = precision_recall_curve(true_labels, logits_array[:, 1])
    prc_auc = average_precision_score(true_labels, logits_array[:, 1])
    # Create a DataFrame using a dictionary, where keys become column names
    auc_fpr_tpr = pd.DataFrame({
        'fpr': fpr,
        'tpr': tpr
    })
    prc_pre_recall = pd.DataFrame({
        'precision': precision,
        'recall': recall
    })

    return acc, precision1, recall1, f1, roc_auc, auprc,auc_fpr_tpr,prc_pre_recall


def save_AUCs(AUCs, filename):
    with open(filename, 'a') as f:
        f.write('\t'.join(map(str, AUCs)) + '\n')


def save_model(model, filename):
    torch.save(model.state_dict(), filename)


def save_evalution_metrics(save_path, row, epoch, train_result=False):
    if train_result == False:

        with open(save_path, 'a', newline='') as file:
            writer = csv.writer(file)
            header = ['epoch', 'train_loss', 'test_acc', 'test_precision', 'test_recall', 'test_f1', 'test_roc_auc',
                      'test_auprc']
            if epoch == 0:
                writer.writerow(header)
            writer.writerow(row)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0,
                        help="which GPU to use. Set -1 to use CPU.")
    parser.add_argument("--batch-size", type=int, default=128,
                        help="loader small_data")
    parser.add_argument("--dataset", type=str, default='data',
                        help="small or big dataset")
    parser.add_argument("--epochs", type=int, default=200,
                        help="number of training epochs")
    parser.add_argument("--gat-head", type=int, default=4,
                        help="number of gat heads")
    parser.add_argument("--compound-head", type=int, default=4,
                        help="number of compound attention heads")
    parser.add_argument("--decoder-head", type=int, default=4,
                        help="number of decoder attention heads")
    parser.add_argument("--lr", type=float, default=0.0001,
                        help="learning rate")
    parser.add_argument("--seed", type=int, default=42,
                        help="random seed")
    parser.add_argument("--weight-decay", type=float, default=0.001,
                        help="weight-decay")
    parser.add_argument("--num-layers_compound_decoder", type=int, default=2,
                        help="num_layers_compound_decoder")
    parser.add_argument("--num-layers_protein_bilstm", type=int, default=1,
                        help="num_layers_protein_bilstm")
    parser.add_argument("--num-layers_decoder", type=int, default=2,
                        help="num_layers_decoder")

    args = parser.parse_args()

    lr = args.lr
    batch_size = args.batch_size
    epochs = args.epochs
    gpu = args.gpu
    seed = args.seed
    dataset = args.dataset
    decay = args.weight_decay
    gat_head = args.gat_head
    compound_head = args.compound_head
    decoder_head = args.decoder_head
    num_layers_compound_decoder = args.num_layers_compound_decoder
    num_layers_protein_bilstm = args.num_layers_protein_bilstm
    num_layers_decoder = args.num_layers_decoder
    # save
    save_dir = args.save_dir

    params = vars(args)
    set_random_seed(seed)
    device = torch.device(f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu')
    train_dataset, dev_dataset, test_dataset = data_loader_chembl(dataset)
    # Instantiate the models
    compound_input_size = 128
    output_size = 768
    protein_input_size = 768
    cpi_model = CPI_model_768(compound_input_size, protein_input_size, output_size, gat_head, compound_head,
                              decoder_head,
                              num_layers_compound_decoder,
                              num_layers_protein_bilstm,
                              num_layers_decoder, dataset, device)

    cpi_model.load_state_dict(torch.load('./model/model254.pth'))

    cpi_model.to(device)
    criterion = nn.CrossEntropyLoss()
    acc, precision, recall, f1, roc_auc, auprc,auc_fpr_tpr,prc_pre_recall = predict(test_dataset, cpi_model,device)
    print(acc,precision,recall,f1,roc_auc,auprc)
    auc_fpr_tpr.to_csv('./label_reversal_auc_prc_plot/' + 'cpi1'+ '_auc.csv', index=False)
    prc_pre_recall.to_csv('./label_reversal_auc_prc_plot/' + 'cpi1' + '_prc.csv', index=False)

