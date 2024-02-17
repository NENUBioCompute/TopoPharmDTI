from data_loader import data_loader_chembl
import time
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
    average_precision_score
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
    # 将提取的信息格式化为用下划线隔开的字符串
    formatted_str = f"{month:02d}_{day:02d}_{hour:02d}_{minute:02d}"
    return formatted_str

def batch_output(input_list, batch_idx):
    batched_output = []

    for i in batch_idx:
        batch = input_list[i]
        batched_output.append(batch)
    return batched_output


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


def train(model, num_epochs, batch_size, lr, decay, train_dataset, dev_dataset, test_dataset, criterion, device,
          save_path):
    optimizer = RAdam(model.parameters(), lr=lr, weight_decay=decay)
    dev_save_path = os.path.join(save_path, f"dev_result.csv")
    # scheduler = MultiStepLR(optimizer, milestones=[10, 20, 30], gamma=0.5)
    scheduler = StepLR(optimizer, step_size=40, gamma=0.5)
    dev_auc_history = 0
    best_models = []
    for epoch in range(num_epochs):
        model.train()
        start_time = time.time()
        total_loss = 0
        N = len(train_dataset)
        i = 0
        optimizer.zero_grad()
        for data in train_dataset:
            compound_rdkit_data = data['compound_rdkit_data'].to(device)
            compound_sub_data = data['compound_sub_data'].to(device)
            compound_len_data = data['compound_len_data']
            idx = data['idx']
            # print(idx)
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
            i = i + 1
            loss = criterion(outputs, labels)
            loss = loss / batch_size
            loss.backward()
            if i % batch_size == 0 or i == N:
                optimizer.step()
                optimizer.zero_grad()
            total_loss += loss.item()
        end_time = time.time()
        epoch_time = end_time - start_time

        # print every epoch time
        print(f'Epoch [{epoch + 1}/{num_epochs}], 'f'Epoch Time: {epoch_time:.2f} seconds', f'Loss: {total_loss:.4f}')

        with torch.no_grad():
            predictions = []
            true_labels = []

            logits_list = []
            for data in dev_dataset:
                compound_rdkit_data = data['compound_rdkit_data'].to(device)
                compound_sub_data = data['compound_sub_data'].to(device)
                compound_len_data = data['compound_len_data']
                idx = data['idx']
                # print(idx)
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
            # Calculate validation metrics

            dev_acc = accuracy_score(true_labels, predictions)
            dev_precision = precision_score(true_labels, predictions)
            dev_recall = recall_score(true_labels, predictions)
            dev_f1 = f1_score(true_labels, predictions)
            dev_roc_auc = roc_auc_score(true_labels, logits_array[:, 1])
            dev_auprc = average_precision_score(true_labels, logits_array[:, 1])
            dev_save_data = [epoch, total_loss, dev_acc, dev_precision, dev_recall, dev_f1, dev_roc_auc,
                              dev_auprc]
            print(
                "dev accuaray:{:.4f}; dev f1_score:{:.4f}; dev recall:{:.4f}; dev precision:{:.4f};dev auc:{:.4f};dev auprc:{:.4f}".format(
                    dev_acc,
                    dev_f1,
                    dev_recall,
                   dev_precision, dev_roc_auc, dev_auprc))

            save_evalution_metrics(dev_save_path, dev_save_data, epoch, train_result=False)
            if dev_roc_auc > dev_auc_history:

                dev_auc_history = dev_roc_auc
                checkpoint_path = os.path.join(checkpoints_path, f'model{epoch}.pth')
                torch.save(model.state_dict(), checkpoint_path)
                best_models.append(os.path.join(checkpoints_path, f'model{epoch}.pth'))
                if len(best_models) > 1:
                    model_to_delete = best_models.pop(0)
                    os.remove(model_to_delete)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0,
                        help="which GPU to use. Set -1 to use CPU.")
    parser.add_argument("--batch-size", type=int, default=128,
                        help="loader small_data")
    parser.add_argument("--dataset", type=str, default='../MCPI_Chembl33',
                        help="small or big dataset")
    parser.add_argument("--epochs", type=int, default=200,
                        help="number of training epochs")
    parser.add_argument("--gat-head", type=int, default=4,
                        help="number of gat heads")
    parser.add_argument("--compound-head", type=int, default=4,
                        help="number of compound attention heads")
    parser.add_argument("--decoder-head", type=int, default=4,
                        help="number of decoder attention heads")
    parser.add_argument("--lr", type=float, default=0.00001,
                        help="learning rate")
    parser.add_argument("--seed", type=int, default=42,
                        help="random seed")
    parser.add_argument("--weight-decay", type=float, default=0.0001,
                        help="weight-decay")
    parser.add_argument("--num-layers_compound_decoder", type=int, default=2,
                        help="num_layers_compound_decoder")
    parser.add_argument("--num-layers_protein_bilstm", type=int, default=1,
                        help="num_layers_protein_bilstm")
    parser.add_argument("--num-layers_decoder", type=int, default=2,
                        help="num_layers_decoder")

    parser.add_argument('--save-dir', type=str, default='./result',
                        help="save dir")
    args = parser.parse_args()

    # train_config
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

    # save path
    save_dir = args.save_dir

    time_str = get_time_str()
    save_path = os.path.join(save_dir,
                             f"lr_{lr}_batch_{batch_size}_epochs{epochs}_decay{decay}_gathead{gat_head}_compoundhead{compound_head}decoder_head{decoder_head}decoderNumber{num_layers_compound_decoder}{num_layers_protein_bilstm}{num_layers_decoder}_time{time_str}")
    checkpoints_path = os.path.join(save_path, f"./checkpoints")
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(checkpoints_path, exist_ok=True)
    config_save_path = os.path.join(save_path, f"parameters.json")
    params = vars(args)
    # save parameter
    with open(config_save_path, 'w') as file:
        for key, value in params.items():
            file.write(f'{key}: {value}\n')

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

    cpi_model.to(device)
    criterion = nn.CrossEntropyLoss()
    train(
        model=cpi_model,
        num_epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        decay=decay,
        train_dataset=train_dataset,
        dev_dataset=dev_dataset,
        test_dataset=test_dataset,
        criterion=criterion,
        device=device,
        save_path=save_path
    )
