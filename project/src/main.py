import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid, TUDataset
from torch_geometric.nn import GCNConv
# from torch_geometric.nn.models import GraphUNet
from torch_geometric.utils import dropout_adj

from network import GraphUNet

# https://github.com/pyg-team/pytorch_geometric/blob/master/examples/graph_unet.py


class Net(torch.nn.Module):
    def __init__(self, num_features, num_classes, num_nodes,
                 hidden_channels=128, depth=4, without_augment_adj=False,
                 without_pool_unpool=False):
        super().__init__()
        pool_ratios = [2000. / num_nodes, 0.5, 0.5, 0.4]
        self.unet = GraphUNet(num_features, hidden_channels, num_classes,
                              depth=depth, pool_ratios=pool_ratios,
                              without_augment_adj=without_augment_adj,
                              without_pool_unpool=without_pool_unpool)

    def forward(self, data):
        edge_index, _ = dropout_adj(data.edge_index, p=0.2,
                                    force_undirected=True,
                                    num_nodes=data.num_nodes,
                                    training=self.training)
        x = F.dropout(data.x, p=0.92, training=self.training)
        x = self.unet(x, edge_index)
        return F.log_softmax(x, dim=1)


def train(model, optimizer, data):
    model.train()
    optimizer.zero_grad()
    F.nll_loss(model(data)[data.train_mask],
               data.y[data.train_mask]).backward()
    optimizer.step()


def test(model, data):
    model.eval()
    logits, accs = model(data), []
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs


def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Cora',
                        choices=['Cora', 'CiteSeer', 'PubMed'])
    parser.add_argument('--hidden_channels', type=int, default=128)
    parser.add_argument('--depth', type=int, default=4)
    parser.add_argument('--without_augment', action='store_true')
    parser.add_argument('--without_pool_unpool', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dataset = Planetoid(root=os.path.join(
        'data', args.dataset), name=args.dataset)
    data = dataset[0].to(device)
    model = Net(dataset.num_features, dataset.num_classes,
                data.num_nodes, hidden_channels=args.hidden_channels,
                depth=args.depth, without_augment_adj=args.without_augment,
                without_pool_unpool=args.without_pool_unpool).to(device)

    # print(get_parameter_number(model))
    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.01, weight_decay=0.001)

    best_val_acc = test_acc = 0
    for epoch in range(1, 201):
        train(model, optimizer, data)
        train_acc, val_acc, tmp_test_acc = test(model, data)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            test_acc = tmp_test_acc
        if args.verbose:
            print(f'Epoch: {epoch:03d}, Train: {train_acc:.4f}, '
                  f'Val: {best_val_acc:.4f}, Test: {test_acc:.4f}')

    print(f'Test Acc: {test_acc:.4f}')
