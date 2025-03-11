#!/usr/bin/env python
import warnings

import torch
import torch_geometric.transforms as T
import torch.nn.functional as F

from ogb.nodeproppred import PygNodePropPredDataset, Evaluator

from torch_geometric.data import DataLoader
from torch_geometric.nn import SAGEConv
from torch_geometric.loader import NeighborLoader
from torch_geometric.utils import to_undirected, remove_self_loops, add_self_loops

warnings.filterwarnings("ignore", message="You are using `torch.load` with `weights_only=False`")

# Constants gotten from https://github.com/LUOyk1999/tunedGNN
LEARNING_RATE = 0.0005
WEIGHT_DECAY = 5e-4
HIDDEN_DIM = 256
NUM_LAYERS = 4
DROPOUT = 0.5
EPOCHS = 2000

ACCURACY_THRESHOLD = 0.001
MODEL_PATH = './arxiv_model.pt'

dataset = PygNodePropPredDataset(name="ogbn-arxiv", root='/global/D1/homes/sboyar/dataset/ogb/data')
data = dataset[0]
split_idx = dataset.get_idx_split()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

transform = T.Compose([T.ToUndirected(), T.AddSelfLoops()])
data = transform(data)
data.to(device)

data.x = data.x.to(device)
data.y = data.y.to(device)
data.edge_index = to_undirected(data.edge_index)
data.edge_index, _ = remove_self_loops(data.edge_index)
data.edge_index, _ = add_self_loops(data.edge_index, num_nodes=data.num_nodes)
data.edge_index = data.edge_index.to(device)

for key in split_idx:
    split_idx[key] = split_idx[key].to(device)

evaluator = Evaluator(name='ogbn-arxiv')

class GNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, aggr='mean', normalize=True, project=False, num_layers=2, dropout=0.0):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels, aggr=aggr, normalize=normalize, project=project))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))

        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels, aggr=aggr, normalize=normalize, project=project))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))

        self.convs.append(SAGEConv(hidden_channels, out_channels, aggr=aggr, normalize=normalize, project=project))
        self.dropout = dropout

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = self.bns[i](x)
            x = x.relu()
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

def train(model, optimizer):
    model.train()

    optimizer.zero_grad()
    out = model(data.x, data.edge_index)

    loss = F.cross_entropy(out[split_idx['train']], data.y.squeeze(1)[split_idx['train']])
    loss.backward()
    optimizer.step()

    return loss.item()

@torch.no_grad()
def test(model):
    model.eval()

    out = model(data.x, data.edge_index)
    y_pred = out.argmax(dim=-1, keepdim=True)

    train_acc = evaluator.eval({
        'y_true': data.y[split_idx['train']],
        'y_pred': y_pred[split_idx['train']],
    })['acc']

    valid_acc = evaluator.eval({
        'y_true': data.y[split_idx['valid']],
        'y_pred': y_pred[split_idx['valid']],
    })['acc']

    test_acc = evaluator.eval({
        'y_true': data.y[split_idx['test']],
        'y_pred': y_pred[split_idx['test']],
    })['acc']

    return train_acc, valid_acc, test_acc

def save_model(model, optimizer, epoch, best_val_acc, best_test_acc, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_val_acc': best_val_acc,
        'best_test_acc': best_test_acc,
    }, path)

def load_and_evaluate_model(path):
    model = GNN(
        in_channels=dataset.num_features,
        hidden_channels=HIDDEN_DIM,
        out_channels=dataset.num_classes,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT
    ).to(device)

    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])

    train_acc, val_acc, test_acc = test(model)

    print(f"Loaded model from epoch {checkpoint['epoch']}")
    print(f"Final results:")
    print(f"Train: {100 * train_acc:.2f}%")
    print(f"Valid: {100 * val_acc:.2f}%")
    print(f"Test: {100 * test_acc:.2f}%")

def main():
    print("Initializing model...")
    model = GNN(
        in_channels=dataset.num_features,
        hidden_channels=HIDDEN_DIM,
        out_channels=dataset.num_classes,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    best_val_acc = 0
    best_test_acc = 0
    best_epoch = 0
    patience = 50
    patience_counter = 0

    print("Starting training...")
    for epoch in range(1, EPOCHS+1):
        loss = train(model, optimizer)

        if epoch % 10 == 0 or epoch == EPOCHS:
            train_acc, val_acc, test_acc = test(model)

            if val_acc > (best_val_acc + ACCURACY_THRESHOLD):
                best_val_acc = val_acc
                best_test_acc = test_acc
                best_epoch = epoch
                patience_counter = 0

                save_model(model, optimizer, epoch, best_val_acc, best_test_acc, MODEL_PATH)
            else:
                patience_counter += 1


            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, '
                  f'Train: {100 * train_acc:.2f}%, '
                  f'Valid: {100 * val_acc:.2f}%, '
                  f'Test: {100 * test_acc:.2f}%, '
                  f'Best Valid: {100 * best_val_acc:.2f}% (Epoch {best_epoch}), '
                  f'Best Test: {100 * best_test_acc:.2f}%')

            if patience_counter >= patience:
                print(f"Early stopping after {patience} evaluations with no improvement ({patience*10} epochs)")
                break

    print(f"Done. best model from epoch {best_epoch}")
    print(f"Best val: {100*best_val_acc:.2f}%")
    print(f"Best test: {100*best_test_acc:.2f}%")

    print("checking saved model...")
    load_and_evaluate_model(MODEL_PATH)

if __name__ == "__main__":
    main()
