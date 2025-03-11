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

# Constants
LEARNING_RATE = 0.01
WEIGHT_DECAY = 5e-4
EPOCHS = 2000
RUNS = 10

# Load dataset (full graph)
dataset = PygNodePropPredDataset(name="ogbn-arxiv", root='/global/D1/homes/sboyar/dataset/ogb/data')
data = dataset[0]
split_idx = dataset.get_idx_split()

# Move everything to device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


transform = T.Compose([T.ToUndirected(), T.AddSelfLoops()])
data = transform(data)
data.to(device)

# Preprocess graph (once)
data.x = data.x.to(device)
data.y = data.y.to(device)
data.edge_index = to_undirected(data.edge_index)
data.edge_index, _ = remove_self_loops(data.edge_index)
data.edge_index, _ = add_self_loops(data.edge_index, num_nodes=data.num_nodes)
data.edge_index = data.edge_index.to(device)

# Convert split indices to device
for key in split_idx:
    split_idx[key] = split_idx[key].to(device)

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

    # Full-batch forward pass
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)

    # Only compute loss on training nodes
    loss = F.cross_entropy(out[split_idx['train']], data.y.squeeze(1)[split_idx['train']])
    loss.backward()
    optimizer.step()

    return loss.item()

@torch.no_grad()
def test(model, evaluator):
    model.eval()

    # Full-batch forward pass
    out = model(data.x, data.edge_index)
    y_pred = out.argmax(dim=-1, keepdim=True)

    # Evaluate on all splits
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
    print(f"Model saved to {path}")

def load_and_evaluate_model(path):
    # Create a new model instance
    model = GNN(
        in_channels=dataset.num_features,
        hidden_channels=HIDDEN_DIM,
        out_channels=dataset.num_classes,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT
    ).to(device)

    # Load the saved state
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Evaluate
    train_acc, val_acc, test_acc = test(model)

    print(f"Loaded model from epoch {checkpoint['epoch']}")
    print(f"Final performance:")
    print(f"Train accuracy: {100 * train_acc:.2f}%")
    print(f"Validation accuracy: {100 * val_acc:.2f}%")
    print(f"Test accuracy: {100 * test_acc:.2f}%")

def main():
    # Create the model
    model = GNN(
        in_channels=dataset.num_features,
        hidden_channels=256,
        out_channels=dataset.num_classes,
        dropout=0.5
    ).to(device)

    # Setup evaluator
    evaluator = Evaluator(name='ogbn-arxiv')

    for run in range(1, RUNS+1):
        print(f"Run {run}/{RUNS}")
        model.reset_parameters()
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

        best_val_acc = 0
        best_test_acc = 0

        for epoch in range(1, EPOCHS+1):
            # Training
            loss = train(model, optimizer)

            # Evaluation
            if epoch % 10 == 0 or epoch == EPOCHS:
                train_acc, val_acc, test_acc = test(model, evaluator)

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_test_acc = test_acc

                print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, '
                      f'Train: {100 * train_acc:.2f}%, '
                      f'Valid: {100 * val_acc:.2f}%, '
                      f'Test: {100 * test_acc:.2f}%, '
                      f'Best Valid: {100 * best_val_acc:.2f}%, '
                      f'Best Test: {100 * best_test_acc:.2f}%')
    
if __name__ == "__main__":
    main()
