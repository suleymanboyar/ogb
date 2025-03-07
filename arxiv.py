from ogb.nodeproppred import PygNodePropPredDataset
import torch
import torch_geometric.transforms as T
import torch.nn.functional as F
import warnings

from torch_geometric.data import DataLoader
from torch_geometric.nn import SAGEConv
from torch_geometric.loader import NeighborLoader

warnings.filterwarnings("ignore", message="You are using `torch.load` with `weights_only=False`")

# Values taken from GraphSAGE's paper
S1 = 25
S2 = 10
BATCH_SIZE=512
# They used 10 epochs for another dataset and got good result with it, 
# while its not enough for ogbn-arxiv
EPOCHS = 80

dataset = PygNodePropPredDataset(name = "ogbn-arxiv", root = './data')
data = dataset[0]

split_detaset = dataset.get_idx_split()

train_loader = NeighborLoader(
    data,
    num_neighbors=[S1,S2],
    batch_size=BATCH_SIZE,
    input_nodes=split_detaset["train"],
    shuffle=True
)

valid_loader = NeighborLoader(
    data,
    num_neighbors=[S1,S2],
    batch_size=BATCH_SIZE,
    input_nodes=split_detaset["valid"],
    shuffle=False
)

test_loader = NeighborLoader(
    data,
    num_neighbors=[S1,S2],
    batch_size=BATCH_SIZE,
    input_nodes=split_detaset["test"],
    shuffle=False
)

class GNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels, aggr='max', normalize=True, project=True)
        self.conv2 = SAGEConv(hidden_channels, out_channels, aggr='max', normalize=True, project=True)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x

def train(model, loader):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    avg_loss = 0

    for epoch in range(EPOCHS):
        total_loss = 0
        num_batches = 0

        print(f'Epoch {epoch+1:03d}/{EPOCHS} starting...')

        for batch in loader:

            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index)
            # Get only the output for target nodes (not neighbor nodes)
            # In NeighborLoader, the first batch.batch_size nodes are the target nodes
            out = out[:batch.batch_size]
            target = batch.y.squeeze()[:batch.batch_size]
            loss = F.cross_entropy(out, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            if num_batches % 10 == 0:
                print(f'  Batch {num_batches}, Current Loss: {loss.item():.4f}')

        avg_loss = total_loss / num_batches
        print(f'Epoch: {epoch+1:03d}/{EPOCHS} completed, Avg Loss: {avg_loss:.4f}')

    return avg_loss

def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in loader:
            out = model(batch.x, batch.edge_index)
            # Get only predictions for target nodes
            target_nodes = batch.batch_size
            out = out[:target_nodes]
            pred = out.argmax(dim=1)
            # Get ground truth labels, squeeze if needed
            if batch.y.dim() > 1:
                target = batch.y.squeeze()[:target_nodes]
            else:
                target = batch.y[:target_nodes]
            # Calculate accuracy
            correct += (pred == target).sum().item()
            total += target_nodes

    return correct / total

def main():
    model = GNN(in_channels=dataset.num_features, hidden_channels=128, out_channels=dataset.num_classes)
    train(model, train_loader)
    test_acc = evaluate(model, test_loader)
    print(f'Test Accuracy: {test_acc:.4f}')
    
if __name__ == "__main__":
    main()
