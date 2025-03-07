import torch
import torch_geometric.transforms as T
import torch.nn.functional as F
from torch_geometric.datasets import OGB_MAG
from torch_geometric.nn import SAGEConv, to_hetero, HGTConv, Linear
from torch_geometric.loader import NeighborLoader


dataset = OGB_MAG(root='./data', preprocess='metapath2vec', transform=T.ToUndirected())
data = dataset[0]


class GNN(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv((-1, -1), hidden_channels)
        self.conv2 = SAGEConv((-1, -1), out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x

def train(model):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    # OPTIMIZATION 1: Create a tiny subset mask for testing
    paper_indices = torch.nonzero(data['paper'].train_mask, as_tuple=True)[0]
    subset_size = min(10000, len(paper_indices))  # Use at most 5000 nodes
    subset_indices = paper_indices[:subset_size]
    subset_mask = torch.zeros_like(data['paper'].train_mask)
    subset_mask[subset_indices] = True

    # OPTIMIZATION 2: More aggressive sampling, larger batches
    train_loader = NeighborLoader(
        data,
        num_neighbors=[5, 5],  # Sample only 5 neighbors per layer
        batch_size=256,        # Larger batch size
        input_nodes=('paper', subset_mask),
    )

    # OPTIMIZATION 3: Process fewer batches per epoch
    max_batches_per_epoch = 100
    
    avg_loss = 0
    for epoch in range(10):
        total_loss = 0
        num_batches = 0

        print(f'Epoch {epoch+1}/{10} starting...')

        for batch_idx, batch in enumerate(train_loader):
            if batch_idx >= max_batches_per_epoch:
                break

            optimizer.zero_grad()
            out = model(batch.x_dict, batch.edge_index_dict)
            loss = F.cross_entropy(out['paper'], batch['paper'].y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1

            if num_batches % 10 == 0:
                print(f'  Batch {num_batches}, Current Loss: {loss.item():.4f}')

        avg_loss = total_loss / num_batches
        print(f'Epoch: {epoch+1:03d}/{10} completed, Avg Loss: {avg_loss:.4f}')

    return avg_loss

def main():
    model = GNN(hidden_channels=4, out_channels=dataset.num_classes)
    # NOTE: to_hetero converts the NN to be capable to handle heterogeneous
    # graphs, and does not change the dataset
    model = to_hetero(model, data.metadata(), aggr='sum')
    train(model)

    # Evaluation
    model.eval()
    with torch.no_grad():  # Disable gradients for evaluation
        # Use dictionary inputs just like in training
        out = model(data.x_dict, data.edge_index_dict)

        # For heterogeneous graphs, specify the node type
        pred = out['paper'].argmax(dim=1)

        # Access test mask for specific node type
        test_mask = data['paper'].test_mask
        correct = (pred[test_mask] == data['paper'].y[test_mask]).sum()
        acc = int(correct) / int(test_mask.sum())
        print(f'Accuracy: {acc:.4f}')

if __name__ == "__main__":
    main()
