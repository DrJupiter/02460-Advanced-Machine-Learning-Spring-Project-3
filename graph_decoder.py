# %%
import torch
from torch.utils.data import random_split
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_dense_adj, to_dense_batch
import matplotlib.pyplot as plt
from test import batch_generate_directed_edges

def generate_directed_edges(n):
    # Generate a tensor of all nodes
    nodes = torch.arange(n)
    # Get Cartesian product of nodes with themselves to form directed edges
    directed_edges = torch.cartesian_prod(nodes, nodes)
    # Filter out self-edges (where source and target are the same node)
    directed_edges = directed_edges[directed_edges[:, 0] != directed_edges[:, 1]]
    # Transpose to get the desired shape (2 x number of edges)
    result = directed_edges.t()
    return result


# %% Interactive plots
plt.ion() # Enable interactive plotting
def drawnow():
    """Force draw the current plot."""
    plt.gcf().canvas.draw()
    plt.gcf().canvas.flush_events()

# %% Device
device = 'cpu'

# %% Load the MUTAG dataset
# Load data
dataset = TUDataset(root='./data/', name='MUTAG').to(device)
node_feature_dim = 7

# Split into training and validation
rng = torch.Generator().manual_seed(0)
train_dataset, validation_dataset, test_dataset = random_split(dataset, (100, 44, 44), generator=rng)

# Create dataloader for training and validation
train_loader = DataLoader(train_dataset, batch_size=100)
validation_loader = DataLoader(validation_dataset, batch_size=44)
test_loader = DataLoader(test_dataset, batch_size=44)

# %% Define a simple graph convolution for graph classification
class SimpleGraphConv(torch.nn.Module):
    """Simple graph convolution for graph classification

    Keyword Arguments
    -----------------
        node_feature_dim : Dimension of the node features
        filter_length : Length of convolution filter
    """

    def __init__(self, node_feature_dim, filter_length, output_feature_dim=1):
        super().__init__()

        # Define dimensions and other hyperparameters
        self.node_feature_dim = node_feature_dim
        self.filter_length = filter_length
        self.output_feature_dim = output_feature_dim

        # Define graph filter
        self.h = torch.nn.Parameter(1e-5*torch.randn(filter_length))
        self.h.data[0] = 1.

        # State output network
        self.output_net = torch.nn.Linear(self.node_feature_dim, self.output_feature_dim)
        

        self.tensor_scale  = torch.nn.Parameter(torch.randn(self.output_feature_dim))
        self.cached = False

    def forward(self, x, number_of_nodes):
        """Evaluate neural network on a batch of graphs.

        Parameters
        ----------
        x : torch.tensor (num_nodes x num_features)
            Node features.
        edge_index : torch.tensor (2 x num_edges)
            Edges (to-node, from-node) in all graphs.
        batch : torch.tensor (num_nodes)
            Index of which graph each node belongs to.

        Returns
        -------
        out : torch tensor (num_graphs)
            Neural network output for each graph.

        """
        lenghts = []
        edge_index = []
        for item in batch_generate_directed_edges(number_of_nodes):
            edge_index.append(item)
            lenghts.append(item.shape[1])
        lenghts = torch.tensor(lenghts)
        edge_index = torch.cat(edge_index, dim=1)
        batch = torch.arange(len(number_of_nodes)).repeat_interleave(lenghts)

        # Extract number of nodes and graphs
        num_graphs = batch.max()+1

        # Compute adjacency matrices and node features per graph
        A = to_dense_adj(edge_index, batch)
        X, idx = to_dense_batch(x, batch)
 

        # ---------------------------------------------------------------------------------------------------------

        # Implementation in vertex domain
        # node_state = torch.zeros_like(X)
        # for k in range(self.filter_length):
        #     node_state += self.h[k] * torch.linalg.matrix_power(A, k) @ X

        # Implementation in spectral domain
        L, U = torch.linalg.eigh(A)        
        exponentiated_L = L.unsqueeze(2).pow(torch.arange(self.filter_length, device=L.device))
        diagonal_filter = (self.h[None,None] * exponentiated_L).sum(2, keepdim=True)
        node_state = U @ (diagonal_filter * (U.transpose(1, 2) @ X))

        # ---------------------------------------------------------------------------------------------------------

        # Aggregate the node states
        graph_state = node_state.sum(1)

        # Output
        out = self.output_net(graph_state).flatten()
        out = torch.tensordot(out, self.tensor_scale, dims=([1], [0]))
        out = torch.tensordot(out, out, dims=([0], [0]))
        return torch.sigmoid(out)
    
# %% Set up the model, loss, and optimizer etc.
# Instantiate the model
filter_length = 3
torch.manual_seed(1)
model = SimpleGraphConv(node_feature_dim, filter_length).to(device)

# Loss function
cross_entropy = torch.nn.BCEWithLogitsLoss()

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=1.)

# Learning rate scheduler
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.995)

# %% Lists to store accuracy and loss
train_accuracies = []
train_losses = []
validation_accuracies = []
validation_losses = []

# %% Fit the model
# Number of epochs
epochs = 500

for epoch in range(epochs):
    # Loop over training batches
    model.train()
    train_accuracy = 0.
    train_loss = 0.
    for data in train_loader:
        out = model(data.x, data.edge_index, batch=data.batch)
        loss = cross_entropy(out, data.y.float())

        # Gradient step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(out.shape,data.x.shape, data.edge_index.shape, data.y.shape, data.batch.shape)
        # Compute training loss and accuracy
        train_accuracy += sum((out>0) == data.y).detach().cpu() / len(train_loader.dataset)
        train_loss += loss.detach().cpu().item() * data.batch_size / len(train_loader.dataset)
    
    # Learning rate scheduler step
    scheduler.step()

    # Validation, print and plots
    with torch.no_grad():    
        model.eval()
        # Compute validation loss and accuracy
        validation_loss = 0.
        validation_accuracy = 0.
        for data in validation_loader:
            out = model(data.x, data.edge_index, data.batch)
            validation_accuracy += sum((out>0) == data.y).cpu() / len(validation_loader.dataset)
            validation_loss += cross_entropy(out, data.y.float()).cpu().item() * data.batch_size / len(validation_loader.dataset)

        # Store the training and validation accuracy and loss for plotting
        train_accuracies.append(train_accuracy)
        train_losses.append(train_loss)
        validation_losses.append(validation_loss)
        validation_accuracies.append(validation_accuracy)

        # Print stats and update plots
        if (epoch+1)%50 == 0:
            print(f'Epoch {epoch+1}')
            print(f'- Learning rate   = {scheduler.get_last_lr()[0]:.1e}')
            print(f'- Train. accuracy = {train_accuracy:.3f}')
            print(f'         loss     = {train_loss:.3f}')
            print(f'- Valid. accuracy = {validation_accuracy:.3f}')
            print(f'         loss     = {validation_loss:.3f}')

            plt.figure('Loss').clf()
            plt.plot(train_losses, label='Train')
            plt.plot(validation_losses, label='Validation')
            plt.legend()
            plt.xlabel('Epoch')
            plt.ylabel('Cross entropy')
            plt.yscale('log')
            plt.tight_layout()
            drawnow()

            plt.figure('Accuracy').clf()
            plt.plot(train_accuracies, label='Train')
            plt.plot(validation_accuracies, label='Validation')
            plt.legend()
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.tight_layout()
            drawnow()

# %% Save final predictions.
with torch.no_grad():
    data = next(iter(test_loader))
    out = model(data.x, data.edge_index, data.batch).cpu()
    torch.save(out, 'test_predictions.pt')