import torch

def generate_directed_edges(n):
    nodes = torch.arange(n)
    directed_edges = torch.cartesian_prod(nodes, nodes)
    directed_edges = directed_edges[directed_edges[:, 0] != directed_edges[:, 1]]
    return directed_edges.t()

def batch_generate_directed_edges(N):
    for n in N:
        yield generate_directed_edges(n) 


if __name__ == "__main__":
    number_of_nodes = torch.tensor([2, 3, 4])
    lenghts = []
    edges = []
    for item in batch_generate_directed_edges(number_of_nodes):
        edges.append(item)
        lenghts.append(item.shape[1])
    lenghts = torch.tensor(lenghts)
    edges = torch.cat(edges, dim=1)
    batch = torch.arange(len(number_of_nodes)).repeat_interleave(lenghts) 
    print(batch)
    print(lenghts)

