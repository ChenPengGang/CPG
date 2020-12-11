import torch
from torch_geometric.nn import GCNConv
from torch_geometric.nn import TopKPooling
from torch.nn import Parameter
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import torch.nn.functional as F


class Net(torch.nn.Module):
    def __init__(self,input_dim,output_dim,num_nodes):
        super(Net, self).__init__()

        self.in_conv1 = GCNConv(input_dim, 64)
        self.in_conv2 = GCNConv(64, output_dim)
        self.out_conv1 = GCNConv(input_dim, 64)
        self.out_conv2 = GCNConv(64, output_dim)
        self.output_dim=output_dim

#改：
    def in_gcn(self, x,edge_index):
        x = torch.relu(self.in_conv1(x, edge_index))
        x = self.in_conv2(x, edge_index)
        return x

    def out_gcn(self, x,edge_index):
        x = torch.relu(self.out_conv1(x, edge_index))
        x = self.out_conv2(x, edge_index)
        return x
