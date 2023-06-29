import torch
from torch_geometric.nn import BatchNorm
from torch_geometric.nn import GCNConv

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCN, self).__init__()
        p=0.3
        self.conv1 = GCNConv(in_channels, 4*hidden_channels)
        self.bn1 = BatchNorm(4*hidden_channels)
        self.dropout1 = torch.nn.Dropout(p=p)
        self.conv2 = GCNConv(4*hidden_channels,2*hidden_channels)
        self.bn2 = BatchNorm(2*hidden_channels)
        self.dropout2 = torch.nn.Dropout(p=p)
        self.conv3 = GCNConv(2*hidden_channels,hidden_channels)
        self.bn3 = BatchNorm(hidden_channels)
        self.dropout3 = torch.nn.Dropout(p=p)
        self.conv4 = GCNConv(hidden_channels,out_channels)
        self.logsoftmax = torch.nn.LogSoftmax(dim=1)
        self.prelu = torch.nn.PReLU()

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = self.dropout1(x)
        x = self.prelu(x)
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = self.dropout2(x)
        x = self.prelu(x)
        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = self.dropout3(x)
        x = self.prelu(x)
        x = self.conv4(x, edge_index)
        return self.logsoftmax(x)
    