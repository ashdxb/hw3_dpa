import torch
from torch_geometric.nn import GATConv
from torch_geometric.nn import BatchNorm

class GAT(torch.nn.Module): # Model that got 0.6256
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GAT, self).__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=8, dropout=0.08) # dropout=0.05
        self.bn1 = BatchNorm(8 * hidden_channels)
        self.conv2 = GATConv(8 * hidden_channels, hidden_channels, heads=7, dropout=0.08) # dropout=0.1
        self.bn2 = BatchNorm(7 * hidden_channels)
        self.conv3 = GATConv(7 * hidden_channels, hidden_channels, heads=7, dropout=0.08) # dropout=0.1
        self.bn4 = BatchNorm(7 * hidden_channels)
        self.conv4 = GATConv(7 * hidden_channels, out_channels, heads=5, concat=False)
        self.logsoftmax = torch.nn.LogSoftmax(dim=1)
        self.prelu = torch.nn.PReLU()


    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = self.prelu(x)
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = self.prelu(x)
        x = self.conv3(x, edge_index)
        x = self.bn4(x)
        x = self.prelu(x)
        x = self.conv4(x, edge_index)
        return self.logsoftmax(x)