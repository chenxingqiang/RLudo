import torch.nn as nn
import torch.nn.functional as F

from players.NeuralNets.GraphConvolutionLayer import GraphConvolution

DROPOUT = 0.5

class GCN(nn.Module):
    def __init__(self, nfeat, nclass):
        super(GCN, self).__init__()

        h = 8
        self.gc1 = GraphConvolution(nfeat, h)
        self.gc2 = GraphConvolution(h, nclass)

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, DROPOUT, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)
