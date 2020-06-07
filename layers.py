import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F


# class CondBatchNorm1d(nn.Module):
#     def __init__(self, num_features, num_classes):
#         super(CondBatchNorm1d, self).__init__()
        
#         self.num_features = num_features
#         self.bn = nn.BatchNorm1d(num_features, affine=False)
#         self.embed = nn.Embedding(num_classes, num_features * 2)
#         self.embed.weight.data[:, :num_features].normal_(1, 0.02)  # Initialise scale at N(1, 0.02)
#         self.embed.weight.data[:, num_features:].zero_()  # Initialise bias at 0

#     def forward(self, x, y):
#         out = self.bn(x)
#         gamma, beta = self.embed(y).chunk(2, 1)
#         out = gamma * out + beta
#         return out


class ConditionalBatchNorm1d(nn.BatchNorm1d):
    def __init__(self, num_features, eps=1e-05, momentum=0.1,
                 affine=False, track_running_stats=True):
        super(ConditionalBatchNorm1d, self).__init__(
            num_features, eps, momentum, affine, track_running_stats
        )

    def forward(self, input, weight, bias, **kwargs):
        self._check_input_dim(input)

        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            self.num_batches_tracked += 1
            if self.momentum is None:  # use cumulative moving average
                exponential_average_factor = 1.0 / self.num_batches_tracked.item()
            else:  # use exponential moving average
                exponential_average_factor = self.momentum

        output = F.batch_norm(input, self.running_mean, self.running_var,
                              self.weight, self.bias,
                              self.training or not self.track_running_stats,
                              exponential_average_factor, self.eps)
        if weight.dim() == 1:
            weight = weight.unsqueeze(0)
        if bias.dim() == 1:
            bias = bias.unsqueeze(0)
        size = output.size()
        weight = weight.expand(size)
        bias = bias.expand(size)
        return weight * output + bias


class CategoricalCondBatchNorm1d(ConditionalBatchNorm1d):
    def __init__(self, num_features, num_classes, eps=1e-5, momentum=0.1,
                 affine=False, track_running_stats=True):
        super(CategoricalCondBatchNorm1d, self).__init__(
            num_features, eps, momentum, affine, track_running_stats
        )
        self.weights = nn.Embedding(num_classes, num_features)
        self.biases = nn.Embedding(num_classes, num_features)

        self._initialize()

    def _initialize(self):
        init.ones_(self.weights.weight.data)
        init.zeros_(self.biases.weight.data)

    def forward(self, input, c, **kwargs):
        weight = self.weights(c)
        bias = self.biases(c)

        return super(CategoricalCondBatchNorm1d, self).forward(input, weight, bias)


class BN_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(BN_block, self).__init__()
        
        self.model = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU()
        )
        
    def forward(self, x):
        return self.model(x)   

    
class CondBN_block(nn.Module):
    def __init__(self, in_channels, out_channels, num_classes):
        super(CondBN_block, self).__init__()
        
        self.linear = nn.Linear(in_channels, out_channels)
        self.condbn = CategoricalCondBatchNorm1d(out_channels, num_classes)
        self.act = nn.ReLU()
        
    def forward(self, x, y):
        x = self.linear(x)
        return self.act(self.condbn(x, y))