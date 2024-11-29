import torch.nn as nn

class NeuralNet(nn.Module):
    
    def __init__(self, layer_sizes):
        super(NeuralNet, self).__init__()
        layers = []
        input_size = 784
        for size in layer_sizes:
            layers.append(nn.Linear(input_size, size))
            layers.append(nn.ReLU())
            input_size = size
        layers.append(nn.Linear(input_size, 10))
        self.network = nn.Sequential(*layers)


    def forward(self, x):
        return self.network(x)