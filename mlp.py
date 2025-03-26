import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_dim=1536, hidden_dim=256, output_dim=16):
        super(MLP, self).__init__()
        print(f"Input dim: {input_dim}, Hidden dim: {hidden_dim}, Output dim: {output_dim}")
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)  # Removed second hidden layer, direct connection to output
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()  # Add sigmoid activation

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)  # Apply sigmoid to ensure outputs are between 0 and 1
        return x 