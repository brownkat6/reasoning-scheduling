class MLP(nn.Module):
    def __init__(self, input_dim=1536, hidden_dims=[256], output_dim=16, activation='relu', dropout=0.0):
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.activation_name = activation
        self.dropout_rate = dropout
        
        # Initialize activations based on name
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'gelu':
            self.activation = nn.GELU() 
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        else:
            raise ValueError(f"Unsupported activation function: {activation}")
        
        # Build the layers
        layers = []
        
        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dims[0]))
        layers.append(self.activation)
        
        # Add dropout if specified
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        
        # Hidden layers
        for i in range(len(hidden_dims) - 1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            layers.append(self.activation)
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        
        # Output layer
        layers.append(nn.Linear(hidden_dims[-1], output_dim))
        layers.append(nn.Sigmoid())  # Always use sigmoid for output to ensure values between 0 and 1
        
        self.model = nn.Sequential(*layers)
        
        print(f"Created MLP: input_dim={input_dim}, hidden_dims={hidden_dims}, output_dim={output_dim}, activation={activation}, dropout={dropout}")
        print(f"Model architecture:\n{self.model}")

    def forward(self, x):
        return self.model(x)