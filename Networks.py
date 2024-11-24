import torch
import torch.nn as nn
import torch.optim as optim

class Action_Conditioned_FF(nn.Module):
    def __init__(self):
        super(Action_Conditioned_FF, self).__init__()

        # Hard-coded architecture parameters
        self.input_size = 6    # Number of input features
        self.hidden_size = 64  # Size of hidden layers
        self.output_size = 1   # Size of the output layer

        # Define the network layers
        self.fc1 = nn.Linear(self.input_size, self.hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc3 = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input):
        x = self.fc1(input)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        output = self.fc3(x)
        return output

    def evaluate(self, model, test_loader, loss_function):
        model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for batch in test_loader:
                inputs = batch['input']
                labels = batch['label']

                outputs = model(inputs)
                loss = loss_function(outputs, labels)
                total_loss += loss.item()

        average_loss = total_loss / len(test_loader)
        return average_loss


def main():
    # Initialize the model without arguments
    model = Action_Conditioned_FF()
    learning_rate = 0.001
    num_epochs = 10

    # Define loss function and optimizer
    loss_function = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Import your Data_Loaders class
    from Data_Loaders import Data_Loaders
    batch_size = 16
    data_loaders = Data_Loaders(batch_size)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        for batch in data_loaders.train_loader:
            inputs = batch['input']
            labels = batch['label']

            # Forward pass
            outputs = model(inputs)
            loss = loss_function(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Evaluate the model on the test set
        test_loss = model.evaluate(model, data_loaders.test_loader, loss_function)
        print(f'Epoch [{epoch+1}/{num_epochs}], Test Loss: {test_loss:.4f}')


if __name__ == '__main__':
    main()
