from Data_Loaders import Data_Loaders
from Networks import Action_Conditioned_FF

import torch
import torch.nn as nn
import matplotlib.pyplot as plt


def train_model(no_epochs):
    batch_size = 16  # Example batch size
    learning_rate = 0.001  # Example learning rate

    # Initialize data loaders and model
    data_loaders = Data_Loaders(batch_size)
    model = Action_Conditioned_FF()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Define loss function and optimizer
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Track losses
    losses = []
    train_losses = []
    test_losses = []

    # Evaluate initial test loss
    min_loss = model.evaluate(model, data_loaders.test_loader, loss_function)
    losses.append(min_loss)
    print(f"Initial Test Loss: {min_loss:.4f}")

    for epoch_i in range(no_epochs):
        model.train()
        running_loss = 0.0

        for idx, sample in enumerate(data_loaders.train_loader):  # sample['input'] and sample['label']
            inputs = sample['input'].to(device)
            labels = sample['label'].to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = loss_function(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Accumulate loss
            running_loss += loss.item()

        # Average training loss for this epoch
        avg_train_loss = running_loss / len(data_loaders.train_loader)
        train_losses.append(avg_train_loss)

        # Evaluate on test set
        model.eval()
        with torch.no_grad():
            test_loss = model.evaluate(model, data_loaders.test_loader, loss_function)
            test_losses.append(test_loss)

        # Log losses
        print(f"Epoch {epoch_i + 1}/{no_epochs}: Train Loss = {avg_train_loss:.4f}, Test Loss = {test_loss:.4f}")

    # Plot training and test losses
    plt.figure(figsize=(10, 6))
    plt.plot(range(no_epochs), train_losses, label="Training Loss")
    plt.plot(range(no_epochs), test_losses, label="Test Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Test Loss Over Epochs")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    no_epochs = 20  # Set number of epochs
    train_model(no_epochs)