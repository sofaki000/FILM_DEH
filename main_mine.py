import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd

# Load the CSV file using pandas
from config import getConfig
from get_model import get_configured_model
from models.Informer import Model

from understanding.data_utilities import read_excel_and_forecast, TimeSeriesDataset

input_data = read_excel_and_forecast()
batch_size = 2
validation_dataset = TimeSeriesDataset(input_data)
validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True)

train_dataset = TimeSeriesDataset(input_data) #MyDataset(X_train_tensor, y_train_tensor)  # Replace MyDataset with your custom dataset class
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)  # Replace batch_size with your desired batch size

# Create an instance of the Model
model, args = get_configured_model() #config = getConfig()
#model = Model(config )  # Replace configs with your desired model configurations

# Define loss function and optimizer
learning_rate = 0.001
num_epochs = 10
criterion = nn.MSELoss()  # Replace with your desired loss function
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  # Replace learning_rate with your desired learning rate

losses = []
# Training loop
for epoch in range(num_epochs):
    for i, (inputs, targets) in enumerate(train_dataloader):
        optimizer.zero_grad()  # Reset gradients
        outputs = model(inputs)  # Forward pass
        loss = criterion(outputs, targets)  # Compute loss
        loss.backward()  # Backward pass
        optimizer.step()  # Update weights

        # Print progress
        print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
              .format(epoch + 1, num_epochs, i + 1, len(train_dataloader), loss.item()))

        losses.append(loss.detach().item())




plt.plot(losses)
plt.savefig("losses_informer.png")

# Validation loop
model.eval()  # Set model to evaluation mode
with torch.no_grad():
    for i, (inputs, targets) in enumerate(train_dataloader):
        val_outputs = model(inputs)  # Forward pass on validation data
        val_loss = criterion(val_outputs, targets)  # Compute validation loss
        print('Validation Loss: {:.4f}'.format(val_loss.item()))

# You can further customize the training loop and validation loop as needed, such as adding learning rate scheduling,
# model checkpointing, and early stopping, based on your specific requirements.
