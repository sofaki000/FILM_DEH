import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim


# Define a function to create a small dataset
from torch.utils.data import DataLoader

from understanding.data_utilities import create_small_dataset, TimeSeriesDataset, read_excel_and_forecast
from understanding.model import Seq2Seq




# Create the dataset
#input_data, target_data = create_small_dataset()
input_data = read_excel_and_forecast()
batch_size  =2
dataset = TimeSeriesDataset(input_data)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


# Instantiate the model
input_size = 1 # num of features in the input time series data
hidden_size = 128 # num of features you want to predict
output_size = 1
model = Seq2Seq(input_size, hidden_size, output_size)

# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

losses = []
# Training loop
num_epochs = 256#00

for epoch in range(num_epochs):
    for i, data in enumerate(data_loader):
        inputs = data

        # rolling window approach, target is the next value in the time series
        targets = torch.roll(data, -4)

        # Forward pass
        outputs = model(inputs)

        # Compute the loss
        loss = criterion(outputs, targets)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        losses.append(loss.detach().item())


plt.plot(losses)
plt.savefig("losses.png")

# Evaluate the model
with torch.no_grad():
    test_input_data = torch.tensor([[10.0]]).view(1, 1, 1)
    test_output_data = model(test_input_data)
    print(f'Test Input: {test_input_data.item():.2f}, Test Output: {test_output_data.item():.2f}')
