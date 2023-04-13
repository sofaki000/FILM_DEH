import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

file_path = "C:\\Users\\Lenovo\\Desktop\\film_deh\\FILM_DEH\\data\\test_data.csv"

row_name = "TOTAL_CONS"

# Define a custom dataset class for time series data
class TimeSeriesDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

def create_small_dataset():
    # Generate input data
    input_data = torch.linspace(0, 9, steps=10).view(10, 1, 1)
    # Generate target data (shifted by 1 step)
    target_data = input_data + 1
    # Return input and target data
    return input_data, target_data

def read_excel_and_forecast( ):
    # Load the Excel file
    df =pd.read_csv(file_path)

    # Extract the row based on the row name (header)
    input_data = torch.tensor(df[row_name].values, dtype=torch.float32).view(-1, 1, 1)


    return input_data