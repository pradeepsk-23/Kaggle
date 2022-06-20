import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

df = pd.read_csv("./Dataset/insurance.csv")

# Input Columns
input_cols = ['age', 'sex', 'bmi', 'children', 'smoker', 'region']

# Output Columns
output_cols = ['charges']

# Non-numeric / Categorical columns
categorical_cols = ['sex', 'smoker', 'region']


def df_to_arrays(df):
    # Make a copy of the original dataframe
    df1 = df.copy(deep=True)
    # Convert non-numeric categorical columns to numbers
    for col in categorical_cols:
        df1[col] = df1[col].astype('category').cat.codes
    # Extract input & outupts as numpy arrays
    inputs_array = df1[input_cols].to_numpy()
    targets_array = df1[output_cols].to_numpy()
    return inputs_array, targets_array

inputs_array, targets_array = df_to_arrays(df)

# Numpy to Torch tensor conversion
inputs = torch.tensor(inputs_array, dtype=torch.float32)
targets = torch.tensor(targets_array, dtype=torch.float32)

# Dataset
train_ds = TensorDataset(inputs, targets)

# DataLoader
batch_size = 50
train_dl = DataLoader(train_ds, batch_size, shuffle=True)

#Linear Regression Model
input_size = len(input_cols)
output_size = len(output_cols)
model = nn.Linear(input_size, output_size)

# Loss and Optimiser
loss_fn = F.mse_loss
opt = torch.optim.SGD(model.parameters(), lr=1e-5)

# Train the model
epochs = 1000
total_step = len(train_dl)
for epoch in range(epochs):
    for i, (inputs, targets) in enumerate(train_dl):

        ## Forward Pass
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)

        ## Backward and Optimize
        opt.zero_grad()
        loss.backward()
        opt.step()
    
        if (epoch+1) % 100 == 0:
            print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, epochs, loss.item()))