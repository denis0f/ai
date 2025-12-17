import yfinance as yf
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

#used to initialy download the data and save it to a csv file
# ticker = "AAPL"
# start_date = "2020-01-01"
# end_date = "2025-11-30"
csv_file = "apple_stock_data.csv"

# df = yf.download(ticker, start=start_date, end=end_date)
# df.to_csv(csv_file)

# print("Data downloaded and saved to apple_stock_data.csv")

#loading the data from the csv file 

df = pd.read_csv(csv_file, index_col=0)
close_prices = df["Close"].values.reshape(-1, 1)

#calling the data 
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_close_prices = scaler.fit_transform(close_prices)

#creating my windowed sequence where i want to use the 60 days prior to predict the close price of the next day

window_size = 60


def create_sequences(data, window_size):
    x, y = [], []
    for i in range(window_size, len(data)):
        x.append(data[i - window_size:i])
        y.append(data[i])
    return np.array(x), np.array(y)


x_data, y_data = create_sequences(scaled_close_prices, window_size)

#splitting the data into train test data (with a seed 50)

x_train, x_test, y_train, y_test = train_test_split(
    x_data,
    y_data,
    test_size=0.2,
    shuffle=False,
    random_state=50
)

#preparing the dataset class for my custom dataset 

class StockDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

#create the dataloader for my train and test datasets 

batch_size = 32

train_dataset = StockDataset(x_train, y_train)
test_dataset = StockDataset(x_test, y_test)

train_dataloader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    drop_last=True
)

test_dataloader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    drop_last=False
)

#feed foward nn model 

class FeedForwardNN(nn.Module):
    def __init__(self, input_size):
        super().__init__()

        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(x.size(0), -1)   # flatten (batch, 60)

        x = self.fc1(x)
        x = self.relu(x)

        x = self.fc2(x)
        x = self.relu(x)

        x = self.fc3(x)
        return x

#training setup and the training loop

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = FeedForwardNN(input_size=window_size).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 100

for epoch in range(epochs):
    model.train()
    train_loss = 0.0

    for x_batch, y_batch in train_dataloader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        predictions = model(x_batch)
        loss = criterion(predictions, y_batch)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_dataloader)

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch + 1}/{epochs}] - Train Loss: {train_loss:.6f}")

#evaluating the model on the test dataset 

model.eval()
test_loss = 0.0

with torch.no_grad():
    for x_batch, y_batch in test_dataloader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        predictions = model(x_batch)
        loss = criterion(predictions, y_batch)
        test_loss += loss.item()

test_loss /= len(test_dataloader)

print(f"Test Loss: {test_loss:.6f}")


#saving the model 
model_path = "stock_predictor_ffnn_model.pth"

torch.save(model.state_dict(), model_path)

print(f"Model saved to {model_path}")


