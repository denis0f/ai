import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import torch 
import torch.nn as nn 
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import yfinance as yf 
from datetime import date
"""
##### Downloading the data  (done once)

# apple_data = yf.download("AAPL", start=date(2020, 1, 1), end=date(2025, 11, 30), auto_adjust=False)

# apple_data.columns = ["Adj Close", "Close", "High", "Low", "Open", "Volume"]

# print(apple_data.head())


# apple_data.to_csv("apple_data.csv")


"""

#loading the data from the csv
def load_csv_data()->pd.DataFrame:
    return pd.read_csv("apple_data.csv", header=[0], index_col=0)


#prepare the data to numpy arrays to be passed to the tensors
def get_array_columns():
    data = load_csv_data()

    close = data["Close"].to_numpy()
    open = data["Open"].to_numpy()
    low = data["Low"].to_numpy()
    high = data["High"].to_numpy()
    volume = data["Volume"].to_numpy()

    return open, high, low, close, volume

def get_model():

    class My_model(nn.Module):
        def __init__(self):
            super(My_model, self).__init__()



def main():

    open, high, low, close, volume = get_array_columns()


    

if __name__ == "__main__":
    main()








