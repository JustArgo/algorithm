import pandas as pd
import numpy as np

def create_interval_dataset(dataset, look_back):
    """
    :param dataset: input array of time intervals
    :param look_back: each training set feature length
    :return: convert an array of values into a dataset matrix.
    """
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        dataX.append(dataset[i:i+look_back])
        dataY.append(dataset[i+look_back])
    return np.asarray(dataX), np.asarray(dataY)

df = pd.read_csv("consume_time.csv")    
dataset_init = np.asarray(df)    # if only 1 column
dataX, dataY = create_interval_dataset(dataset_init, 3)    # look back if the training set sequence length