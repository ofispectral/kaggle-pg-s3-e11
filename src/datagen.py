import tensorflow as tf
import numpy as np


# Standarize the data
def standardize(dataset):
    mean = dataset.mean(axis=0)
    std = dataset.std(axis=0)
    dataset = (dataset - mean) / std
    return dataset


# Data generator
def data_generator(dataset, batch_size=32):
    dataset_size = dataset.shape[0]
    
    # Shuffle the dataset
    np.random.shuffle(dataset)

    # Generate batches
    for i in range(0, dataset_size, batch_size):
        batch = dataset[i : i + batch_size]
        yield batch[:, :-1], batch[:, -1]


# Load dataset
def load_dataset():
    dataset = np.loadtxt("dataset/train.csv", delimiter=",", skiprows=1)
    return dataset


# Build data generator
def build_data_generator(batch_size=32):
    dataset = load_dataset()
    dataset = standardize(dataset)
    return data_generator(dataset, batch_size)


# data_gen = build_data_generator()
# for x, y in data_gen:
#     print(x.shape, y.shape, x.mean(), x.std(), y.mean(), y.std())
