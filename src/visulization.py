import matplotlib.pyplot as plt
import numpy as np


# Load CSV dataset from "dataset/train.csv"
def load_dataset():
    dataset = np.loadtxt("dataset/train.csv", delimiter=",", skiprows=1)
    return dataset


# Plot cost variable (latest column) distribution with min, max median and mean values
def plot_cost_distribution(dataset):
    # Extract cost variable
    cost = dataset[:, -1]

    # Plot histogram
    plt.hist(cost, bins=100)
    plt.xlabel("Cost")
    plt.ylabel("Frequency")
    plt.title("Distribution of cost")

    # Plot mean, median, min and max values
    plt.axvline(cost.mean(), color="b", linestyle="dashed", linewidth=2)
    plt.axvline(np.median(cost), color="y", linestyle="dashed", linewidth=2)
    plt.axvline(cost.min(), color="r", linestyle="dashed", linewidth=2)
    plt.axvline(cost.max(), color="r", linestyle="dashed", linewidth=2)

    # Show plot
    plt.show()


# Plot cost as a function of other variables
def plot_cost_vs_other_variables(dataset):
    # Extract cost variable
    cost = dataset[:, -1]

    # Extract other variables
    variables = dataset[:, :-1]

    # Plot cost vs other variables
    for i in range(variables.shape[1]):
        plt.scatter(variables[:, i], cost)
        plt.xlabel("Variable {}".format(i))
        plt.ylabel("Cost")
        plt.title("Cost vs Variable {}".format(i))
        plt.show()


def handler():
    dataset = load_dataset()
    # plot_cost_distribution(dataset)
    # plot_cost_vs_other_variables(dataset)
    mean = dataset.mean(axis=0)
    std = dataset.std(axis=0)
    dataset = (dataset - mean) / std
    print(f"mean: {mean}")
    print(f"std: {std}")


if __name__ == "__main__":
    handler()