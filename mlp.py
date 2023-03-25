import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


# Standarize the data
def standarize(X):
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    X = (X - mean) / std
    return X


# Load dataset
def load_dataset():
    dataset = np.loadtxt("train.csv", delimiter=",", skiprows=1, usecols=range(1, 17))
    # Split dataset into train and validation
    train_dataset = dataset[: int(0.8 * len(dataset))]
    val_dataset = dataset[int(0.8 * len(dataset)) :]
    return train_dataset, val_dataset


# load dataset
train_dataset, val_dataset = load_dataset()
print(train_dataset.shape)
# Split train dataset into train and validation
X_train, y_train = train_dataset[:, :-1], train_dataset[:, -1]
# Standarize train dataset
X_train = standarize(X_train)
# Split validation dataset into train and validation
X_val, y_val = val_dataset[:, :-1], val_dataset[:, -1]
# Standarize validation dataset
X_val = standarize(X_val)


# Model MLP with 1000 hidden units in each hidden layer
model = tf.keras.models.Sequential(
    [
        tf.keras.layers.Dense(1000, activation="relu", input_shape=(X_train.shape[1],)),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1000, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1000, activation="relu"),
        tf.keras.layers.Dense(1),
    ]
)


# Adam Optimizer with learning rate 0.001, loss function MSE and metrics MAE and MAPE
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, weight_decay=0.0001),
    loss=tf.keras.losses.MeanSquaredError(),
    metrics=[
        tf.keras.metrics.MeanAbsoluteError(),
        tf.keras.metrics.MeanAbsolutePercentageError(),
    ]
)


# Train model with 100 epochs, 32 batch size, 0.2 validation split and 10% of train dataset as validation
history = model.fit(
    X_train,
    y_train,
    batch_size=64,
    epochs=2,
    validation_data=(X_val, y_val),
)


# Save model
model.save("models/mlp.h5")
# Plot model training history
plt.plot(history.history["loss"], label="loss")
plt.plot(history.history["val_loss"], label="val_loss")
plt.legend()
plt.savefig("models/mlp_history.png")
# Plot model
tf.keras.utils.plot_model(model, to_file="models/mlp.png", show_shapes=True, show_layer_names=True)


