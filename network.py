import numpy as np
import keras
from keras.layers.core import Dense
import os.path


class Network:
    def __init__(self, height, width, hiddens=(20, 20), model_path=None):
        self.in_dim = width * height + width
        self.hiddens = hiddens
        if model_path is not None and os.path.isfile(model_path):
            print("Loading Model from", model_path)
            self.model = keras.models.load_model(model_path)
        else:
            self.model = keras.models.Sequential()
            self.model.add(Dense(hiddens[0], input_dim=self.in_dim))  # input_shape=(self.in_dim,), activation="sigmoid"))
            for i in range(1, len(hiddens)):
                self.model.add(Dense(hiddens[i], activation="relu"))
            self.model.add(Dense(1, activation="linear"))
            self.model.compile(loss="mse", metrics=["accuracy"], optimizer="rmsprop")
            self.model.save("4x5_20-20-1.h5")
        self.requests = []

    def eval(self, netInput):
        output = self.model.predict(netInput)
        self.requests.append((output, netInput))
        return float(output)

    def train(self, input, target):
        loss, _ = self.model.train_on_batch(input, target)
        return loss
