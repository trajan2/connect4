import numpy as np
import keras
import game
from keras.layers.core import Dense


class Network:
    def __init__(self, height, width, hiddens=(20, 20)):
        self.model = keras.models.Sequential()
        self.in_dim = width * height + width
        self.hiddens = hiddens
        self.model.add(Dense(hiddens[0], input_dim=self.in_dim))  # input_shape=(self.in_dim,), activation="sigmoid"))
        for i in range(1, len(hiddens)):
            self.model.add(Dense(hiddens[i], activation="sigmoid"))
        self.model.add(Dense(1, activation="relu"))
        self.model.compile(loss="mse", metrics=["accuracy"], optimizer="rmsprop")

    def eval(self, netInput):
        output = self.model.predict(netInput)
        return output

    def train(self, input, target):
        self.model.train_on_batch(input, target)

