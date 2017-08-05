import keras
from keras.layers.core import Dense
import os.path


class Network:
    def __init__(self, in_dim=None, hiddens=(20, 20), model_path=None):
        """either in_dim or model_path has to be set"""
        self.model_path = model_path
        self.in_dim = in_dim
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
        self.requests = []

    def eval(self, netInput):
        output = self.model.predict(netInput)
        self.requests.append((output, netInput))
        return float(output)

    def train(self, input, target):
        loss, _ = self.model.train_on_batch(input, target)
        return loss

    def save(self, model_path):
        self.model.save(model_path)