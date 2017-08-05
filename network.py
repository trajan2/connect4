import keras
from keras.layers.core import Dense


class Network:
    def __init__(self, in_dim: int, hiddens=(20, 20)):
        self.model = keras.models.Sequential()
        self.in_dim = in_dim
        self.hiddens = hiddens
        self.model.add(Dense(hiddens[0], input_dim=self.in_dim))  # input_shape=(self.in_dim,), activation="sigmoid"))
        for i in range(1, len(hiddens)):
            self.model.add(Dense(hiddens[i], activation="sigmoid"))
        self.model.add(Dense(1, activation="linear"))
        self.model.compile(loss="mse", metrics=["accuracy"], optimizer="rmsprop")

    def eval(self, net_input):
        output = self.model.predict(net_input)
        return output

    def train(self, net_input, target):
        self.model.train_on_batch(net_input, target)
