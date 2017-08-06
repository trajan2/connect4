import keras
from keras.layers.core import Dense
from keras.models import load_model


class Network:
    def __init__(self, in_dim: int, load_file: str = None, hiddens=(100, 100)):
        if load_file is not None:
            print("Load model", load_file)
            self.model = load_model(load_file + ".h5")
            return

        print("Create new model")
        self.model = keras.models.Sequential()
        self.model.add(Dense(hiddens[0], input_dim=in_dim))  # input_shape=(self.in_dim,), activation="sigmoid"))
        for i in range(1, len(hiddens)):
            self.model.add(Dense(hiddens[i], activation="sigmoid"))
        self.model.add(Dense(1, activation="linear"))
        self.model.compile(loss="mse", metrics=["accuracy"], optimizer="rmsprop")

    def eval(self, net_input):
        output = self.model.predict(net_input)
        return output

    def train(self, net_input, target):
        self.model.train_on_batch(net_input, target)

    def store(self, name):
        print("Save model", name)
        self.model.save(name + ".h5")
