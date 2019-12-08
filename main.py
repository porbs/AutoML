import numpy as np
import random
import json
import os
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Dropout, Flatten
from keras.datasets import mnist
from keras.utils import to_categorical

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(60000, 784).astype('float32')[:1000] / 255
x_test = x_test.reshape(10000, 784).astype('float32')[:1000] / 255
y_train = to_categorical(y_train, 10)[:1000]
y_test = to_categorical(y_test, 10)[:1000]

classes = 10
population = 10
generations = 300
threshold = 0.995
input_shape = [784, ]

losses = [
    'categorical_crossentropy',
    'binary_crossentropy',
    'mean_squared_error',
    'mean_absolute_error',
    'sparse_categorical_crossentropy'
]
acts = ['sigmoid', 'relu', 'softmax', 'tanh', 'elu', 'selu', 'linear']
types = ['dense']
opts = ['sgd', 'rmsprop', 'adagrad', 'adadelta', 'adam', 'adamax', 'nadam']


def build_and_train_model(xtrain, ytrain, hyperparams):
    model = Sequential()

    model.add(Dense(hyperparams['layers'][0]['units'],
                    activation=hyperparams['layers'][0]['act'], input_shape=input_shape))
    for layer in hyperparams['layers'][1:]:
        model.add(Dense(layer['units'], activation=layer['act']))
    model.add(Dense(classes, activation=np.random.choice(acts)))

    model.compile(loss=hyperparams['loss'],
                  optimizer=hyperparams['opt'], metrics=['acc'])
    model.fit(xtrain, ytrain,
              batch_size=hyperparams['batch_size'], epochs=hyperparams['epochs'], verbose=0)
    return model


class Network():
    def __init__(self):
        self._num_layers = np.random.randint(1, 3)
        self._epochs = np.random.randint(1, 15)
        self._batch_size = 64  # np.random.randint(64, 128)
        self._topology = []
        self._loss = random.choice(losses)
        self._opt = random.choice(opts)
        self._accuracy = 0
        for _ in range(self._num_layers):
            self._topology.append(Network.rand_layer())

    @staticmethod
    def rand_layer():
        layer = np.random.choice(types)
        act = np.random.choice(acts)
        units = np.random.randint(20, 500)
        return {
            'type': layer,
            'act': act,
            'units': units
        }

    def init_hyperparams(self):
        hyperparams = {
            'epochs': self._epochs,
            'loss': self._loss,
            'opt': self._opt,
            'batch_size': self._batch_size,
            'layers': self._topology
        }
        return hyperparams


def init_networks(population):
    return [Network() for _ in range(population)]


def fitness(networks):
    for network in networks:
        hyperparams = network.init_hyperparams()

        try:
            model = build_and_train_model(x_train, y_train, hyperparams)
            accuracy = model.evaluate(x_test, y_test, verbose=0)[1]
            network._accuracy = accuracy
            print('Accuracy: {}'.format(network._accuracy))
        except:
            network._accuracy = 0
            print('Build failed.')

    return networks


def selection(networks):
    networks = sorted(
        networks, key=lambda network: network._accuracy, reverse=True)
    networks = networks[:int(0.4 * len(networks))]

    return networks


def cross_layers(l1, l2):
    v = sorted([l1['units'], l2['units']])
    if (v[0] == v[1]):
        v[1] += 1
    return {
        'type': np.random.choice([l1['type'], l2['type']]),
        'act': np.random.choice([l1['act'], l2['act']]),
        'units': np.random.randint(v[0], v[1])
    }


def crossover(networks):
    offspring = []
    for _ in range(int((population - len(networks)) / 2)):
        parent1 = random.choice(networks)
        parent2 = random.choice(networks)
        child1 = Network()
        child2 = Network()

        # Crossing over parent hyper-params
        child1._epochs = int(parent1._epochs * 0.75) + \
            int(parent2._epochs * 0.25)
        child2._epochs = int(parent1._epochs * 0.25) + \
            int(parent2._epochs * 0.75)

        child1._batch_size = int(
            parent1._batch_size * 0.75) + int(parent2._batch_size * 0.25)
        child2._batch_size = int(
            parent1._batch_size * 0.25) + int(parent2._batch_size * 0.75)

        p1t = parent1._topology
        p1l = len(p1t)
        p2t = parent2._topology
        p2l = len(p2t)

        t = parent1._topology.copy()
        t.extend(parent2._topology)

        child1._topology = [cross_layers(
            p1t[i], p2t[i % p2l]) for i in range(p1l)]

        child2._topology = [cross_layers(
            p2t[i], p1t[i % p1l]) for i in range(p2l)]

        # child1._topology = p1t[:int(p1l * 0.75)] + p2t[int(p2l * 0.75):]
        # child2._topology = p2t[:int(p2l * 0.75)] + p1t[int(p1l * 0.75):]

        offspring.append(child1)
        offspring.append(child2)

    networks.extend(offspring)

    return networks


def mutate(networks):
    for network in networks:
        if np.random.uniform(0, 1) <= 0.05:
            network._epochs += np.random.randint(0, 100)
            for i in range(len(network._topology)):
                network._topology[i]['units'] += np.random.randint(0, 100)

    return networks


def extract_stats(networks):
    res = []
    for n in networks:
        res.append({
            'topology': n._topology,
            'epoch': n._epochs,
            'batch_size': n._batch_size,
            'loss': n._loss,
            'opt': n._opt,
            'accuracy': n._accuracy
        })
    return res


def main():
    networks = init_networks(population)
    stats = []

    for i, gen in enumerate(range(generations)):
        print('Generation {}'.format(gen+1))

        networks = fitness(networks)
        stats.append(extract_stats(networks))
        networks = selection(networks)
        networks = crossover(networks)
        networks = mutate(networks)

        try:
            with open('dump_{}.json'.format(i), 'w') as fp:
                json.dump(stats, fp)
            if i > 1:
                os.remove('dump_{}.json'.format(i - 2))
        except:
            pass

        for network in networks:
            if network._accuracy > threshold:
                print('Threshold met')
                print(network.init_hyperparams())
                print('Best accuracy: {}'.format(network._accuracy))
                exit(0)


if __name__ == '__main__':
    main()
