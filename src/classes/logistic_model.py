import os
import numpy as np
import csv
import sys
from tqdm import trange

current = os.path.dirname(os.path.realpath(__file__))

# Getting the parent directory name
# where the current directory is present.
parent = os.path.dirname(current)

# adding the parent directory to
# the sys.path.
sys.path.append(parent)

from utils import get_path
from classes.dataset import Dataset


class LogisticModel:
    """
        Stores the data of known Hogwarts students
    """

    learning_rate = 0.5
    accuracy = 0.001
    houses = ['Gryffindor', 'Hufflepuff', 'Ravenclaw', 'Slytherin']

    def __init__(self) -> None:
        self.nb_samples = 0
        self.nb_features = 0

    @staticmethod
    def _normalize(values):
        min_values = [values[i].min() for i in range(len(values))]
        max_values = [values[i].max() for i in range(len(values))]
        m = [max(abs(min_values[i]), abs(max_values[i])) for i in range(len(min_values))]
        return [values[i] / m[i] for i in range(len(values))]

    @staticmethod
    def _sigmoid(z):
        return 1.0 / (1.0 + np.exp(-z))

    @staticmethod
    def _mean_scores(raw_scores):
        means = []
        for index in range(len(raw_scores[0])):
            mean = 0
            for ft in range(len(raw_scores)):
                mean += raw_scores[ft][index]
            mean /= len(raw_scores)
            means.append(mean)
        return means

    @staticmethod
    def _store_model(intercept, weight, features, file):
        directory = get_path('models/')
        if not os.path.isdir(directory):
            os.makedirs(directory)
        if os.path.exists(file):
            os.remove(file)

        with open(file, 'x', newline='') as csvfile:
            fieldnames = ['House'] + ['intersect'] + ['weight_' + ft for ft in features]
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerow(fieldnames)
            for house in LogisticModel.houses:
                writer.writerow([house] + [intercept[house]] + weight[house])

    @staticmethod
    def _store_prediction(houses, file):
        directory = get_path('predictions/')
        if not os.path.isdir(directory):
            os.makedirs(directory)
        if os.path.exists(file):
            os.remove(file)

        with open(file, 'x', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerow(['Index', 'Hogwarts House'])
            for i in range(len(houses)):
                writer.writerow([i, houses[i]])

    def _accuracy_reached(self, derivative_intercept_dict, derivative_weight_dict):
        for house in derivative_weight_dict:
            for value in derivative_weight_dict[house]:
                if abs(value) > self.accuracy:
                    return False

        return True

    def _derivative_cost_function(self, notes, binary_dict, intercept, weight):
        intercept_dict = {}
        weight_dict = {}

        for house in LogisticModel.houses:
            linear_output = intercept[house]
            for i in range(self.nb_features):
                linear_output += weight[house][i] * notes[i]

            sig = [self._sigmoid(linear_output[i]) for i in range(len(linear_output))]
            error = [sig[i] - binary_dict[house][i] for i in range(len(sig))]
            intercept_dict[house] = (1 / self.nb_samples) * np.sum(error)
            weight_dict[house] = [(1 / self.nb_samples) * np.sum(error * notes[i]) for i in range(self.nb_features)]

        return intercept_dict, weight_dict

    def train(self, training_dataset, training_features):
        self.nb_samples = training_dataset.get_nb_samples()
        self.nb_features = len(training_features)

        intercept = {'Gryffindor': 0.0, 'Ravenclaw': 0.0, 'Slytherin': 0.0, 'Hufflepuff': 0.0}
        weight = {'Gryffindor': [0.0] * len(training_features), 'Ravenclaw': [0.0] * len(training_features), 'Slytherin': [0.0] * len(training_features), 'Hufflepuff': [0.0] * len(training_features)}

        features = training_dataset.get_features()
        samples = training_dataset.get_samples()
        pupils_houses = training_dataset.get_pupils_houses()

        list_notes_raw = [np.array([samples[i][features.index(ft)] for i in range(len(samples))]) for ft in training_features]

        binary_dict = {'Gryffindor': [float(pupils_houses[i] == 'Gryffindor') for i in range(len(pupils_houses))],
                       'Ravenclaw': [float(pupils_houses[i] == 'Ravenclaw') for i in range(len(pupils_houses))],
                       'Hufflepuff': [float(pupils_houses[i] == 'Hufflepuff') for i in range(len(pupils_houses))],
                       'Slytherin': [float(pupils_houses[i] == 'Slytherin') for i in range(len(pupils_houses))]}

        notes = self._normalize(list_notes_raw)

        for _ in trange(1000, desc='Training'):
            derivative_intercept_dict, derivative_weight_dict = self._derivative_cost_function(notes, binary_dict, intercept, weight)
            if self._accuracy_reached(derivative_intercept_dict, derivative_weight_dict):
                break
            for house in LogisticModel.houses:
                intercept[house] = intercept[house] - (derivative_intercept_dict[house] * self.learning_rate) / self.nb_samples
                weight[house] = [weight[house][i] - (derivative_weight_dict[house][i] * self.learning_rate) for i in range(len(derivative_weight_dict[house]))]

        self._store_model(intercept, weight, training_features, 'models/models.csv')

    def predict(self, model_csv, predict_dataset):
        self.nb_samples = predict_dataset.get_nb_samples()

        with open(model_csv, 'r') as f:
            reader = csv.reader(f, delimiter=',')
            header = next(reader)

            # number of learned features
            n = (len(header) - 2)

            # keep only one feature list
            weight_cols = header[2:n+2]
            features = [c.split('_', 1)[1] for c in weight_cols]
            self.nb_features = len(features)

            intercept = {}
            weight = {}
            for row in reader:
                house = row[0]
                intersect_str = row[1]
                weight_str = row[2:n+2]
                intercept[house] = float(intersect_str)
                weight[house] = [float(x) for x in weight_str]

        scores = {}
        samples = predict_dataset.get_samples()
        notes_raw = [np.array([samples[i][predict_dataset.get_features().index(ft)] for i in range(self.nb_samples)]) for ft in features]
        notes = self._normalize(notes_raw)
        for house in LogisticModel.houses:
            linear_output = [intercept[house]] * self.nb_samples
            for i in range(self.nb_samples):
                for j in range(self.nb_features):
                    linear_output[i] += weight[house][j] * notes[j][i]

            scores[house] = [self._sigmoid(linear_output[i]) for i in range(self.nb_samples)]

        house_index = [np.argmax(np.array([scores[house][i] for house in LogisticModel.houses])) for i in range(self.nb_samples)]
        houses = [LogisticModel.houses[house_index[i]] for i in range(len(house_index))]
        self._store_prediction(houses, get_path('predictions/predictions.csv'))

