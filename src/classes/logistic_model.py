import os
import numpy as np
import csv
import sys
from tqdm import trange

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from utils import get_path
from classes.dataset import Dataset


class LogisticModel:
    """
        Logistic Regression One vs Rest.
    """

    learning_rate = 0.05
    accuracy = 0.0001
    houses = ['Gryffindor', 'Hufflepuff', 'Ravenclaw', 'Slytherin']

    def __init__(self) -> None:
        pass

    @staticmethod
    def _normalize(values):
        # values is a list of np.ndarray (one per feature)
        if not values or any(v.size == 0 for v in values):
            raise ValueError("No data to normalize (empty list or empty features)."
                            " Verify that the test dataset contains valid rows.")

        min_values = [v.min() for v in values]
        max_values = [v.max() for v in values]
        m = [max(abs(a), abs(b), 1e-12) for a, b in zip(min_values, max_values)]  # avoid / 0
        return [v / m[i] for i, v in enumerate(values)]


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
            fieldnames = ['House'] + ['intersect_' + ft for ft in features] + ['weight_' + ft for ft in features]
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerow(fieldnames)
            for house in LogisticModel.houses:
                writer.writerow([house] + intercept[house] + weight[house])

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
        for house in derivative_intercept_dict:
            for value in derivative_intercept_dict[house]:
                if value > self.accuracy:
                    return False

        for house in derivative_weight_dict:
            for value in derivative_weight_dict[house]:
                if value > self.accuracy:
                    return False

        return True


    def _derivative_cost_function(self, notes, binary_dict, intercept, weight):
        intercept_dict = {}
        weight_dict = {}

        for house in LogisticModel.houses:
            linear_output = [intercept[house][i] + weight[house][i] * notes[i] for i in range(len(intercept[house]))]
            sig = [[self._sigmoid(linear_output[j][i]) for i in range(len(linear_output[j]))] for j in range(len(linear_output))]
            error = [[sig[j][i] - binary_dict[house][i] for i in range(len(sig[j]))] for j in range(len(sig))]
            intercept_dict[house] = [(1 / len(notes[i])) * np.sum(error[i]) for i in range(len(error))]
            weight_dict[house] = [(1 / len(notes[i])) * np.sum(error[i] * notes[i]) for i in range(len(error))]
        return intercept_dict, weight_dict


    def train(self, training_dataset, training_features):
        intercept = {'Gryffindor': [0.0] * len(training_features), 'Ravenclaw': [0.0] * len(training_features), 'Slytherin': [0.0] * len(training_features), 'Hufflepuff': [0.0] * len(training_features)}
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

        # Need to add another stop condition (derivative small enough)
        for _ in trange(100000, desc='Training'):
            derivative_intercept_dict, derivative_weight_dict = self._derivative_cost_function(notes, binary_dict, intercept, weight)
            if self._accuracy_reached(derivative_intercept_dict, derivative_weight_dict):
                break
            for house in LogisticModel.houses:
                intercept[house] = [intercept[house][i] - (derivative_intercept_dict[house][i] * self.learning_rate) for i in range(len(derivative_intercept_dict[house]))]
                weight[house] = [weight[house][i] - (derivative_weight_dict[house][i] * self.learning_rate) for i in range(len(derivative_weight_dict[house]))]

        self._store_model(intercept, weight, training_features, 'models/models.csv')

    def predict(self, model_csv, predict_dataset):
        # features to get from csv file
        intercept = {}
        weight = {}
        with open(model_csv, 'r') as f:
            reader = csv.reader(f, delimiter=',')
            row_count = 0
            for row in reader:
                if row_count == 0:
                    features = [row[i].split('_')[1] for i in range(1, len(row))]
                else:
                    intersect_str = row[1:int((len(row) + 1) / 2)]
                    weight_str = row[int((len(row) + 1) / 2):]
                    intercept[row[0]] = [float(intersect_str[i]) for i in range(len(intersect_str))]
                    weight[row[0]] = [float(weight_str[i]) for i in range(len(weight_str))]
                row_count += 1

        scores = {}
        samples = predict_dataset.get_samples()
        notes_raw = [np.array([samples[i][features.index(ft)] for i in range(len(samples))]) for ft in features]
        notes = self._normalize(notes_raw)
        for house in LogisticModel.houses:
            linear_output = [intercept[house][i] + weight[house][i] * notes[i] for i in range(len(intercept[house]))]
            raw_scores = [self._sigmoid(linear_output[i]) for i in range(len(linear_output))]
            scores[house] = self._mean_scores(raw_scores)

        house_index = [np.argmax(np.array([scores[house][i] for house in LogisticModel.houses])) for i in range(len(notes[0]))]
        houses = [LogisticModel.houses[house_index[i]] for i in range(len(house_index))]
        self._store_prediction(houses, get_path('predictions/predictions.csv'))
