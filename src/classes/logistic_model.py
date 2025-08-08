import os
import numpy as np
import pandas as pds
import math
import matplotlib.pyplot as plt
import csv
import sys

current = os.path.dirname(os.path.realpath(__file__))

# Getting the parent directory name
# where the current directory is present.
parent = os.path.dirname(current)

# adding the parent directory to
# the sys.path.
sys.path.append(parent)

from utils import get_path


class LogisticModel:
    """
        Stores the data of known Hogwarts students
    """

    learning_rate = 0.05
    houses = ['Gryffindor', 'Hufflepuff', 'Ravenclaw', 'Slytherin']

    def __init__(self, datafile) -> None :
        df = pds.read_csv(datafile)
        df = df.dropna()
        self.pupils_houses = list(df['Hogwarts House'])

        numeric_df = df.select_dtypes(include=[np.number])
        self.features = list(numeric_df.columns)
        self.np_samples = numeric_df.values

        df['birth_year'] = (
            pds.to_datetime(df['Birthday'], errors='coerce')
               .dt.year
               .astype('float')
        )
        df['is_right_handed'] = (
            df['Best Hand']
              .str.lower()
              .map({'right': 1.0, 'left': 0.0})
              .astype('float')
        )

        samples_by_house = {
            house: group
            for house, group in df.groupby(list(df.columns)[1])
        }
        self.np_samples_by_house = {
            house: group[numeric_df.columns].values
            for house, group in samples_by_house.items()
        }

    @staticmethod
    def _count(feature) :
        return len(feature)

    @staticmethod
    def _mean(feature) :
        return np.sum(feature) / len(feature)

    def _std(self, feature) :
        return math.sqrt(np.sum((feature - self._mean(feature)) ** 2) / len(feature))

    @staticmethod
    def _mini(feature) :
        m = math.inf
        for feat in feature :
            if feat < m :
                m = feat
        return m

    @staticmethod
    def _maxi(feature) :
        m = -math.inf
        for feat in feature :
            if feat > m :
                m = feat
        return m

    @staticmethod
    def _quartile(feature, p, q) :
        feature = np.sort(feature)
        h = (len(feature) + 1 / 4) * p / q + 3 / 8
        return feature[math.floor(h)] + (h - math.floor(h)) * (feature[math.ceil(h)] - feature[math.floor(h)])

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
            fieldnames = ['House'] + ['intersect_' + ft for ft in features] + ['weight_' + ft for ft in features]
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerow(fieldnames)
            for house in LogisticRegression.houses:
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

    def describe(self) :
        print(f"{'':10}" + " | ".join(f"{feat:12.12}" for feat in self.features))
        print(f"{'Count':10}" + " | ".join(f"{self._count(self.np_samples[:, i]):12.5f}" for i in range(self.np_samples.shape[1])))
        print(f"{'Mean':10}" + " | ".join(f"{self._mean(self.np_samples[:, i]):12.5f}" for i in range(self.np_samples.shape[1])))
        print(f"{'Std':10}" + " | ".join(f"{self._std(self.np_samples[:, i]):12.5f}" for i in range(self.np_samples.shape[1])))
        print(f"{'Min':10}" + " | ".join(f"{self._mini(self.np_samples[:, i]):12.5f}" for i in range(self.np_samples.shape[1])))
        print(f"{'25%':10}" + " | ".join(f"{self._quartile(self.np_samples[:, i], 1, 4):12.5f}" for i in range(self.np_samples.shape[1])))
        print(f"{'50%':10}" + " | ".join(f"{self._quartile(self.np_samples[:, i], 2, 4):12.5f}" for i in range(self.np_samples.shape[1])))
        print(f"{'75%':10}" + " | ".join(f"{self._quartile(self.np_samples[:, i], 3, 4):12.5f}" for i in range(self.np_samples.shape[1])))
        print(f"{'Max':10}" + " | ".join(f"{self._maxi(self.np_samples[:, i]):12.5f}" for i in range(self.np_samples.shape[1])))

    def histogram(self, feature):
        colors = ['red', 'yellow', 'blue', 'green']
        if feature not in self.features :
            raise ValueError("Feature do not exist")

        ft_index = self.features.index(feature)

        fig = plt.figure()
        plt.hist([self.np_samples_by_house[house][:,ft_index] for house in self.np_samples_by_house], color=colors, label=LogisticRegression.houses)
        plt.xlabel('Grade')
        plt.ylabel('Frequency')
        plt.title(feature)
        plt.legend(loc='upper right')

        directory = get_path('figures/')
        if not os.path.isdir(directory):
            os.makedirs(directory)
        fig.savefig(directory + 'histogram.png')

    def scatter(self, feature1, feature2):
        if feature1 not in self.features or feature2 not in self.features :
            raise ValueError("Feature do not exist")

        fig = plt.figure()
        ft1_index = self.features.index(feature1)
        ft2_index = self.features.index(feature2)

        for house in self.np_samples_by_house :
            x = self.np_samples_by_house[house][:, ft1_index]
            y = self.np_samples_by_house[house][:, ft2_index]
            plt.scatter(x, y, label=house)
        plt.legend(loc='upper right')
        plt.xlabel(self.features[ft1_index])
        plt.ylabel(self.features[ft2_index])

        directory = get_path('figures/')
        if not os.path.isdir(directory):
            os.makedirs(directory)
        fig.savefig(directory + 'scatter.png')

    def pair_plot(self):
        colors = ['red', 'yellow', 'blue', 'green']

        figure, axis = plt.subplots(nrows=len(self.features), ncols=len(self.features), figsize=(50, 50))
        for i, ax in enumerate(axis.flat) :
            ft = int(i / len(self.features))
            j = i % len(self.features)
            if j > len(self.features) - 1 :
                continue
            if ft == j :
                ax.hist([self.np_samples_by_house[house][:, ft] for house in LogisticRegression.houses],
                         color=colors, label=LogisticRegression.houses)
                plt.setp(axis[-1, j], xlabel = self.features[ft])
                plt.setp(axis[j, 0], ylabel = self.features[j])

            else:
                idx = 0
                for house in self.np_samples_by_house :
                    x = self.np_samples_by_house[house][:, j]
                    y = self.np_samples_by_house[house][:, ft]
                    ax.scatter(x, y, s=5, color=colors[idx])
                    idx += 1

        figure.tight_layout()
        directory = get_path('figures/')
        if not os.path.isdir(directory):
            os.makedirs(directory)
        figure.savefig(directory + 'pair_plot.png')

    def print(self) :
        for house_name, house_data in self.np_samples_by_house.items() :
            print(house_name)
            print(house_data)

    def _derivative_cost_function(self, notes, binary_dict, intercept, weight):
        intercept_dict = {}
        weight_dict = {}

        for house in LogisticRegression.houses:
            linear_output = [intercept[house][i] + weight[house][i] * notes[i] for i in range(len(intercept[house]))]
            sig = [[self._sigmoid(linear_output[j][i]) for i in range(len(linear_output[j]))] for j in range(len(linear_output))]
            error = [[sig[j][i] - binary_dict[house][i] for i in range(len(sig[j]))] for j in range(len(sig))]
            intercept_dict[house] = [(1 / len(notes[i])) * np.sum(error[i]) for i in range(len(error))]
            weight_dict[house] = [(1 / len(notes[i])) * np.sum(error[i] * notes[i]) for i in range(len(error))]
        return intercept_dict, weight_dict

    def train(self, training_features):
        intercept = {'Gryffindor': [0.0] * len(training_features), 'Ravenclaw': [0.0] * len(training_features), 'Slytherin': [0.0] * len(training_features), 'Hufflepuff': [0.0] * len(training_features)}
        weight = {'Gryffindor': [0.0] * len(training_features), 'Ravenclaw': [0.0] * len(training_features), 'Slytherin': [0.0] * len(training_features), 'Hufflepuff': [0.0] * len(training_features)}

        features = self.features.copy()
        features.remove('Index')

        list_notes_raw = [np.array([self.np_samples[i][self.features.index(ft)] for i in range(len(self.np_samples))]) for ft in training_features]

        binary_dict = {'Gryffindor': [float(self.pupils_houses[i] == 'Gryffindor') for i in range(len(self.pupils_houses))],
                       'Ravenclaw': [float(self.pupils_houses[i] == 'Ravenclaw') for i in range(len(self.pupils_houses))],
                       'Hufflepuff': [float(self.pupils_houses[i] == 'Hufflepuff') for i in range(len(self.pupils_houses))],
                       'Slytherin': [float(self.pupils_houses[i] == 'Slytherin') for i in range(len(self.pupils_houses))]}

        notes = self._normalize(list_notes_raw)

        for _ in range(100):
            print(_)
            derivative_intercept_dict, derivative_weight_dict = self._derivative_cost_function(notes, binary_dict, intercept, weight)
            for house in LogisticRegression.houses:
                intercept[house] = [intercept[house][i] - (derivative_intercept_dict[house][i] * self.learning_rate) for i in range(len(derivative_intercept_dict[house]))]
                weight[house] = [weight[house][i] - (derivative_weight_dict[house][i] * self.learning_rate) for i in range(len(derivative_weight_dict[house]))]

        self._store_model(intercept, weight, training_features, 'models/models.csv')
        self.predict(get_path('models/models.csv'), get_path('datasets/dataset_test.csv'))
        return intercept, weight

    def predict(self, modelcsv, datacsv):
        # features to get from csv file
        intercept = {}
        weight = {}
        with open(modelcsv, 'r') as f:
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
        notes_raw = [np.array([self.np_samples[i][self.features.index(ft)] for i in range(len(self.np_samples))]) for ft in features]
        notes = self._normalize(notes_raw)
        for house in LogisticRegression.houses:
            linear_output = [intercept[house][i] + weight[house][i] * notes[i] for i in range(len(intercept[house]))]
            raw_scores = [self._sigmoid(linear_output[i]) for i in range(len(linear_output))]
            scores[house] = self._mean_scores(raw_scores)

        house_index = [np.argmax(np.array([scores[house][i] for house in LogisticRegression.houses])) for i in range(len(notes[0]))]
        houses = [LogisticRegression.houses[house_index[i]] for i in range(len(house_index))]
        self._store_prediction(houses, get_path('predictions/predictions.csv'))

    def graph(self, notes, are_gryffindor, weight, intercept):
        x1 = notes
        y1 = are_gryffindor
        x2 = [i * 0.01 for i in range(-100, 100)]
        y2 = [self._sigmoid(weight * x + intercept) for x in x2]

        plt.plot(x2, y2, label='sigmoid', color='blue')
        plt.scatter(x1, y1, label='are_gryffindor', color='red')
        plt.show()

