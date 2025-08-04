import os
import numpy as np
import pandas as pds
import math
import matplotlib.pyplot as plt

from utils import get_path


class LogisticRegression :
    """
        Stores the data of known Hogwarts students
    """

    learning_rate = 0.01

    min = 0.0
    max = 0.0

    features = []
    houses = []
    np_samples = np.array([])
    np_samples_by_house = {'Gryffindor': np.array([]), 'Hufflepuff': np.array([]),
                           'Ravenclaw': np.array([]), 'Slytherin': np.array([])}

    def __init__(self, datafile) -> None :
        df = pds.read_csv(datafile)
        df = df.dropna()
        numeric_df = df.select_dtypes(include=[np.number])

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

        self.features = list(df.columns)
        self.features = list(numeric_df.columns)
        self.np_samples = numeric_df.values

        samples_by_house = {
            house: group
            for house, group in df.groupby(self.features[1])
        }
        self.np_samples_by_house = {
            house: group[numeric_df.columns].values
            for house, group in self.samples_by_house.items()
        }

        self.houses = list(df['Hogwarts House'])

    def _count(self, feature) :
        return len(feature)

    def _mean(self, feature) :
        return np.sum(feature) / len(feature)

    def _std(self, feature) :
        return math.sqrt(np.sum((feature - self._mean(feature)) ** 2) / len(feature))

    def _mini(self, feature) :
        m = math.inf
        for feat in feature :
            if feat < m :
                m = feat
        return m

    def _maxi(self, feature) :
        m = -math.inf
        for feat in feature :
            if feat > m :
                m = feat
        return m

    def _quartile(self, feature, p, q) :
        feature = np.sort(feature)
        h = (len(feature) + 1 / 4) * p / q + 3 / 8
        return feature[math.floor(h)] + (h - math.floor(h)) * (feature[math.ceil(h)] - feature[math.floor(h)])

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
        houses = ['Gryffindor', 'Hufflepuff', 'Ravenclaw', 'Slytherin']

        fig = plt.figure()
        plt.hist([self.np_samples_by_house[house][:,ft_index] for house in self.np_samples_by_house], color=colors, label=houses)
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
        houses = ['Gryffindor', 'Hufflepuff', 'Ravenclaw', 'Slytherin']

        figure, axis = plt.subplots(nrows=len(self.features), ncols=len(self.features), figsize=(50, 50))
        for i, ax in enumerate(axis.flat) :
            ft = int(i / len(self.features))
            j = i % len(self.features)
            if j > len(self.features) - 1 :
                continue
            if ft == j :
                ax.hist([self.np_samples_by_house[house][:, ft] for house in self.np_samples_by_house],
                         color=colors, label=houses)
                plt.setp(axis[-1, j], xlabel = self.features[j])
                plt.setp(axis[j, 0], ylabel = self.features[ft])

            else:
                idx = 0
                for house in self.np_samples_by_house :
                    x = self.np_samples_by_house[house][:, ft]
                    y = self.np_samples_by_house[house][:, j]
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


    def normalize(self, values, min_values, max_values):
        if min_values == max_values:
            return np.zeros_like(values)
        return (values - min_values) / (max_values - min_values)


    def _sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-z))


    def _derivative_cost_function(self, notes, areGryffindor, intercept, weight):
        linear_output = intercept + weight * notes
        sig = self._sigmoid(linear_output)
        error = sig - areGryffindor

        m = len(notes)
        derivative_intercept = (1 / m) * np.sum(error)
        derivative_weight = (1 / m) * np.sum(error * notes)

        return derivative_intercept, derivative_weight

    def gradient_descent(self):
        intercept = 0.0
        weight = 0.0

        features = self.features.copy()
        features.remove('Index')

        notes_raw = [self.np_samples[i][self.features.index('Transfiguration')] for i in range(len(self.np_samples))]
        areGryffindor = [float(self.houses[i] == 'Gryffindor') for i in range(len(self.houses))]

        notes_raw = np.asarray(notes_raw, dtype=float)
        areGryffindor = np.asarray(areGryffindor, dtype=float)

        self.min = notes_raw.min()
        self.max = notes_raw.max()
        notes = self.normalize(notes_raw, self.min, self.max)

        for _ in range(2000):
            derivative_intercept, derivative_weight = self._derivative_cost_function(notes, areGryffindor, intercept, weight)
            intercept -= derivative_intercept * self.learning_rate
            weight -= derivative_weight * self.learning_rate

        return intercept, weight
