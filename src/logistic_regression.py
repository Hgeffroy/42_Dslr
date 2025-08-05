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

    features = []
    houses = []
    np_samples = np.array([])
    np_samples_by_house = {'Gryffindor': np.array([]), 'Hufflepuff': np.array([]),
                           'Ravenclaw': np.array([]), 'Slytherin': np.array([])}

    def __init__(self, datafile) -> None :
        df = pds.read_csv(datafile)
        df = df.dropna()

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

        self.houses = list(df['Hogwarts House'])

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
        return [values[i] / m[i] for i in range(len(min_values))]

    @staticmethod
    def _sigmoid(z):
        return [1.0 / (1.0 + np.exp(-x)) for x in z]

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
                ax.hist([self.np_samples_by_house[house][:, ft] for house in houses],
                         color=colors, label=houses)
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

    def _derivative_cost_function(self, notes, are_gryffindor, intercept, weight):
        linear_output = [intercept[i] + weight[i] * notes[i] for i in range(len(intercept))]
        sig = self._sigmoid(linear_output)
        error = [sig[i] - are_gryffindor for i in range(len(sig))]

        derivative_intercept = [(1 / len(notes[i])) * np.sum(error[i]) for i in range(len(error))]
        derivative_weight = [(1 / len(notes[i])) * np.sum(error[i] * notes[i]) for i in range(len(error))]

        return derivative_intercept, derivative_weight

    def gradient_descent(self):
        # Passer le nom des features pour la regression logistique en parametre
        intercept = [0.0, 0.0]
        weight = [0.0, 0.0]

        features = self.features.copy()
        features.remove('Index')

        notes_raw = [[self.np_samples[i][self.features.index('Herbology')] for i in range(len(self.np_samples))],
                    [self.np_samples[i][self.features.index('Defense Against the Dark Arts')] for i in range(len(self.np_samples))]]
        notes_raw = np.asarray(notes_raw, dtype=float)
        are_gryffindor = [float(self.houses[i] == 'Gryffindor') for i in range(len(self.houses))]

        notes = self._normalize(notes_raw)

        for _ in range(100000):
            derivative_intercept, derivative_weight = self._derivative_cost_function(notes, are_gryffindor, intercept, weight)
            intercept = [intercept[i] - (derivative_intercept[i] * self.learning_rate) for i in range(len(derivative_intercept))]
            weight = [weight[i] - (derivative_weight[i] * self.learning_rate) for i in range(len(derivative_weight))]

        self.predict(intercept, weight, [])
        return intercept, weight

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

    def predict(self, intercept, weight, features):
        # Faire la moyenne des scores correspondant a chaque feature
        # Ensuite prendre la moyenne la plus haute
        features = ['Herbology', 'Defense Against the Dark Arts']
        notes_raw = np.array([[self.np_samples[i][self.features.index(ft)] for i in range(len(self.np_samples))] for ft in features])
        notes = self._normalize(notes_raw)
        linear_output = [intercept[i] + weight[i] * notes[i] for i in range(len(intercept))]
        raw_scores = self._sigmoid(linear_output)
        scores = self._mean_scores(raw_scores)

    def graph(self, notes, are_gryffindor, weight, intercept):
        x1 = notes
        y1 = are_gryffindor
        x2 = [i * 0.01 for i in range(-100, 100)]
        y2 = [self._sigmoid(weight * x + intercept) for x in x2]

        plt.plot(x2, y2, label='sigmoid', color='blue')
        plt.scatter(x1, y1, label='are_gryffindor', color='red')
        plt.show()
