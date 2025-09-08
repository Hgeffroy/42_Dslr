import pandas as pds
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import math

current = os.path.dirname(os.path.realpath(__file__))

# Getting the parent directory name
# where the current directory is present.
parent = os.path.dirname(current)

# adding the parent directory to
# the sys.path.
sys.path.append(parent)

from utils import get_path


class Dataset:
    houses = ['Gryffindor', 'Hufflepuff', 'Ravenclaw', 'Slytherin']

    def __init__(self, datafile) -> None:
        df = pds.read_csv(datafile)
        df = df.dropna(subset=[c for c in df.columns if c != 'Hogwarts House'])
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


    def __repr__(self):
        # samples/features
        n_samples = int(self.np_samples.shape[0]) if hasattr(self, "np_samples") else 0
        n_features = int(self.np_samples.shape[1]) if hasattr(self, "np_samples") and self.np_samples.size else 0

        # class balance (order aligned with Dataset.houses)
        house_counts = []
        if hasattr(self, "np_samples_by_house") and isinstance(self.np_samples_by_house, dict):
            for h in Dataset.houses:
                house_counts.append(len(self.np_samples_by_house.get(h, [])))
        else:
            house_counts = [0, 0, 0, 0]

        # feature preview
        feat_preview_len = 5
        feats = getattr(self, "features", [])
        feat_preview = ", ".join(feats[:feat_preview_len]) + ("..." if len(feats) > feat_preview_len else "")

        # build a compact repr
        counts_str = ", ".join(f"{h}:{c}" for h, c in zip(Dataset.houses, house_counts))
        return (f"<Dataset {n_samples} samples, {n_features} features | "
                f"classes[{counts_str}] | feats[{feat_preview}]>")

    def __str__(self):
        return self.__repr__()

    def __len__(self):
        return int(self.np_samples.shape[0]) if hasattr(self, "np_samples") else 0

    @staticmethod
    def _count(feature):
        return len(feature)

    @staticmethod
    def _mean(feature):
        return np.sum(feature) / len(feature)

    def _std(self, feature):
        return math.sqrt(
            np.sum((feature - self._mean(feature)) ** 2) / len(feature))


    @staticmethod
    def _mini(feature):
        m = math.inf
        for feat in feature:
            if feat < m:
                m = feat
        return m

    @staticmethod
    def _maxi(feature):
        m = -math.inf
        for feat in feature:
            if feat > m:
                m = feat
        return m

    @staticmethod
    def _range(feature):
        return Dataset._maxi(feature) - Dataset._mini(feature)

    @staticmethod
    def _quartile(feature, p, q):
        feature = np.sort(feature)
        h = (len(feature) + 1 / 4) * p / q + 3 / 8
        return feature[math.floor(h)] + (h - math.floor(h)) * (
                    feature[math.ceil(h)] - feature[math.floor(h)])

    @staticmethod
    def _median(feature):
        feature = np.sort(feature)
        if len(feature) % 2 == 0:
            return (feature[int(len(feature) / 2)] + feature[int(len(feature) / 2) - 1]) / 2
        else:
            return feature[int(len(feature) / 2)]

    def get_features(self):
        return self.features

    def get_pupils_houses(self):
        return self.pupils_houses

    def get_samples(self):
        return self.np_samples

    def describe(self):
        print(f"{'':10}" + " | ".join(f"{feat:12.12}" for feat in self.features))
        print(f"{'Count':10}" + " | ".join(f"{self._count(self.np_samples[:, i]):12.5f}" for i in range(self.np_samples.shape[1])))
        print(f"{'Mean':10}" + " | ".join(f"{self._mean(self.np_samples[:, i]):12.5f}" for i in range(self.np_samples.shape[1])))
        print(f"{'Median':10}" + " | ".join(f"{self._median(self.np_samples[:, i]):12.5f}" for i in range(self.np_samples.shape[1])))
        print(f"{'Std':10}" + " | ".join(f"{self._std(self.np_samples[:, i]):12.5f}" for i in range(self.np_samples.shape[1])))
        print(f"{'Min':10}" + " | ".join(f"{self._mini(self.np_samples[:, i]):12.5f}" for i in range(self.np_samples.shape[1])))
        print(f"{'25%':10}" + " | ".join(f"{self._quartile(self.np_samples[:, i], 1, 4):12.5f}" for i in range(self.np_samples.shape[1])))
        print(f"{'50%':10}" + " | ".join(f"{self._quartile(self.np_samples[:, i], 2, 4):12.5f}" for i in range(self.np_samples.shape[1])))
        print(f"{'75%':10}" + " | ".join(f"{self._quartile(self.np_samples[:, i], 3, 4):12.5f}" for i in range(self.np_samples.shape[1])))
        print(f"{'Max':10}" + " | ".join(f"{self._maxi(self.np_samples[:, i]):12.5f}" for i in range(self.np_samples.shape[1])))
        print(f"{'Range':10}" + " | ".join(f"{self._range(self.np_samples[:, i]):12.5f}" for i in range(self.np_samples.shape[1])))

    def histogram(self, feature, filename):
        colors = ['red', 'yellow', 'blue', 'green']
        if feature not in self.features:
            raise ValueError("Feature do not exist")

        ft_index = self.features.index(feature)

        fig = plt.figure()
        plt.hist([self.np_samples_by_house[house][:,ft_index] for house in self.np_samples_by_house], color=colors, label=Dataset.houses)
        plt.xlabel('Grade')
        plt.ylabel('Frequency')
        plt.title(feature)
        plt.legend(loc='upper right')

        directory = get_path('figures/')
        if not os.path.isdir(directory):
            os.makedirs(directory)
        fig.savefig(directory + filename + '.png')

    def scatter(self, feature1, feature2, filename):
        if feature1 not in self.features or feature2 not in self.features:
            raise ValueError("Feature do not exist")

        fig = plt.figure()
        ft1_index = self.features.index(feature1)
        ft2_index = self.features.index(feature2)

        for house in self.np_samples_by_house:
            x = self.np_samples_by_house[house][:, ft1_index]
            y = self.np_samples_by_house[house][:, ft2_index]
            plt.scatter(x, y, label=house)
        plt.legend(loc='upper right')
        plt.xlabel(self.features[ft1_index])
        plt.ylabel(self.features[ft2_index])

        directory = get_path('figures/')
        if not os.path.isdir(directory):
            os.makedirs(directory)
        fig.savefig(directory + filename + '.png')

    def pair_plot(self):
        colors = ['red', 'yellow', 'blue', 'green']

        figure, axis = plt.subplots(nrows=len(self.features), ncols=len(self.features), figsize=(50, 50))
        for i, ax in enumerate(axis.flat):
            ft = int(i / len(self.features))
            j = i % len(self.features)
            if j > len(self.features) - 1:
                continue
            if ft == j:
                ax.hist([self.np_samples_by_house[house][:, ft] for house in Dataset.houses],
                         color=colors, label=Dataset.houses)
                plt.setp(axis[-1, j], xlabel = self.features[ft])
                plt.setp(axis[j, 0], ylabel = self.features[j])

            else:
                idx = 0
                for house in self.np_samples_by_house:
                    x = self.np_samples_by_house[house][:, j]
                    y = self.np_samples_by_house[house][:, ft]
                    ax.scatter(x, y, s=5, color=colors[idx])
                    idx += 1

        figure.tight_layout()
        directory = get_path('figures/')
        if not os.path.isdir(directory):
            os.makedirs(directory)
        figure.savefig(directory + 'pair_plot.png')
