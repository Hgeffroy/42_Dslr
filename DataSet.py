import os
import csv
import numpy as np
import math
import matplotlib.pyplot as plt

def keep_numerical(features, samples) :
    ref = samples[0]
    for i in reversed(range(len(ref))) :
        try :
            float(ref[i])
        except :
            for sample in samples :
                del sample[i]
            del features[i]

def count(feature) :
    return len(feature)

def mean(feature) :
    return np.sum(feature) / len(feature)

def std(feature) :
    return math.sqrt(np.sum((feature - mean(feature)) ** 2) / len(feature))

def mini(feature) :
    m = math.inf
    for feat in feature :
        if feat < m :
            m = feat
    return m

def maxi(feature) :
    m = -math.inf
    for feat in feature :
        if feat > m :
            m = feat
    return m

def quantile(feature, p, q) :
    feature = np.sort(feature)
    h = (len(feature) + 1 / 4) * p / q + 3 / 8
    return feature[math.floor(h)] + (h - math.floor(h)) * (feature[math.ceil(h)] - feature[math.floor(h)])

class DataSet :
    """
        Stores the data of known Hogwarts students
    """

    np_samples = np.array([])
    samples = []
    features = []
    samples_by_house = {'Gryffindor': [], 'Hufflepuff': [],
                        'Ravenclaw': [], 'Slytherin': []}
    np_samples_by_house = {'Gryffindor': np.array([]), 'Hufflepuff': np.array([]),
                           'Ravenclaw': np.array([]), 'Slytherin': np.array([])}

    def __init__(self, datafile) -> None :
        with open(datafile) as cvsfile :
            datareader = csv.reader(cvsfile)
            self.features = next(datareader)
            for row in datareader :
                if '' in row :
                    pass
                else :
                    self.samples.append(row)
                    self.samples_by_house[row[1]].append(row)
            tmp_samples = self.samples[:]
        keep_numerical(self.features, tmp_samples)
        self.np_samples = np.array(tmp_samples, dtype='d')
        for house in self.np_samples_by_house :
            self.np_samples_by_house[house] = np.array(self.samples_by_house[house], dtype='d')

    def describe(self) :
        print(f"{'':10}" + " | ".join(f"{feat:12.12}" for feat in self.features))
        print(f"{'Count':10}" + " | ".join(f"{count(self.np_samples[:, i]):12.5f}" for i in range(self.np_samples.shape[1])))
        print(f"{'Mean':10}" + " | ".join(f"{mean(self.np_samples[:, i]):12.5f}" for i in range(self.np_samples.shape[1])))
        print(f"{'Std':10}" + " | ".join(f"{std(self.np_samples[:, i]):12.5f}" for i in range(self.np_samples.shape[1])))
        print(f"{'Min':10}" + " | ".join(f"{mini(self.np_samples[:, i]):12.5f}" for i in range(self.np_samples.shape[1])))
        print(f"{'25%':10}" + " | ".join(f"{quantile(self.np_samples[:, i], 1, 4):12.5f}" for i in range(self.np_samples.shape[1])))
        print(f"{'50%':10}" + " | ".join(f"{quantile(self.np_samples[:, i], 2, 4):12.5f}" for i in range(self.np_samples.shape[1])))
        print(f"{'75%':10}" + " | ".join(f"{quantile(self.np_samples[:, i], 3, 4):12.5f}" for i in range(self.np_samples.shape[1])))
        print(f"{'Max':10}" + " | ".join(f"{maxi(self.np_samples[:, i]):12.5f}" for i in range(self.np_samples.shape[1])))


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

        directory = 'figures/'
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

        directory = 'figures/'
        if not os.path.isdir(directory):
            os.makedirs(directory)
        fig.savefig(directory + 'scatter.png')

    def matrix(self):
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
        directory = 'figures/'
        if not os.path.isdir(directory):
            os.makedirs(directory)
        figure.savefig(directory + 'matrix.png')

    def print(self) :
        for house_name, house_data in self.samples_by_house.items() :
            print(house_name)
            print(house_data)

