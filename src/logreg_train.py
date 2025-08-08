from classes.logistic_model import LogisticModel
from classes.dataset import Dataset
from utils import get_path


def main():
    dataset = Dataset(get_path("datasets/dataset_train.csv"))
    model = LogisticModel(dataset)
    theta = model.train(['Herbology', 'Defense Against the Dark Arts'])

    print(theta)


if __name__ == "__main__":
    main()
