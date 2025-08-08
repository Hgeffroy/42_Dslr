from classes.logistic_model import LogisticModel
from classes.dataset import Dataset
from utils import get_path


def main():
    dataset = Dataset(get_path("datasets/dataset_train.csv"))
    model = LogisticModel(dataset)
    # model.predict(get_path("datasets/dataset_test.csv"))


if __name__ == "__main__":
    main()