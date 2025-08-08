from classes.logistic_model import LogisticModel
from classes.dataset import Dataset
from utils import get_path


def main():
    dataset = Dataset(get_path("datasets/dataset_train.csv"))
    model = LogisticModel()
    model.predict(get_path("models/models.csv"), dataset)


if __name__ == "__main__":
    main()