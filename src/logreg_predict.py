from classes.logistic_model import LogisticModel
from utils import get_path

def main() :
    dataset = LogisticModel(get_path("datasets/dataset_train.csv"))
    dataset.predict(get_path("datasets/dataset_test.csv"))



if __name__ == "__main__":
    main()