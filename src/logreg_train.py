from classes.logistic_model import LogisticModel
from utils import get_path

def main() :
    dataset = LogisticModel(get_path("datasets/dataset_train.csv"))
    theta = dataset.train(['Herbology', 'Defense Against the Dark Arts'])

    print(theta)


if __name__ == "__main__":
    main()