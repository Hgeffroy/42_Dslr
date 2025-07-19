from logistic_regression import LogisticRegression
from utils import get_path

def main() :
    dataset = LogisticRegression(get_path("datasets/dataset_train.csv"))
    theta = dataset.gradient_descent()

    print(theta)


if __name__ == "__main__":
    main()