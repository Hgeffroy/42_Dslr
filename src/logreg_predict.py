from classes.logistic_model import LogisticModel
from classes.dataset import Dataset
from utils import get_path
import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-d", "--data",
        type=str,
        default=get_path("datasets/dataset_test.csv"),
        help="The path to the data to predict CSV."
    )

    parser.add_argument(
        "-m", "--model",
        type=str,
        default=get_path("models/models.csv"),
        help="The path to the model CSV."
    )

    return parser

def main():
    parser = build_parser()
    args = parser.parse_args()
    dataset = Dataset(args.data)
    model = LogisticModel()
    model.predict(args.model, dataset)


if __name__ == "__main__":
    main()