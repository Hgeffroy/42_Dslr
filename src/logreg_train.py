from classes.logistic_model import LogisticModel
from classes.dataset import Dataset
from utils import get_path
import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-d", "--data",
        type=str,
        default=get_path("datasets/dataset_train.csv"),
        help="The path to the data training CSV."
    )

    parser.add_argument(
        "-f", "--features",
        type=str,
        default="Herbology,Defense Against the Dark Arts,Ancient Runes,Astronomy",
        help="Features to use for training separated by a comma.",
    )

    parser.add_argument(
        "-b", "--batch_size",
        type=int,
        default=0,
        help="Batch size while training. If given a value <= 0, will train with the biggest batch.",
    )

    parser.add_argument(
        "-i", "--iterations",
        type=int,
        default=1000,
        help="Number of iterations for training.",
    )

    parser.add_argument(
        "-l", "--learning_rate",
        type=int,
        default=0.5,
        help="Learning rate during training.",
    )

    return parser

def check_args(args, dataset):
    for ft in args.features.split(","):
        if ft not in dataset.get_features():
            raise ValueError(f'Feature {ft} does not exist')

def main():
    parser = build_parser()
    args = parser.parse_args()
    dataset = Dataset(args.data)
    model = LogisticModel()
    model.train(dataset, args.features.split(","), args.batch_size, args.iterations, args.learning_rate)


if __name__ == "__main__":
    main()
