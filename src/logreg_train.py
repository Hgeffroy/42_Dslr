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
        default="Herbology,Defense Against the Dark Arts",
        help="Features to use for training separated by a comma.",
    )

    parser.add_argument(
        "-b", "--batch_size",
        type=int,
        default=0,
        help="Batch size while training. 1 for stochastic GD.",
    )

    parser.add_argument(
        "-i", "--iterations",
        type=int,
        default=1000,
        help="Number of iterations for training.",
    )

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    dataset = Dataset(args.data)
    model = LogisticModel()
    model.train(dataset, args.features.split(","), args.batch_size, args.iterations)


if __name__ == "__main__":
    main()
