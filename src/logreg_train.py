from classes.logistic_model import LogisticModel
from classes.dataset import Dataset
from utils import get_path
import argparse
import sys

def check_args(args, dataset):
    for ft in args.features.split(","):
        if ft not in dataset.get_features():
            raise ValueError(f'Feature {ft} does not exist')

def range_limited_float_type(arg):
    """ Type function for argparse - a int within some predefined bounds """
    try:
        maxfloat = sys.float_info.max
        f = float(arg)
    except ValueError:
        raise argparse.ArgumentTypeError("Must be a float number")
    if f <= 0 or f >= maxfloat:
        raise argparse.ArgumentTypeError("Argument must be < " + str(maxfloat) + " and > " + str(0))
    return f

def range_limited_int_type(arg):
    """ Type function for argparse - a int within some predefined bounds """
    try:
        maxint = sys.maxsize
        f = int(arg)
    except ValueError:    
        raise argparse.ArgumentTypeError("Must be a int number")
    if f <= 0 or f >= maxint:
        raise argparse.ArgumentTypeError("Argument must be < " + str(maxint) + " and > " + str(0))
    return f

def range_limited_int_type_zero(arg):
    """ Type function for argparse - a int within some predefined bounds """
    try:
        maxint = sys.maxsize
        f = int(arg)
    except ValueError:    
        raise argparse.ArgumentTypeError("Must be a int number")
    if f < 0 or f >= maxint:
        raise argparse.ArgumentTypeError("Argument must be < " + str(maxint) + " and > " + str(0))
    return f

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
        type=range_limited_int_type_zero,
        default=0,
        help="Batch size while training. 0 = max-size",
    )

    parser.add_argument(
        "-i", "--iterations",
        type=range_limited_int_type,
        default=1000,
        help="Number of iterations for training.",
    )

    parser.add_argument(
        "-l", "--learning_rate",
        type=range_limited_float_type,
        default=0.5,
        help="Learning rate during training.",
    )

    return parser

def main():
    parser = build_parser()
    args = parser.parse_args()
    dataset = Dataset(args.data)
    check_args(args, dataset)
    model = LogisticModel()
    model.train(dataset, args.features.split(","), args.batch_size, args.iterations, args.learning_rate)


if __name__ == "__main__":
    main()
