import argparse
import pandas as pds

from DataSet import DataSet
from utils import get_path

def build_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
        
    parser.add_argument(
        "-d", "--data",
        type=str,
        default=get_path("datasets/dataset_train.csv"),
        help="The path to the data training CSV."
    )
    parser.add_argument(
        "-f", "--feature",
        type=str,
        default="Potions",
        help="The feature that we want to compare."
    )

    return parser

def main():
    parser = build_parser(description='Describe statistical indicators of a dataset.')
    args = parser.parse_args()

    try:
        df = pds.read_csv(args.data)
    except FileNotFoundError:
        print(f"Error: File not found ({args.data}).")
        return
    except pds.errors.ParserError as e:
        print(f"Error: Can't parse the CSV: {e}")
        return
    
    if df.shape[1] < 2:
        print("Error: The CSV should have at least two columns.")
        return

    dataset = DataSet(args.data)
    dataset.histogram(args.feature)


if __name__ == "__main__":
    main()
