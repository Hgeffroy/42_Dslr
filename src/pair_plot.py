import argparse
import pandas as pds

from LogisticRegression import LogisticRegression
from utils import get_path

def build_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
        
    parser.add_argument(
        "-d", "--data",
        type=str,
        default=get_path("datasets/dataset_train.csv"),
        help="The path to the data training CSV."
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

    dataset = LogisticRegression(args.data)
    dataset.pair_plot()


if __name__ == "__main__":
    main()