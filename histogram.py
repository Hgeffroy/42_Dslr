from DataSet import DataSet
import argparse


def main():
    parser = argparse.ArgumentParser(description='Describe statistical indicators of a dataset.')
    parser.add_argument('datafile')
    parser.add_argument('feature')
    args = parser.parse_args()
    dataset = DataSet('datasets/' + args.datafile)
    dataset.histogram(args.feature)


if __name__ == "__main__":
    main()
