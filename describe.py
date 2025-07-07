import matplotlib.pyplot as plt
import argparse
from DataSet import DataSet

def main() :
    parser = argparse.ArgumentParser(description='Describe statistical indicators of a dataset.')
    parser.add_argument('datafile')
    args = parser.parse_args()
    dataset = DataSet(args.datafile)
    dataset.describe()

if __name__ == "__main__":
    main()
