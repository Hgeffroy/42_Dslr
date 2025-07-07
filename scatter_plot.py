from DataSet import DataSet
import argparse


def main():
    parser = argparse.ArgumentParser(description='Describe statistical indicators of a dataset.')
    parser.add_argument('datafile')
    args = parser.parse_args()
    dataset = DataSet('datasets/' + args.datafile)
    dataset.scatter('Astronomy', 'Defense Against the Dark Arts')


if __name__ == "__main__":
    main()