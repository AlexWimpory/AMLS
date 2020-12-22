import csv


"""Finds the label for an image in the labels.csv file"""


class GroundtruthReader:
    def __init__(self, groundtruth_filename):
        self.groundtruth_records = {}
        with open(groundtruth_filename, 'r') as fin:
            reader = csv.reader(fin, delimiter='\t')
            for row in reader:
                self.groundtruth_records[row[3]] = row[2]

    def lookup_filename(self, filename):
        return self.groundtruth_records[filename]


if __name__ == '__main__':
    gtp = GroundtruthReader('../Datasets/cartoon_set/labels.csv')
    print(gtp.lookup_filename('3.png'))
