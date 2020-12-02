import csv


class GroundtruthReader:
    def __init__(self, groundtruth_filename):
        self.groundtruth_records = {}
        with open(groundtruth_filename, 'r') as fin:
            reader = csv.reader(fin, delimiter='\t')
            for row in reader:
                self.groundtruth_records[row[1]] = self.smiling_or_not_smiling(row[3])

    def lookup_filename(self, filename):
        return self.groundtruth_records[filename]

    @staticmethod
    def smiling_or_not_smiling(number):
        if number == '1':
            return 'smiling'
        elif number == '-1':
            return 'not_smiling'
        else:
            return None


if __name__ == '__main__':
    gtp = GroundtruthReader('../Datasets/celeba/labels.csv')
    print(gtp.lookup_filename('3.jpg'))
