import csv

"""Finds the label for an image in the labels.csv file"""


class GroundtruthReader:
    def __init__(self, groundtruth_filename):
        self.groundtruth_records = {}
        with open(groundtruth_filename, 'r') as fin:
            reader = csv.reader(fin, delimiter='\t')
            for row in reader:
                self.groundtruth_records[row[1]] = self.male_or_female(row[2])

    def lookup_filename(self, filename):
        return self.groundtruth_records[filename]

    @staticmethod
    def male_or_female(number):
        """Renames label 1 and -1 to male and female"""
        if number == '1':
            return 'male'
        elif number == '-1':
            return 'female'
        else:
            return None


if __name__ == '__main__':
    gtp = GroundtruthReader('../Datasets/celeba/labels.csv')
    print(gtp.lookup_filename('3.jpg'))
