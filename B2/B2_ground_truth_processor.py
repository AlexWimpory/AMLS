import csv


class GroundtruthReader:
    def __init__(self, groundtruth_filename):
        self.groundtruth_records = {}
        with open(groundtruth_filename, 'r') as fin:
            reader = csv.reader(fin, delimiter='\t')
            for row in reader:
                self.groundtruth_records[row[3]] = self.number_to_colour(row[1])

    def lookup_filename(self, filename):
        return self.groundtruth_records[filename]

    @staticmethod
    def number_to_colour(number):
        if number == '0':
            return 'brown'
        elif number == '1':
            return 'blue'
        elif number == '2':
            return 'green'
        elif number == '3':
            return 'grey'
        elif number == '4':
            return 'black'
        else:
            return None

if __name__ == '__main__':
    gtp = GroundtruthReader('../Datasets/cartoon_set/labels.csv')
    print(gtp.lookup_filename('3.png'))
