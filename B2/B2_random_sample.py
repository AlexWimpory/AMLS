import os
import shutil
import random


def get_file_list(input_dir):
    return [file for file in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, file))]


def get_random_files(file_list, n):
    return random.sample(file_list, n)


def copy_files(random_files, input_dir, output_dir):
    for file in random_files:
        shutil.copy(os.path.join(input_dir, file), output_dir)


def main(input_dir, output_dir, n):
    file_list = get_file_list(input_dir)
    random_files = get_random_files(file_list, n)
    copy_files(random_files, input_dir, output_dir)


if __name__ == '__main__':
    main('../Datasets/cartoon_set/img', '../Datasets/cartoon_set/random', 120)
