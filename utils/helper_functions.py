"""
helper fuctions, create directoriees and move files

"""

import os
import shutil
import numpy as np


def dataset_directories(splits, folders, data_path=None):
    """
    Create directories for each dataset split object and annotations
    """

    for split in splits:
        for folder in folders:
            try:
                os.makedirs(f'{data_path}/{split}/{folder}')
            except FileExistsError:
                pass


def list_to_text(text_path, values_list):
    """
    Function that converts the content of a list to a text file
    """
    assert isinstance(text_path, str), 'must be a string'
    with open(text_path, 'w', encoding='utf-8') as textfile:
        for element in values_list:
            textfile.write(element + "\n")


def update_annotations(filename, label):
    """
    Updates the annotations with the label class - yolo format
    """

    with open(filename, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    new_lines = [str(label) + line[1:] for line in lines]

    with open(filename, 'w', encoding='utf-8') as file:
        file.writelines(new_lines)


def copy_files(files_to_copy, dest_path):
    """
    Copy files to a directory
    """
    for f in files_to_copy:
        try:
            shutil.copy(f, dest_path)
        except shutil.Error:
            print(f'file {f} not copied')


def move_files(files_to_move, dest_path=None):
    """
    Moves files to directory
    """
    assert isinstance(files_to_move, (list, np.ndarray)), 'must be a list'
    # assert isinstance(dest_path, (str, PathLike))
    for f in files_to_move:
        try:
            shutil.move(f,  dest_path)
        except shutil.Error:
            print(f'file {f} not moved')