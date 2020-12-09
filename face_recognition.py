import os
import random

import pandas as pd
import cv2
import numpy as np

# Face dataset constants
width_i = 2
height_i = 3
labels_path_faces = "C:/Users/naz/data/Object_Detection/faces/labels/wider_face_val_bbx_gt.txt"
files_path_faces = "C:/Users/naz/data/Object_Detection/faces/files/"

# Choose train or test part of the VGGFace dataset
mode = 'train'

vgg_path = "C:/Users/naz/data/Object_Detection/faces/VGG2_" + mode

vgg_persons = [person for person in os.listdir(vgg_path)]
head_annotations = pd.read_csv(os.path.join(vgg_path, f'bb_{mode}_vgg2.csv'))

# Label file constants
x_1_i = 0
y_1_i = 1
x_2_i = 2
y_2_i = 3
class_i = 4
filename_i = 5
name_index = 5


def get_face_annotation(annotation_path: str):
    """Get the annotation for a face of the VGGFace dataset
    :param annotation_path: Path to the image file

    :return: Annotation for a VGGFace face in X, Y, W, H
    """
    annotation_key = annotation_path.replace("\\", "/")[:annotation_path.find('.jpg')]
    annot = head_annotations.loc[head_annotations['NAME_ID'] == annotation_key]
    return np.squeeze(annot[['X', 'Y', 'W', 'H']].values).astype(np.int64)


def get_random_VGG_face():
    """ Get random face from the VGGFace dataset

    :return: VGGFace image and corresponding annotation path
    """
    try:
        i = random.randint(0, len(vgg_persons) - 1)
        persons_path = vgg_persons[i]
        person_instances = [instance for instance in os.listdir(os.path.join(vgg_path, persons_path))]
        ii = random.randint(0, len(person_instances) - 1)
        person_instance = person_instances[ii]
        full_path = os.path.join(vgg_path, persons_path, person_instance)

        img = cv2.imread(full_path)

        annot_path = os.path.join(persons_path, person_instance)

    except:
        print('Directory Error, trying to load VGG face again..')
        return get_random_VGG_face()

    return img, annot_path


# Deprecated generator for WiderFace dataset
def get_random_cropped_head(skip_heads=100):
    with open(labels_path_faces, 'r') as f:

        annotation_limit = 10000
        annotation_index = 0

        for i, line in enumerate(f):
            if i == 0 or annotation_index == (annotation_limit + 2):
                file_name = line.strip()
                annotation_index = 1

            elif annotation_index == 1:
                annotation_limit = int(line.strip())
                annotation_index += 1

            else:
                annotation_index += 1

                gt = line.split()
                x_1 = int(gt[x_1_i])
                y_1 = int(gt[y_1_i])
                width = int(gt[width_i])
                height = int(gt[height_i])

                if width > 80 and height > 100 and i > skip_heads and height < 400:
                    img = cv2.imread(os.path.join(files_path_faces, file_name))
                    head_cropped = img[y_1:y_1 + height, x_1:x_1 + width]

                    yield head_cropped