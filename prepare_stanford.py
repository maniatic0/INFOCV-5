from pathlib import Path
from datetime import datetime

from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras import Sequential
from tensorflow.keras.layers.experimental.preprocessing import Rescaling, RandomContrast

import numpy as np

import colab_test
from utils import createIfNecessaryDir

ROOT = Path(".") / "datasets"

STANFORD = ROOT / "Stanford40"

STANFORD_TRAINING = STANFORD / "training"
createIfNecessaryDir(STANFORD_TRAINING)

STANFORD_TESTING = STANFORD / "testing"
createIfNecessaryDir(STANFORD_TESTING)

BATCH_SIZE = 32
IMAGE_SIZE = (128, 128)
IMAGE_SHAPE = (128, 128, 3)
STANFORD_NO_CLASSES = 40
VAL_SPLIT = 0.1


def preprocessStanfordData():
    with open(STANFORD / "ImageSplits/train.txt", "r") as f:
        train_files = list(map(str.strip, f.readlines()))
        train_labels = ["_".join(name.split("_")[:-1]) for name in train_files]
        print(f"Train files ({len(train_files)}):\n\t{train_files}")
        print(f"Train labels ({len(train_labels)}):\n\t{train_labels}\n")

    with open(STANFORD / "ImageSplits/test.txt", "r") as f:
        test_files = list(map(str.strip, f.readlines()))
        test_labels = ["_".join(name.split("_")[:-1]) for name in test_files]
        print(f"Test files ({len(test_files)}):\n\t{test_files}")
        print(f"Test labels ({len(test_labels)}):\n\t{test_labels}\n")

    action_categories = sorted(
        list(set(["_".join(name.split("_")[:-1]) for name in train_files]))
    )
    print(f"Action categories ({len(action_categories)}):\n{action_categories}")

    # Category folder creation
    if not colab_test.RUNNING_IN_COLAB:
        for action_category in action_categories:
            class_folder_training = STANFORD_TRAINING / action_category
            if not class_folder_training.exists():
                class_folder_training.mkdir()

            class_folder_testing = STANFORD_TESTING / action_category
            if not class_folder_testing.exists():
                class_folder_testing.mkdir()

    stanford_original = STANFORD / "JPEGImages"

    def move_sets(ori_folder, tgt_folder, filenames, labels):
        for i in range(len(filenames)):
            filename = filenames[i]
            label = labels[i]

            old_folder = ori_folder / filename
            new_folder = tgt_folder / label / filename
            print(f'"{old_folder}" -> "{new_folder}"')
            old_folder.replace(new_folder)

    move_sets(stanford_original, STANFORD_TRAINING, train_files, train_labels)
    move_sets(stanford_original, STANFORD_TESTING, test_files, test_labels)


def loadStanfordDatasets():
    """Example of use
    training, validation, testing = loadStanfordDatasets()
    print(training)
    from tensorflow.data.experimental import cardinality
    print(cardinality(training).numpy())
    print(training.class_names)
    """
    rand_seed = int(datetime.now().timestamp())

    training = image_dataset_from_directory(
        STANFORD_TRAINING,
        labels="inferred",
        label_mode="int",
        class_names=None,
        color_mode="rgb",
        batch_size=BATCH_SIZE,
        image_size=IMAGE_SIZE,
        shuffle=True,
        seed=rand_seed,
        validation_split=VAL_SPLIT,
        subset="training",
        interpolation="bilinear",
        follow_links=False,
    )

    validation = image_dataset_from_directory(
        STANFORD_TRAINING,
        labels="inferred",
        label_mode="int",
        class_names=None,
        color_mode="rgb",
        batch_size=BATCH_SIZE,
        image_size=IMAGE_SIZE,
        shuffle=True,
        seed=rand_seed,
        validation_split=VAL_SPLIT,
        subset="validation",
        interpolation="bilinear",
        follow_links=False,
    )

    testing = image_dataset_from_directory(
        STANFORD_TESTING,
        labels="inferred",
        label_mode="int",
        class_names=None,
        color_mode="rgb",
        batch_size=BATCH_SIZE,
        image_size=IMAGE_SIZE,
        shuffle=True,
        seed=None,
        validation_split=None,
        subset=None,
        interpolation="bilinear",
        follow_links=False,
    )

    return training, validation, testing


def colorPreprocessingLayer():
    preprocessing_layer = Sequential(
        [Rescaling(1.0 / 255, input_shape=IMAGE_SHAPE), RandomContrast(0.1)],
        name="Stanford_Preprocessing",
    )
    return preprocessing_layer


if __name__ == "__main__":

    preprocessStanfordData()
