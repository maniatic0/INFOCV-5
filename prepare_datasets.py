from pathlib import Path

from tensorflow.keras.preprocessing import image_dataset_from_directory

import numpy as np

import colab_test

ROOT = Path(".") / "datasets"

STANFORD = ROOT / "Stanford40"

TRAINING = STANFORD / 'training'
if not colab_test.RUNNING_IN_COLAB and not TRAINING.exists():
    TRAINING.mkdir()

TESTING = STANFORD / 'testing'
if not colab_test.RUNNING_IN_COLAB and not TESTING.exists():
    TESTING.mkdir()

LABEL_NAMES = STANFORD / "labels.txt"

def loadLabels():
    with open(LABEL_NAMES, "r") as f:
        labels = f.readlines()
    
    for i in range(len(labels)):
        labels[i] = labels[i].strip()

    return labels

BATCH_SIZE = 64
IMAGE_SIZE = (128, 128)
IMAGE_SHAPE = (128, 128, 3)

def loadDatasets():
    training = image_dataset_from_directory(
        TRAINING, labels='inferred', label_mode='int',
        class_names=None, color_mode='rgb', batch_size=BATCH_SIZE, image_size=IMAGE_SIZE, 
        shuffle=True, seed=None, validation_split=None, subset=None,
        interpolation='bilinear', follow_links=False
    )

    testing = image_dataset_from_directory(
        TESTING, labels='inferred', label_mode='int',
        class_names=None, color_mode='rgb', batch_size=BATCH_SIZE, image_size=IMAGE_SIZE, 
        shuffle=True, seed=None, validation_split=None, subset=None,
        interpolation='bilinear', follow_links=False
    )
    return training, testing



if __name__ == "__main__":
    

    with open(STANFORD / 'ImageSplits/train.txt', 'r') as f:
        train_files = list(map(str.strip, f.readlines()))
        train_labels = ['_'.join(name.split('_')[:-1]) for name in train_files]
        print(f'Train files ({len(train_files)}):\n\t{train_files}')
        print(f'Train labels ({len(train_labels)}):\n\t{train_labels}\n')

    with open(STANFORD / 'ImageSplits/test.txt', 'r') as f:
        test_files = list(map(str.strip, f.readlines()))
        test_labels = ['_'.join(name.split('_')[:-1]) for name in test_files]
        print(f'Test files ({len(test_files)}):\n\t{test_files}')
        print(f'Test labels ({len(test_labels)}):\n\t{test_labels}\n')
        
    action_categories = sorted(list(set(['_'.join(name.split('_')[:-1]) for name in train_files])))
    print(f'Action categories ({len(action_categories)}):\n{action_categories}')

    # Category folder creation
    if not colab_test.RUNNING_IN_COLAB:
        for action_category in action_categories:
            class_folder_training = TRAINING / action_category
            if not class_folder_training.exists():
                class_folder_training.mkdir()

            class_folder_testing = TESTING / action_category
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

    move_sets(stanford_original, TRAINING, train_files, train_labels)
    move_sets(stanford_original, TESTING, test_files, test_labels)

    with open(LABEL_NAMES, "w") as f:
        for category in action_categories:
            print(category, file=f)

    






    

