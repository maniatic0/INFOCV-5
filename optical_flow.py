from pathlib import Path
import os, glob
import math

import colab_test
from utils import createIfNecessaryDir
from prepare_stanford import IMAGE_SIZE, IMAGE_SHAPE, BATCH_SIZE, VAL_SPLIT

import numpy as np
import cv2
import tensorflow as tf

ROOT = Path(".") / "datasets"

TVHI = ROOT / "TV-HI"

TVHI_VIDEOS = TVHI / "tv_human_interactions_videos"

TVHI_TRAINING = TVHI / "training"
createIfNecessaryDir(TVHI_TRAINING)

TVHI_TESTING = TVHI / "testing"
createIfNecessaryDir(TVHI_TESTING)

TVHI_CLASSES = ["handShake", "highFive", "hug", "kiss"]  # we ignore the negative class
TVHI_NO_CLASSES = len(TVHI_CLASSES)
TVHI_STACK_SIZE = 8
TVHI_FLOW_SHAPE = (TVHI_STACK_SIZE, *IMAGE_SHAPE)


def loadRGBDataset(folder_pattern):
    # based on https://www.tensorflow.org/tutorials/load_data/images
    file_glob = glob.glob(str(folder_pattern), recursive=True)
    list_ds = tf.data.Dataset.from_tensor_slices(file_glob)
    list_ds = list_ds.shuffle(len(file_glob), reshuffle_each_iteration=False)

    class_names = np.array(TVHI_CLASSES)

    def get_label(file_path):
        # convert the path to a list of path components
        parts = tf.strings.split(file_path, os.path.sep)
        # The second to last is the class-directory
        one_hot = parts[-3] == class_names
        # Integer encode the label
        return tf.argmax(one_hot)

    def decode_img(img):
        # convert the compressed string to a 3D uint8 tensor
        img = tf.image.decode_png(img, channels=3)
        # resize the image to the desired size
        return tf.image.resize(img, [IMAGE_SIZE[0], IMAGE_SIZE[1]])

    def process_path(file_path):
        label = get_label(file_path)
        # load the raw data from the file as a string
        img = tf.io.read_file(file_path)
        img = decode_img(img)
        return img, label

    list_ds = list_ds.map(
        process_path, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False
    )
    list_ds = list_ds.batch(BATCH_SIZE)

    list_ds.class_names = TVHI_CLASSES

    return list_ds


def loadTVHIRGB():
    testing = loadRGBDataset(TVHI_TESTING / "**" / "rgb_*")
    total_training = loadRGBDataset(TVHI_TRAINING / "**" / "rgb_*")

    val_size = int(tf.data.experimental.cardinality(total_training).numpy() * VAL_SPLIT)
    training = total_training.skip(val_size)
    validation = total_training.take(val_size)

    return training, validation, testing


def loadFlowDataset(folder_pattern, image_pattern):
    dir_glob = glob.glob(str(folder_pattern), recursive=True)
    final_stacks = []
    for folder in dir_glob:
        img_glob = glob.glob(str(Path(folder) / image_pattern), recursive=True)
        img_glob.sort()
        for i in range(TVHI_STACK_SIZE - 1, len(img_glob)):
            img_stack = [""] * TVHI_STACK_SIZE
            for j in range(TVHI_STACK_SIZE):
                img_stack[j] = img_glob[i - TVHI_STACK_SIZE + j + 1]
            final_stacks.append(img_stack)

    list_ds = tf.data.Dataset.from_tensor_slices(final_stacks)
    list_ds = list_ds.shuffle(len(final_stacks), reshuffle_each_iteration=False)

    class_names = np.array(TVHI_CLASSES)

    def get_label(file_path):
        # convert the path to a list of path components
        parts = tf.strings.split(file_path, os.path.sep)
        # The second to last is the class-directory
        one_hot = parts[-3] == class_names
        # Integer encode the label
        return tf.argmax(one_hot)

    def decode_img(img):
        # convert the compressed string to a 3D uint8 tensor
        img = tf.image.decode_png(img, channels=3)
        # resize the image to the desired size
        return tf.image.resize(img, [IMAGE_SIZE[0], IMAGE_SIZE[1]])

    def process_path(file_path):
        # load the raw data from the file as a string
        img = tf.io.read_file(file_path)
        img = decode_img(img)
        return img

    def process_stack(stack):
        label = get_label(tf.gather(stack, 0, axis=0))
        processed = tf.map_fn(
            process_path,
            stack,
            fn_output_signature=tf.TensorSpec(
                IMAGE_SHAPE, dtype=tf.dtypes.float32, name=None
            ),
        )
        return processed, label

    list_ds = list_ds.map(
        process_stack, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False
    )
    list_ds = list_ds.batch(BATCH_SIZE)

    list_ds.class_names = TVHI_CLASSES
    return list_ds


def loadFlowTVHI():
    testing = loadFlowDataset(TVHI_TESTING / "*" / "*", "flow_*.png")
    total_training = loadFlowDataset(TVHI_TRAINING / "*" / "*", "flow_*.png")

    val_size = int(tf.data.experimental.cardinality(total_training).numpy() * VAL_SPLIT)
    training = total_training.skip(val_size)
    validation = total_training.take(val_size)

    return training, validation, testing


def processSet(folder, videos, labels):
    def getDigits(size):
        return int(math.ceil(math.log(size, 10)))

    def paddInt(num, pad):
        return str(num).rjust(pad, "0")

    video_digits = getDigits(len(videos))

    label_count = {}
    for label in TVHI_CLASSES:
        label_count[label] = 0

    for i in range(len(videos)):
        video = videos[i]
        label = labels[i]

        vi_num = label_count[label]
        label_count[label] += 1

        video_folder = folder / label / paddInt(vi_num, video_digits)
        createIfNecessaryDir(video_folder)

        cap = cv2.VideoCapture(str(TVHI_VIDEOS / video))
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        frames_digits = getDigits(length)

        i = 0
        ret, frame = cap.read()
        if not ret:
            print(f"Can't receive frame for video {video}. Skipping ...")
            cap.release()
            continue

        prvs = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hsv = np.zeros_like(frame)
        hsv[..., 1] = 255
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        padded_index = paddInt(i, frames_digits)
        cv2.imwrite(
            str(video_folder / f"rgb_{padded_index}.png"),
            cv2.resize(frame, IMAGE_SIZE, interpolation=cv2.INTER_LANCZOS4),
        )
        cv2.imwrite(
            str(video_folder / f"flow_{padded_index}.png"),
            cv2.resize(rgb, IMAGE_SIZE, interpolation=cv2.INTER_LANCZOS4),
        )

        i += 1

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print(f"Finished video {video}...")
                break
            curr = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            flow = cv2.calcOpticalFlowFarneback(
                prvs, curr, None, 0.5, 3, 15, 3, 5, 1.2, 0
            )
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            hsv[..., 0] = ang * 180 / np.pi / 2
            hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
            rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

            padded_index = paddInt(i, frames_digits)
            cv2.imwrite(
                str(video_folder / f"rgb_{padded_index}.png"),
                cv2.resize(frame, IMAGE_SIZE, interpolation=cv2.INTER_LANCZOS4),
            )
            cv2.imwrite(
                str(video_folder / f"flow_{padded_index}.png"),
                cv2.resize(rgb, IMAGE_SIZE, interpolation=cv2.INTER_LANCZOS4),
            )

            prvs = curr
            i += 1

        cap.release()


def main():
    set_1_indices = [
        [
            2,
            14,
            15,
            16,
            18,
            19,
            20,
            21,
            24,
            25,
            26,
            27,
            28,
            32,
            40,
            41,
            42,
            43,
            44,
            45,
            46,
            47,
            48,
            49,
            50,
        ],
        [
            1,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            23,
            24,
            25,
            27,
            28,
            29,
            30,
            31,
            32,
            33,
            34,
            35,
            44,
            45,
            47,
            48,
        ],
        [
            2,
            3,
            4,
            11,
            12,
            15,
            16,
            17,
            18,
            20,
            21,
            27,
            29,
            30,
            31,
            32,
            33,
            34,
            35,
            36,
            42,
            44,
            46,
            49,
            50,
        ],
        [
            1,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            16,
            17,
            18,
            22,
            23,
            24,
            26,
            29,
            31,
            35,
            36,
            38,
            39,
            40,
            41,
            42,
        ],
    ]
    set_2_indices = [
        [
            1,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            17,
            22,
            23,
            29,
            30,
            31,
            33,
            34,
            35,
            36,
            37,
            38,
            39,
        ],
        [
            2,
            3,
            4,
            5,
            14,
            15,
            16,
            17,
            18,
            19,
            20,
            21,
            22,
            26,
            36,
            37,
            38,
            39,
            40,
            41,
            42,
            43,
            46,
            49,
            50,
        ],
        [
            1,
            5,
            6,
            7,
            8,
            9,
            10,
            13,
            14,
            19,
            22,
            23,
            24,
            25,
            26,
            28,
            37,
            38,
            39,
            40,
            41,
            43,
            45,
            47,
            48,
        ],
        [
            2,
            3,
            4,
            5,
            6,
            15,
            19,
            20,
            21,
            25,
            27,
            28,
            30,
            32,
            33,
            34,
            37,
            43,
            44,
            45,
            46,
            47,
            48,
            49,
            50,
        ],
    ]
    classes = TVHI_CLASSES

    # test set
    set_1 = [
        f"{classes[c]}_{i:04d}.avi"
        for c in range(len(classes))
        for i in set_1_indices[c]
    ]
    set_1_label = [
        f"{classes[c]}" for c in range(len(classes)) for i in set_1_indices[c]
    ]
    print(f"Set 1 to be used for test ({len(set_1)}):\n\t{set_1}")
    print(f"Set 1 labels ({len(set_1_label)}):\n\t{set_1_label}\n")

    # training set
    set_2 = [
        f"{classes[c]}_{i:04d}.avi"
        for c in range(len(classes))
        for i in set_2_indices[c]
    ]
    set_2_label = [
        f"{classes[c]}" for c in range(len(classes)) for i in set_2_indices[c]
    ]
    print(f"Set 2 to be used for train and validation ({len(set_2)}):\n\t{set_2}")
    print(f"Set 2 labels ({len(set_2_label)}):\n\t{set_2_label}")

    for label in classes:
        createIfNecessaryDir(TVHI_TRAINING / label)
        createIfNecessaryDir(TVHI_TESTING / label)

    processSet(TVHI_TRAINING, set_2, set_2_label)
    processSet(TVHI_TESTING, set_1, set_1_label)


if __name__ == "__main__":
    main()
