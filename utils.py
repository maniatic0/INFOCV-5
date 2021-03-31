import tensorflow as tf
from tensorflow.python.platform import tf_logging as logging
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix

from colab_test import RUNNING_IN_COLAB

from keras.callbacks import Callback

import math


def checkTensorflow():
    print(f"Running in Collab: {RUNNING_IN_COLAB}")
    print(f"Tensor Flow Version: {tf.__version__}\n")
    print("Devices: ", tf.config.list_physical_devices())
    print("Num GPUs Available: ", len(tf.config.list_physical_devices("GPU")))

    if RUNNING_IN_COLAB:
        import warnings as wn

        device_name = tf.test.gpu_device_name()
        if device_name != "/device:GPU:0":
            wn.warn("GPU device not found", ResourceWarning)
            print(
                "\n\nThis error most likely means that this notebook is not "
                "configured to use a GPU.  Change this in Notebook Settings via the "
                "command palette (cmd/ctrl-shift-P) or the Edit menu.\n\n"
            )


# Check Tensorflow Configuration
checkTensorflow()


def createIfNecessaryDir(path):
    """Creates, if needed, a directory"""
    if not RUNNING_IN_COLAB and not path.exists():
        path.mkdir()

def plotTrainingHistory(folder, title, filename, history, bestEpoch):
    """Plot training history"""
    fig = plt.figure()
    plt.axes()
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Value")

    plt.ylim(0.0, 2.0)

    # Calculate epoch info
    epoch_number = len(next(iter(history.values())))
    plt.xticks(range(1, epoch_number + 1))

    # Plot everything
    plt.plot(range(1, epoch_number + 1), history["loss"], label="Loss", color="r")
    plt.plot(
        range(1, epoch_number + 1), history["accuracy"], label="Accuracy", color="gold"
    )
    plt.plot(
        range(1, epoch_number + 1),
        history["val_loss"],
        label="Validation Loss",
        color="b",
    )
    plt.plot(
        range(1, epoch_number + 1),
        history["val_accuracy"],
        label="Validation  Accuracy",
        color="magenta",
    )
    plt.axvline(x=bestEpoch + 1, label="Best Epoch", color="lime")

    # Draw legend
    plt.legend(loc="lower left")

    fig.tight_layout()
    if RUNNING_IN_COLAB:
        # On Google Colab is better to show the image
        plt.show()
    else:
        fig.savefig(folder / f"{filename}.png", dpi=fig.dpi)


def plotConfusionMatrix(folder, title, filename, categories, y_pred, y_real):
    """Confusion Matrix Plotting"""
    # Get the confusion matrix
    cf_matrix = confusion_matrix(y_real, y_pred)
    cf_normalized = cf_matrix / np.sum(cf_matrix)

    no_classes = len(categories)

    cf_text = [["" for _ in range(no_classes)] for _ in range(no_classes)]
    xticklabels = ["" for _ in range(no_classes)]
    yticklabels = ["" for _ in range(no_classes)]
    total_correct = 0
    total_incorrect = 0
    total = 0
    for i in range(no_classes):
        # Get Row Info
        total_row = np.sum(cf_matrix[i, :])
        correct_row = cf_matrix[i, i] / total_row
        incorrect_row = (total_row - cf_matrix[i, i]) / total_row
        xticklabels[
            i
        ] = f"{categories[i]}\nT: {total_row}\nC: {correct_row:.2%}\nW: {incorrect_row:.2%}"

        # Add to total
        total += total_row
        total_correct += cf_matrix[i, i]
        total_incorrect += total_row - cf_matrix[i, i]

        # Get Col Info
        total_col = np.sum(cf_matrix[:, i])
        correct_col = cf_matrix[i, i] / total_col
        incorrect_col = (total_col - cf_matrix[i, i]) / total_col
        yticklabels[
            i
        ] = f"{categories[i]}\nT: {total_row}\nC: {correct_col:.2%}\nW: {incorrect_col:.2%}"

        for j in range(no_classes):
            cf_text[i][j] = f"T: {cf_matrix[i,j]}\nP: {cf_normalized[i,j]:.2%}"

    # Plot Confusion matrix
    fig = plt.figure()
    plt.axes()
    plt.title(
        f"{title} T: {total} C: {total_correct/total:.2%} W: {total_incorrect/total:.2%}"
    )

    cf_text = np.asarray(cf_text)
    sns.heatmap(
        cf_matrix,
        annot=cf_text,
        fmt="",
        cmap="Blues",
        xticklabels=xticklabels,
        yticklabels=yticklabels,
    )
    plt.yticks(rotation=0)  # Fix rotation
    plt.xticks(rotation=0)  # Fix rotation
    plt.ylabel("Prediction Class")
    plt.xlabel("Real Class")

    fig.set_size_inches(19.20, 10.80)  # 1920x1080
    fig.tight_layout()

    if RUNNING_IN_COLAB:
        # On Google Colab is better to show the image
        plt.show()
    else:
        fig.savefig(folder / f"{filename}.png", dpi=100)


def saveModelsSummary(folder, models):
    """Save models info"""
    for (name, model) in models:
        filename = None
        if RUNNING_IN_COLAB:
            # On Google Colab is better to show the image
            filename = f"{name.lower()}"
            model.summary()
        else:
            filename = folder / f"{name.lower()}"
            with open(f"{filename}.txt", "w") as f:
                model.summary(print_fn=lambda x: f.write(x + "\n"))
                f.close()

        # Model Summary Image
        tf.keras.utils.plot_model(
            model,
            to_file=f"{filename}.png",
            show_shapes=True,
            show_layer_names=True,
            rankdir="TB",
            expand_nested=False,
            dpi=96,
        )

        # Save Model Architecture
        model.save(filename)


class BestEpochCallback(Callback):
    def __init__(self, weights_filename, *args, **kwargs):
        super(BestEpochCallback, self).__init__(*args, **kwargs)
        self.weights_filename = weights_filename
        self.monitor = "val_loss"
        self.reset()

    def reset(self):
        self.bestEpoch = -1
        self.bestValue = math.inf

    def get_monitor_value(self, logs):
        # From https://github.com/tensorflow/tensorflow/blob/v2.4.1/tensorflow/python/keras/callbacks.py#L1660-L1791
        logs = logs or {}
        monitor_value = logs.get(self.monitor)
        if monitor_value is None:
            logging.warning(
                "Best Epoch conditioned on metric `%s` "
                "which is not available. Available metrics are: %s",
                self.monitor,
                ",".join(list(logs.keys())),
            )
        return monitor_value

    def on_epoch_end(self, epoch, logs=None):
        value = self.get_monitor_value(logs)
        if value < self.bestValue:
            self.bestEpoch = epoch
            self.bestValue = value
            self.bestWeights = self.model.save_weights(
                self.weights_filename, overwrite=True
            )
