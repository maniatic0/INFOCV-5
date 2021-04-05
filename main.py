from utils import (
    BestEpochCallback,
    createIfNecessaryDir,
    plotTrainingHistory,
    plotConfusionMatrix,
    getDirSize,
    saveModelSummary,
)
from prepare_stanford import loadStanfordDatasets
from optical_flow import loadTVHIRGB, loadFlowTVHI
from models import stanfordModel, transferModel, opticalFlowModel, twoStreamsModel
from colab_test import RUNNING_IN_COLAB

from pathlib import Path
import tempfile
import csv
import math

import numpy as np
import tensorflow as tf
from keras.callbacks import EarlyStopping

ROOT = Path(".") / "results"
createIfNecessaryDir(ROOT)

MODELS_FOLDER = ROOT / "models"
createIfNecessaryDir(MODELS_FOLDER)

TRAINING_FOLDER = ROOT / "training"
createIfNecessaryDir(TRAINING_FOLDER)

HISTORY_FOLDER = TRAINING_FOLDER / "history"
createIfNecessaryDir(HISTORY_FOLDER)

TESTING_FOLDER = ROOT / "testing"
createIfNecessaryDir(TESTING_FOLDER)


def trainAndTestModel(
    name, model, training, validation, test, confussion_res_pixel=(1920, 1080)
):
    filename = name.lower()

    # Training parameters
    no_epochs = 2
    batch_size = 64
    verbosity = 1

    # Early stopping to avoid overfitting
    patience = int(math.ceil(0.3 * no_epochs))
    monitor = "val_loss"
    mode = "min"

    # Callbacks for training
    callbacks = []
    early_stopping = EarlyStopping(
        monitor=monitor, mode=mode, verbose=verbosity, patience=patience
    )
    callbacks.append(early_stopping)

    val_loss = -1
    val_acc = -1
    test_loss = -1
    test_acc = -1

    with tempfile.TemporaryDirectory() as tmpdirname:
        # Saving the best model
        best_epoch = BestEpochCallback(tmpdirname)
        callbacks.append(best_epoch)

        print(f'Training: "{name}"')
        history = model.fit(
            training,
            batch_size=batch_size,
            epochs=no_epochs,
            verbose=verbosity,
            validation_data=validation,
            callbacks=callbacks,
        )

        # Get best weights
        model.load_weights(tmpdirname)

        # Save Model
        model.save(MODELS_FOLDER / filename)

        # Save Weights
        weights_folder = TESTING_FOLDER / filename
        createIfNecessaryDir(weights_folder)
        model.save_weights(weights_folder / "weights")

        plotTrainingHistory(
            HISTORY_FOLDER,
            f'Model "{name}" Training. Best Epoch {best_epoch.bestEpoch+1}.',
            filename,
            history.history,
            best_epoch.bestEpoch,
        )

        # Get and Print Results
        val_loss = history.history["val_loss"][best_epoch.bestEpoch]
        val_acc = history.history["val_accuracy"][best_epoch.bestEpoch]

        print(f'Testing: "{name}"')
        test_loss, test_acc = model.evaluate(
            test, batch_size=batch_size, verbose=verbosity
        )

        print(
            f'Model "{name}" with Testing Loss {test_loss:.4f}, Testing Accuracy {test_acc:.4f}, Validation Loss {val_loss:.4f} and Validation Accuracy {val_acc:.4f}\n'
        )

    y_pred = model.predict(test)
    y_pred = tf.argmax(y_pred, axis=1)
    y_pred = y_pred.numpy()

    y_test = np.concatenate([y for _, y in test], axis=0)

    # Plot Confussion Matrix
    plotConfusionMatrix(
        TESTING_FOLDER,
        f'Model "{name}" Training.',
        f"{filename}-confusion",
        test.class_names,
        y_pred,
        y_test,
        confussion_res_pixel,
    )

    size = getDirSize(MODELS_FOLDER / filename)

    results = {
        "model_name": name,
        "val_loss": val_loss,
        "val_acc": val_acc,
        "test_loss": test_loss,
        "test_acc": test_acc,
        "size": size,
        "size_gb": size / 1000 ** 3,
    }

    return model, results


def main():

    # Load Datasets
    training_stanford, validation_stanford, testing_stanford = loadStanfordDatasets()
    training_tvhi_rgb, validation_tvhi_rgb, testing_tvhi_rgb = loadTVHIRGB()
    training_tvhi_flow, validation_tvhi_flow, testing_tvhi_flow = loadFlowTVHI()

    # Load Stanford Model
    name_stanford, model_stanford = stanfordModel()

    # Save Stanford Model Summary
    saveModelSummary(MODELS_FOLDER, name_stanford, model_stanford)

    # Train Stanford Model
    model_stanford, results_stanford = trainAndTestModel(
        name_stanford,
        model_stanford,
        training_stanford,
        validation_stanford,
        testing_stanford,
        (10000, 10000),
    )

    # Load Transfer Model
    name_transfer, model_transfer = transferModel(model_stanford)

    # Save Transfer Model Summary
    saveModelSummary(MODELS_FOLDER, name_transfer, model_transfer)

    # Train Transfer Model
    model_transfer, results_transfer = trainAndTestModel(
        name_transfer,
        model_transfer,
        training_tvhi_rgb,
        validation_tvhi_rgb,
        testing_tvhi_rgb,
    )

    # Load Second Model
    name_transfer, model_transfer = transferModel(model_stanford)

    # Save Second Model Summary
    saveModelSummary(MODELS_FOLDER, name_transfer, model_transfer)

    # Train Second Model
    model_transfer, results_transfer = trainAndTestModel(
        name_transfer,
        model_transfer,
        training_tvhi_rgb,
        validation_tvhi_rgb,
        testing_tvhi_rgb,
    )

    # Load Flow Model
    name_flow, model_flow = opticalFlowModel()

    # Save Flow Model Summary
    saveModelSummary(MODELS_FOLDER, name_flow, model_flow)

    # Train Flow Model
    model_flow, results_flow = trainAndTestModel(
        name_flow,
        model_flow,
        training_tvhi_flow,
        validation_tvhi_flow,
        testing_tvhi_flow,
    )

    # Save Results
    with open(TESTING_FOLDER / "models_values.csv", "w") as f:
        fieldnames = [
            "model_name",
            "test_loss",
            "test_acc",
            "val_loss",
            "val_acc",
            "size",
            "size_gb",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        writer.writeheader()
        writer.writerow(results_stanford)
        writer.writerow(results_transfer)
        writer.writerow(results_flow)


if __name__ == "__main__":
    main()
