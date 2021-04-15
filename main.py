from utils import (
    BestEpochCallback,
    createIfNecessaryDir,
    plotTrainingHistory,
    plotConfusionMatrix,
    getDirSize,
    saveModelSummary,
)
from prepare_stanford import loadStanfordDatasets, BATCH_SIZE
from optical_flow import loadTVHIRGB, loadFlowTVHI, loadDualTVHI
from models import (
    stanfordModel,
    transferModel,
    opticalFlowModel,
    twoStreamsModel,
    hydraModel,
)
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


def saveModel(name, model):
    filename = name.lower()
    # Save Model
    model.save(MODELS_FOLDER / filename)


def loadModel(name):
    filename = name.lower()
    # Load Model
    return tf.keras.models.load_model(str(MODELS_FOLDER / filename))


def saveModelWeights(name, model):
    filename = name.lower()
    weights_folder = TESTING_FOLDER / filename
    createIfNecessaryDir(weights_folder)
    model.save_weights(weights_folder / "weights")


def loadModelWeights(name, model):
    filename = name.lower()
    weights_folder = TESTING_FOLDER / filename / "weights"
    model.load_weights(str(weights_folder))
    return model


def trainAndTestModel(
    name,
    model,
    training,
    validation,
    test,
    confussion_res_pixel=(1920, 1080),
    history_res_pixel=(4096, 1080),
    history_max_y=10,
):
    filename = name.lower()

    # Training parameters
    no_epochs = 500
    batch_size = BATCH_SIZE
    verbosity = 1

    # Early stopping to avoid overfitting
    patience = int(math.ceil(0.1 * no_epochs))
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
        saveModel(name, model)

        # Save Weights
        saveModelWeights(name, model)

        plotTrainingHistory(
            HISTORY_FOLDER,
            f'Model "{name}" Training. Best Epoch {best_epoch.bestEpoch+1}.',
            filename,
            history.history,
            best_epoch.bestEpoch,
            res_pixel=history_res_pixel,
            max_y=history_max_y,
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

    load_stanford = True
    load_transfer = True
    load_flow = True
    load_dual = True
    load_hydra = False
    loading_options = [load_stanford, load_transfer, load_flow, load_dual, load_hydra]

    # Load Datasets
    training_stanford, validation_stanford, testing_stanford = loadStanfordDatasets()
    training_tvhi_rgb, validation_tvhi_rgb, testing_tvhi_rgb = loadTVHIRGB()
    training_tvhi_flow, validation_tvhi_flow, testing_tvhi_flow = loadFlowTVHI()
    training_tvhi_dual, validation_tvhi_dual, testing_tvhi_dual = loadDualTVHI()

    # Generate Stanford Model
    name_stanford, model_stanford = stanfordModel()

    if not load_stanford:
        # Train from scratch

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
    else:
        # Load from previous
        model_stanford = loadModelWeights(name_stanford, model_stanford)

    # Generate Transfer Model
    name_transfer, model_transfer = transferModel(model_stanford)

    if not load_transfer:
        # Train From Scratch

        # Save Transfer Model Summary
        saveModelSummary(MODELS_FOLDER, name_transfer, model_transfer)

        # Train Transfer Model
        model_transfer, results_transfer = trainAndTestModel(
            name_transfer,
            model_transfer,
            training_tvhi_rgb,
            validation_tvhi_rgb,
            testing_tvhi_rgb,
            history_max_y=300,
        )
    else:
        # Load previous model
        model_transfer = loadModelWeights(name_transfer, model_transfer)

    # Generate Flow Model
    name_flow, model_flow = opticalFlowModel()

    if not load_flow:
        # Train From Scratch

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
    else:
        # Load previous model
        model_flow = loadModelWeights(name_flow, model_flow)

    # Generate Two Stream Model
    name_dual, model_dual = twoStreamsModel(model_transfer, model_flow)

    if not load_dual:
        # Train From Scratch

        # Save Two Stream Model Summary
        saveModelSummary(MODELS_FOLDER, name_dual, model_dual)

        # Train Two Stream Model
        model_dual, results_dual = trainAndTestModel(
            name_dual,
            model_dual,
            training_tvhi_dual,
            validation_tvhi_dual,
            testing_tvhi_dual,
        )
    else:
        # Load previous model
        model_dual = loadModelWeights(name_dual, model_dual)

    # Generate Hydra Models
    (name_hydra_stanford, model_hydra_stanford), (
        name_hydra,
        model_hydra,
    ) = hydraModel()

    if not load_hydra:
        # Train From Scratch

        # Save Hydra Stanford Head Model Summary
        saveModelSummary(MODELS_FOLDER, name_hydra_stanford, model_hydra_stanford)

        # Train Hydra Stanford Head
        model_hydra_stanford, results_hydra_stanford = trainAndTestModel(
            name_hydra_stanford,
            model_hydra_stanford,
            training_stanford,
            validation_stanford,
            testing_stanford,
            (10000, 10000),
        )

        # Save Hydra Model Summary
        saveModelSummary(MODELS_FOLDER, name_hydra, model_hydra)

        # Train Hydra Model
        model_hydra, results_hydra = trainAndTestModel(
            name_hydra,
            model_hydra,
            training_tvhi_dual,
            validation_tvhi_dual,
            testing_tvhi_dual,
        )

    else:
        # Load previous model
        model_hydra = loadModelWeights(name_hydra, model_hydra)

    # Save Results
    trained_models = "".join(["0" if loaded else "1" for loaded in loading_options])
    with open(TESTING_FOLDER / f"models_values_{trained_models}.csv", "w") as f:
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
        if not load_stanford:
            writer.writerow(results_stanford)
        if not load_transfer:
            writer.writerow(results_transfer)
        if not load_flow:
            writer.writerow(results_flow)
        if not load_dual:
            writer.writerow(results_dual)
        if not load_hydra:
            writer.writerow(results_hydra_stanford)
            writer.writerow(results_hydra)


if __name__ == "__main__":
    main()
