from utils import *
from prepare_datasets import *
from models import *
from colab_test import *

from pathlib import Path
import tempfile
import csv

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


def trainAndTestModel(name, model, training, validation, test):
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
            f'Model "{name}" with Testing Loss {test_loss:.4f}, Testing Accuracy {test_acc:.4f}, Validation Loss {val_loss:.4f} and Validation Accuracy {val_acc:.4f}'
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

    training, validation, testing = loadStanfordDatasets()
    name_stanford, model_stanford = stanfordModel()

    saveModelSummary(MODELS_FOLDER, name_stanford, model_stanford)
    model_stanford, stanford_results = trainAndTestModel(
        name_stanford, model_stanford, training, validation, testing
    )

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
        writer.writerow(stanford_results)


if __name__ == "__main__":
    main()
