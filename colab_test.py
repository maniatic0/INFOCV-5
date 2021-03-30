RUNNING_IN_COLAB = False

try:
    import google.colab  # pylint: disable=import-error

    RUNNING_IN_COLAB = True
except:
    RUNNING_IN_COLAB = False