from keras.metrics import categorical_accuracy


def error_rate (y_true, y_pred):
    return 1 - categorical_accuracy(y_true, y_pred)