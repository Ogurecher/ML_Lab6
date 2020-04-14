import keras
from keras import backend as K


def cut_data_to_size (data, train_size=1000, val_size=200):
    return data[:train_size], data[train_size:train_size + val_size]


def prepare_data (num_classes, img_rows, img_cols, dataset):
    (x_train, y_train), (x_test_image, y_test_numeric) = dataset.load_data()
    (x_train, x_val), (y_train, y_val) = cut_data_to_size(x_train), cut_data_to_size(y_train)

    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_val = x_val.reshape(x_val.shape[0], 1, img_rows, img_cols)
        x_test = x_test_image.reshape(x_test_image.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_val = x_val.reshape(x_val.shape[0], img_rows, img_cols, 1)
        x_test = x_test_image.reshape(x_test_image.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    x_train = x_train.astype('float32')
    x_val = x_val.astype('float32')
    x_test = x_test.astype('float32')

    x_train /= 255
    x_val /= 255
    x_test /= 255

    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_val.shape[0], 'validation samples')
    print(x_test.shape[0], 'test samples')

    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_val = keras.utils.to_categorical(y_val, num_classes)
    y_test = keras.utils.to_categorical(y_test_numeric, num_classes)

    return input_shape, x_train, y_train, x_val, y_val, x_test, y_test, x_test_image, y_test_numeric