import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D
from sklearn.model_selection import ParameterGrid

from metrics import error_rate
from prepare_data import prepare_data
import parameters


batch_size = parameters.batch_size
num_classes = parameters.num_classes

img_rows, img_cols = parameters.img_rows, parameters.img_cols


input_shape, x_train, y_train, x_val, y_val, x_test, y_test, _, _ = prepare_data(num_classes, img_rows, img_cols, mnist)

error_rates = []

param_grid = list(ParameterGrid({
    "epochs": parameters.epochs,
    **{'conv1_'+key: val for key, val in parameters.conv2d_params.items()},
    **{'conv2_'+key: val for key, val in parameters.conv2d_params.items()},
    **{'conv3_'+key: val for key, val in parameters.conv2d_params.items()},
    **parameters.pooling_params
}))

for params in param_grid:
    model = Sequential()

    model.add(Conv2D(params['conv1_filters'], kernel_size=params['conv1_kernel_size'], padding=params['conv1_padding'], activation=params['conv1_activation'], input_shape=input_shape))
    model.add(Conv2D(params['conv2_filters'], kernel_size=params['conv2_kernel_size'], padding=params['conv2_padding'], activation=params['conv2_activation']))
    model.add(Conv2D(params['conv3_filters'], kernel_size=params['conv3_kernel_size'], padding=params['conv3_padding'], activation=params['conv3_activation']))
    model.add(MaxPooling2D(pool_size=params['pool_size']))

    model.add(Flatten())
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=[error_rate])

    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=params['epochs'],
              verbose=1,
              validation_data=(x_val, y_val))

    score = model.evaluate(x_test, y_test, verbose=0)

    print('Test loss:', score[0])
    print('Test error rate:', score[1])

    error_rates.append({'error_rate': score[1], 'parameters': params})

error_rates = sorted(error_rates, key=lambda k: (k['error_rate'], k['parameters']['epochs']))
print(error_rates)
