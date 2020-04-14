import keras
from keras.datasets import fashion_mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

from sklearn.metrics import confusion_matrix
import scipy.misc
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from metrics import error_rate
from prepare_data import prepare_data
import parameters


epochs = parameters.epochs[0]
batch_size = parameters.batch_size
num_classes = parameters.num_classes

img_rows, img_cols = parameters.img_rows, parameters.img_cols


input_shape, x_train, y_train, x_val, y_val, x_test, y_test, x_test_image, y_test_numeric = prepare_data(num_classes, img_rows, img_cols, fashion_mnist)

model = Sequential()

model.add(Conv2D(16, kernel_size=(3,3), padding='valid', activation='relu', input_shape=input_shape))
model.add(Conv2D(32, kernel_size=(5,5), padding='valid', activation='relu'))
model.add(Conv2D(128, kernel_size=(5,5), padding='valid', activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=[error_rate])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_val, y_val))

score = model.evaluate(x_test, y_test, verbose=0)

y_pred_probabilities = model.predict(x_test)
y_pred_classes = np.argmax(y_pred_probabilities, axis=1)

# confusion matrix: rows = actual values; columns = predicted values
confusion_matrix = confusion_matrix(y_test_numeric, y_pred_classes)

likelihood_matrix = []

for i in range(num_classes):
    likelihood_row = []

    for j in range(num_classes):
        predicted_i_likely_j = x_test_image[np.argmax(np.array([y_pred_probabilities[:, j][k] if y_test_numeric[k] == i else 0 for k in range(len(y_pred_probabilities[:, j]))]))]
        likelihood_row.append(predicted_i_likely_j)

        scipy.misc.imsave('./results/{}/{}.jpg'.format(i, j), predicted_i_likely_j)

    likelihood_matrix.append(likelihood_row)

likelihood_matrix = np.array(likelihood_matrix)

print('Test loss:', score[0])
print('Test error rate:', score[1])


fig, ax = plt.subplots(num_classes, num_classes)

filenames = ['./results/{}/{}.jpg'.format(i, j) for i in range(num_classes) for j in range(num_classes)]

for i in range(num_classes * num_classes):
    with open(filenames[i], 'rb') as f:
        image = Image.open(f)
        ax[i % num_classes][i // num_classes].set_yticklabels([])
        ax[i % num_classes][i // num_classes].set_xticklabels([])
        ax[i % num_classes][i // num_classes].imshow(image)

fig.savefig('./results/likelihood_matrix.jpg')
