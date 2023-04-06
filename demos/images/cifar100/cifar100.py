import random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from tensorflow.python.keras import utils
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.models import Sequential
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Conv2D, MaxPooling2D, Flatten

from tensorflow.keras.datasets import cifar100
(X_train100, y_train100), (X_test100, y_test100) = cifar100.load_data()



fig, axes = plt.subplots(1, 10, figsize=(25,3))
for i in range(10):
  label_index = np.where(X_train100 == i)[0]
  index = random.choice(label_index)
  img = X_train100[index]
  axes[i].imshow(Image.fromarray(img))
plt.show()



y_train100 = utils.to_categorical(y_train100, 100)
y_test100 = utils.to_categorical(y_test100, 100)

print(X_train100.shape)
print(X_test100.shape)
print(y_train100.shape)
print(y_test100.shape)






#задаём batch_size
batch_size = 512 

model = Sequential()
model.add(BatchNormalization(input_shape=(32, 32, 3)))
model.add(Conv2D(32,(3, 3), padding='same', activation='relu'))
model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))


model.add(BatchNormalization())
model.add(Conv2D(64, 3, padding='same', activation='relu'))
model.add(Conv2D(64, 3, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))


model.add(BatchNormalization())
model.add(Conv2D(256, 3, padding='same', activation='relu'))
model.add(Conv2D(256, 3, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))


model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(100, activation = 'softmax'))

model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(
    X_train100,
    y_train100,
    batch_size = batch_size,
    epochs = 40,
    validation_data = (X_test100, y_test100),
    verbose=1
)

plt.plot(history.history['accuracy'], label='Качество на обучающей выборке')
plt.plot(history.history['val_accuracy'], label='Качество на тестовой выборке')
plt.xlabel('Эпоха обучение')
plt.ylabel('Доля верных ответов')
plt.show()

model.summary()