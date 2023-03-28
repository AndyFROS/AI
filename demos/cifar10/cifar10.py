import random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from tensorflow.python.keras import utils
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.models import Sequential
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Conv2D, MaxPooling2D, Flatten

from tensorflow.keras.datasets import cifar10
(X_train10, y_train10), (X_test10, y_test10) = cifar10.load_data()


# Определение классов по порялдку
classes = ['самолет', 'автомобиль', 'птица', 'кот', 'олень', 'собака', 'лягушка', 'лошадь', 'корабль', 'грузовик']

# Вывод всех картинок, для примера
fig, axes = plt.subplots(1, 10, figsize=(25, 3)) # Создаем полотно из 10 графиков
for i in range(10):  # Проход по всем 10 классам
  label_indexes = np.where(y_train10==i)[0] # Получаем список из индексов положенийкласса i в y_train
  index = random.choice(label_indexes) # Случайным образом выбираем из списка индекс
  img = X_train10[index] # Выбираем изобрадение
  axes[i].imshow(Image.fromarray(img, 'RGB')) # Отображение картинки на полотне
plt.show() # Демонстрация полотна




y_train10 = utils.to_categorical(y_train10)
y_test10 = utils.to_categorical(y_test10)

print(X_train10.shape)
print(X_test10.shape)
print(y_train10.shape)
print(y_test10.shape)




batch_size = 128

# Создание последовательной модели
model = Sequential()
# Солой пакетной нормализации
model.add(BatchNormalization(input_shape=(32, 32, 3)))
# Первый сверточный слой
model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
# Второй сверточный слой
model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
# Первый слой подвыборки
model.add(MaxPooling2D(pool_size=(2, 2)))
# Регуляризация Droput
model.add(Dropout(0.25))


# Слой покаетной нормазизации
model.add(BatchNormalization())
# Третий сверточный слой
model.add(Conv2D(64, 3, padding='same', activation='relu'))
# Четвертый сверточный слой
model.add(Conv2D(64, 3, padding='same', activation='relu'))
# Второй слой подвыборки
model.add(MaxPooling2D(pool_size=(2, 2)))
# Слой регуляризации
model.add(Dropout(0.25))

# Слой пакетной нормализации
model.add(BatchNormalization())
# Пятный сверточный слой
model.add(Conv2D(128, 3, padding='same', activation='relu'))
# Шестой сверточный слой
model.add(Conv2D(128, 3, padding='same', activation='relu'))
# Третий слой подвыборки
model.add(MaxPooling2D(pool_size=(2, 2)))
# Слой регуляризации
model.add(Dropout(0.25))


# Слой преобразования данных в 2D проедставление в плоскости
model.add(Flatten())
# Полносвязный слой для классификации
model.add(Dense(512, activation='relu'))
# Слой регуляризации
model.add(Dropout(0.25))
# Выходной полносвязный слой
model.add(Dense(10, activation='softmax'))

# Компиляция сети
model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Обучение сети на данный cifar10
model.fit(
    X_train10,
    y_train10,
    batch_size=batch_size,
    epochs = 20,
    validation_data=(X_test10, y_test10),
    verbose=1
)





prediction = model.predict(X_test10)
n = 2052

plt.imshow(Image.fromarray(X_test10[n]).convert('RGBA'))
plt.show()

# Выводим на экран результаты
print("Вызод сети: ", prediction[n])
print("Распознаный образ: ", np.argmax(prediction[n]))
print("Верный ответ: ", y_test10[n])
print("Распознаный образ на картинке: ", classes[np.argmax(prediction[n])])