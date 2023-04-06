#Загрузка mnist
import random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from tensorflow.python.keras import utils
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.models import Sequential
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Conv2D, MaxPooling2D, Flatten


from tensorflow.keras.datasets import mnist #Загружаем базу mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()







#Вывод изображений
fig, axes = plt.subplots(1, 10, figsize = (25, 3)) # Полотно из 10 картинок
for i in range(10): # Проход по всем классам
    label_index = np.where(y_train == i)[0] # Получение списка из индексов положений класса в i в y_train
    index = random.choice(label_index) # Слаучайным обрахом выбираем из списка индекс
    img = X_train[index] # Выбриаем из X_train нужное изобрадение
    axes[i].imshow(Image.fromarray(img), cmap='gray') # Отображаем изображение i-ым графиком
plt.show() # Покахать полотно





# Преобразование (y_train, y_test) one hot encoding
y_train = utils.to_categorical(y_train, 10)
y_test = utils.to_categorical(y_test, 10)

# Изменение формата mnist
# Надо добавить в конец размерности 1
# Для того что бы ограничить выбор цвета у сети
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

# Теперь можно посмотреть на формат изммененных данных
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)





# Размер батча (через сколько нужно сменить параметры модели)
batch_size = 128

# Модель
model = Sequential()
# Слой пакетной нормализации
model.add(BatchNormalization(input_shape=(28, 28, 1)))
# Первый сверточный слой
model.add(Conv2D(32, 3, padding='same', activation='relu'))
# Второй сверточный слой
model.add(Conv2D(32, 3, padding='same', activation='relu'))
# Первый слой подвыборки (размер квадрата)
model.add(MaxPooling2D(pool_size = (2, 2)))
# Слой регуляризации Dropout
model.add(Dropout(0.25))
# Выпрямляет матрицу в одномерый массив
model.add(Flatten())
# Полносвязный слой для классификации
model.add(Dense(256, activation='relu'))
# Слой регуляризации Dropout
model.add(Dropout(0.25))
# Выходной полносвязный слой 
model.add(Dense(10, activation = 'softmax'))


# Компиляция сети
model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(X_train,
                    y_train,
                    batch_size = batch_size,
                    epochs = 15,
                    validation_data = (X_test, y_test),
                    verbose = 1
                   )

# Отображение графиков
plt.plot(history.history['accuracy'], label = 'Доля верных ответов на обучающем наборе')
plt.plot(history.history['val_accuracy'], label = 'Доля верных ответов на проверочном наборе')
plt.xlabel('Эпоха обучение')
plt.ylabel('Доля верных ответов')
plt.show()


# Намер примера
n = 2020
# получение ответа на пример
prediction = model.predict(X_test)
# Вывод результата на экран
print("Выход сети: ", prediction[n])
print("Распознонная цифра: ", np.argmax(prediction[n]))
print("Верный ответ: ", np.argmax(y_test[n]))




# Предсказание на реальных цифрах
path = 'digits/'

# Вывод картинки для примера по каждому классу
def change_contrast(img, factor): # Функция для увеличения контрастности
  def contrast(pixel): # Функция для изменения цвета пикселя
    # Изменяем цвет каждого пикселя следующим образом
    # Если цвет пикселя в численой мере меньше 128, то значение будет уменьшатся
    # на меру factor * (pixel - 128). Иначе - увеличиваться на эту меру
    # Очевидно, что чем сильнее цвет отличен от 128, тем сильнее он изменится
    return 128 + factor * (pixel - 128)
  return img.point(contrast)

xTestReal = [] # Создаем xTestReal для загруженых картинок
yTestReal = [] # Создаем yTestR3eal для классовых изображений

for i in range(10): # Проходимся по классам от 0 до 9
  img_path = path + str(i) + '.png' # Определяем изображение
  # Загружаем изображение изменяя его размер на размер входного массива нейросети
  # Другими словами, подгоняем изображение к размеру картинок, на которых обучалась сеть 
  # Указываем grayscale для того чтобы цвет пикселя задавался одним числом
  img = image.load_img(img_path, grayscale = True, target_size=(28, 28))
  img1 = change_contrast(img, factor = 0.5) # Увеличиваем контраст изображения
  xTestReal.append(255 - np.asarray(img1)) # Инвертируем изобрадение и добалвяем в выборку
  yTestReal.append(i) # Добавление в y_train номер класса

xTestReal = np.array(xTestReal) # преобразование в numpy array
yTestReal = np.array(yTestReal) # преобразование в numpy array





fig, axes = plt.subplots(1, 10, figsize=(25, 3))
for i in range(10):
  axes[i].imshow(Image.fromarray(xTestReal[i]), cmap='gray')
plt.show()





xTestReal = xTestReal.reshape(xTestReal.shape[0], 28, 28, 1)
xTestReal.shape





prediction = model.predict(xTestReal)
for i in range(10):
  print("Распознаный образ: ", np.argmax(prediction[i]), "Верный ответ: ", yTestReal[i])