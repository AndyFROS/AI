import os
import random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import BatchNormalization, Conv2D, MaxPooling2D, Dropout, Flatten, Dense





train_path = 'data/train/'
test_path = 'data/test/'

# Вывод данных тестового набора
fig, axes = plt.subplots(1, 2, figsize=(25, 5)) # Создание пустого полотна 1 на 2
for i in range(2): # ПРи помощи цикла прохожусь по папкам 
  img_path = test_path + os.listdir(test_path)[i] # Полный путь к папке
  img = img_path+'/'+random.choice(os.listdir(img_path)) # Путь к рандомной картинке класса
  axes[i].imshow(image.load_img(img, target_size = (img_height, img_width))) # Вывод изображения на полотно
plt.show() # Показать полотно





batch_size =  16 #Размер выборки
img_width = 256  #Ширина изображения
img_height = 256 #Высота изображения

# Создание генератора
data_generator = ImageDataGenerator(
    rescale = 1/ 255, # Значения цвета меняем на дробные показания
    rotation_range = 10, # Поворот изображения при генерации выборки
    height_shift_range = 0.1, # Двигаем по высоте изображение
    width_shift_range = 0.1, # Двигаем по ширине изобажение
    zoom_range = 0.1, # Зумировнаие изображения при генерации выборки
    horizontal_flip = True, # Отзеркаливание изображений
    fill_mode = 'nearest', # Заполнение пикселей вне границ ввода
    validation_split = 0.15 # Указываем разделение изображений на обучающую и тестовую выборку
)

# Данные, созданные генератором
train_generator = data_generator.flow_from_directory(
    train_path, # Пусть ко всей выборке
    target_size = (img_height, img_width), # Размеры изображений
    batch_size = batch_size,
    shuffle = True, # Перемешка выборки
    class_mode = 'categorical', # Категориальный тип выборки. Рахбиение выборки по маркам авто
    subset = 'training' # Установка выборки как обучающая
)

validation_generator = data_generator.flow_from_directory(
    train_path, # Пусть ко всей выборке
    target_size = (img_height, img_width), # Размеры изображений
    batch_size = batch_size,
    shuffle = True, # Перемешка выборки
    class_mode = 'categorical', # Категориальный тип выборки. Рахбиение выборки по маркам авто
    subset = 'validation' # Установка выборки как валидационная
)





# Создание модели
model = Sequential()
# Первый слой пакетной нормализации
model.add(BatchNormalization(input_shape=(img_height, img_width, 3)))
# Первый сверточный слой
model.add(Conv2D(128, 3, padding='same', activation='relu'))
# Второй сверточный слой
model.add(Conv2D(256, 3, padding='same', activation='relu'))
# Первый слой подвыборки
model.add(MaxPooling2D(pool_size=(3, 3)))

#Второй слой пакетной нормализации
model.add(BatchNormalization())
#Третий сверточный слой
model.add(Conv2D(512, 3, padding='same', activation='relu'))
#Четвертый сверточный слой
model.add(Conv2D(1024, 3, padding='same', activation='relu'))
#Второй слой подвыборки
model.add(MaxPooling2D(pool_size = (3, 3)))
# Слой регуляризации
model.add(Dropout(0.2))

# Третий слой пакетной нормализации
model.add(BatchNormalization())
#Пятый сверточный слой
model.add(Conv2D(1024, 3, padding='same', activation='relu'))
#Шестой сверточный слой
model.add(Conv2D(512, 3, padding='same', activation='relu'))
#Третий слой подвыборки
model.add(MaxPooling2D(pool_size=(3, 3)))
# Слой регуляризации
model.add(Dropout(0.2))

# Четвертый слой пакетной нормализации
model.add(BatchNormalization())
#Седьмой сверточный слой
model.add(Conv2D(256, 3, padding='same', activation='relu'))
#Восьмой сверточный слой
model.add(Conv2D(128, 3, padding='same', activation='relu'))
#Четвертый слой подвыборки
model.add(MaxPooling2D(pool_size=(3, 3)))
# Слой регуляризации
model.add(Dropout(0.2))

# Слой преобразования данных в 2D проедставление в плоскости
model.add(Flatten())
# Первый полносвязный слой для классификации
model.add(Dense(256, activation='relu'))
# Второй полносвязный слой для классификации
model.add(Dense(128, activation='relu'))
#Выходной слой
model.add(Dense(len(train_generator.class_indices), activation='softmax'))

# Компиляция сети
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])





#Обучение сети
history = model.fit(
    train_generator, # Данные для обучения
    steps_per_epoch = train_generator.samples // batch_size, # Шаг генерации тренеровочных данных
    validation_data = validation_generator, # Валидационные данные
    validation_steps = validation_generator.samples // batch_size, # Шаг генерации валидационных данных
    epochs = 30, # Количество эпох для обучения
    verbose = 1 # Покзывать статистику по каждой эпохе
)





# График отражающий точность обучения обучения
plt.plot(history.history['accuracy'], label = 'Точность не тренировочном наборе') # Точность на тренеровочном наборе
plt.plot(history.history['val_accuracy'], label = 'Точность не тестовом наборе') # Точность на тестовом наборе
plt.xlabel('Количество эпох') # Наименование оси X
plt.ylabel('Точность модели') # Наименование оси y
plt.show() # Показать полотно




# Общие параметры нейронной сети
model.summary()





#Наглядная
classes = ['tomatoes', 'apples']
for n in range(len(classes)):
  for i in os.listdir(test_path+classes[n]):
    test_img_path = test_path+classes[n]+'/'+i
    
    with Image.open(test_img_path) as img:
      img.load()
    img = img.resize((img_height, img_width), 3)
    img = np.array(img)[None]
    plt.figure()
    plt.title(f'Изображение из класса {classes[n]}\nСеть отнесла к классу {classes[np.argmax(model.predict(img,verbose = 0))]}')
    img = img[0]
    plt.imshow(np.squeeze(img))


