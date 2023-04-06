import os
import random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt



from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import BatchNormalization, Conv2D, MaxPooling2D, Dropout, Flatten, Dense


# Specify root paths
train_path = "/data/train/"
test_path  = "/data/test/"

batch_size = 10 #Размер выборки
img_width = 256  #Ширина изображения
img_height = 256 #Высота изображения


# Генератор изображений
datagen = ImageDataGenerator(
    rescale = 1. / 255, # Значения цвета меняем на дробные показания
    rotation_range = 10, # Поворот изображения при генерации выборки
    width_shift_range = 0.1, # Двигаем по ширине изобажение
    height_shift_range = 0.1, # Двигаем по высоте изображение
    zoom_range = 0.1, # Зумировнаие изображения при генерации выборки
    horizontal_flip = True, # Отзеркаливание изображений
    fill_mode = 'nearest', # Заполнение пикселей вне границ ввода
    validation_split = 0.1 # Указываем разделение изображений на обучающую и тестовую выборку
)


train_generator = datagen.flow_from_directory(
    train_path, # Пусть ко всей выборке
    target_size=(img_height, img_width), # Размеры изображений
    batch_size = batch_size,
    class_mode = 'categorical', # Категориальный тип выборки. Рахбиение выборки по маркам авто
    shuffle = True, # Перемешка выборки
    subset = 'training' # Установка выборки как обучающая
)

validation_generator = datagen.flow_from_directory(
    test_path, # Путь к выборке
    target_size=(img_height, img_width), # Размер изображений
    batch_size = batch_size,
    class_mode = 'categorical', # Категориальный тип выборки
    shuffle = True, # Перемешать выборку
    subset = 'validation' # Установка выборки как валидационная
)




fig, axes = plt.subplots(1, 2, figsize=(25, 5))
for i in range(2):
  car_path = train_path +'/'+ os.listdir(train_path)[i] + '/'
  img_path = car_path + random.choice(os.listdir(car_path))
  axes[i].imshow(image.load_img(img_path, target_size=(img_height, img_width)))
plt.show()





model = Sequential()
model.add(Conv2D(256, 3, padding='same', activation='relu', input_shape=(img_height, img_width, 3)))
model.add(Conv2D(256, 3, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(3, 3)))
model.add(Conv2D(256, 3, padding='same', activation='relu'))
model.add(Dropout(0.2))

model.add(Conv2D(256, 3, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size = (3, 3)))
model.add(Dropout(0.2))

model.add(Conv2D(512, 3, padding='same', activation='relu'))
model.add(Conv2D(1024, 3, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(3, 3)))
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(2048, activation='relu'))
model.add(Dense(4096, activation='relu'))
model.add(Dense(len(train_generator.class_indices), activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])






history = model.fit(
    train_generator,
    steps_per_epoch = train_generator.samples // batch_size,
    validation_data = validation_generator,
    validation_steps = validation_generator.samples // batch_size,
    epochs = 45,
    verbose = 1
)




#Оображаем график точности обучения
plt.plot(history.history['accuracy'], 
         label='Доля верных ответов на обучающем наборе')
plt.plot(history.history['val_accuracy'], 
         label='Доля верных ответов на проверочном наборе')
plt.xlabel('Эпоха обучения')
plt.ylabel('Доля верных ответов')
plt.legend()
plt.show()


model.summary()




classes = ['кошкач', 'собакич']
with Image.open('/content/1.jpg') as img:
  img.load()

img = img.resize((img_height, img_width))
img.show()
img = np.array(img)
img = img.reshape( -1,256, 256,3)
prediction = model.predict(img)
print('Животное на пикче это никто иной как - ' + classes[np.argmax(prediction)])