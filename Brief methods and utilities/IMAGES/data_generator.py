import os
import random
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator


train_path = 'data/train/' # Путь к тренировочным данным
test_path = 'data/test/' # Путь к тестовым данным

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
    batch_size = batch_size, # размер пакетов данных
    shuffle = True, # Перемешка выборки
    class_mode = 'categorical', # Категориальный тип выборки. Рахбиение выборки по маркам авто
    subset = 'training' # Установка выборки как обучающая
)

validation_generator = data_generator.flow_from_directory(
    train_path, # Пусть ко всей выборке
    target_size = (img_height, img_width), # Размеры изображений
    batch_size = batch_size, # размер пакетов данных
    shuffle = True, # Перемешка выборки
    class_mode = 'categorical', # Категориальный тип выборки. Рахбиение выборки по маркам авто
    subset = 'validation' # Установка выборки как валидационная
)