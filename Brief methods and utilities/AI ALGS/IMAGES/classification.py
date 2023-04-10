from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization, Conv2D, MaxPooling2D, Dropout, Flatten, Dense


# Создание модели
# BatchNormalization - метод, который позволяет повысить производительность и стабилизировать работу искусственных нейронных сетей.

# Conv2D - cлой 2D свертки (например, пространственная свертка над изображениями). 
#Этот слой создает ядро свертки, которое свертывается со входом слоя для получения тензора выходов. 
#(Количество карт свертки, размер ядра свертки, заполнять срезанные пиксели, активационная функция relu)

#MaxPooling2D - операция максимальной подвыборки(субдискретизации) для пространственных данных.
#pool_size: целое число или кортеж из 2-х целых чисел, факторы, по которым следует уменьшать масштаб (вертикальный, горизонтальный).
#(2, 2) уменьшит входное значение в обоих пространственных измерениях наполовину.

#Dropout - dropout) — метод регуляризации искусственных нейронных сетей, предназначен для уменьшения переобучения сети
#за счет предотвращения сложных коадаптаций отдельных нейронов на тренировочных данных во время обучения.

#Flatten - возвращает копию массива сжатую до одного измерения.

#Dense - реализует операцию: output = activation(dot(input, kernel) + bias), где активация — это функция активации по элементам, переданная в качестве аргумента
#активации, кернел — это матрица весов, созданная слоем, а смещение — это вектор смещения, созданный слоем (применимо только в случае, если use_bias — True).

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





#Наглядная демонстрация классификации нейронной сети
#Классы
classes = ['name1', 'name2']
for n in range(len(classes)): # При помощи цикла прохожусь по всем классам
  for i in os.listdir(test_path+classes[n]): # Вложенным циклом прохожусь по всем картинкам текущего класса
    test_img_path = test_path+classes[n]+'/'+i # Получение полного пути к изображению
    
    with Image.open(test_img_path) as img: # Открытие изображения
      img.load() # Загрузка изображения
    img = img.resize((img_height, img_width), 3) # Изменение размеров изображения
    img = np.array(img)[None] # Создание матрицы numpy из изображения с размерностью (1, 256, 256, 3)
    plt.figure() # Создание фигуры
    plt.title(f'Изображение из класса {classes[n]}\nСеть отнесла к классу {classes[np.argmax(model.predict(img,verbose = 0))]}') # Заголовок пигуры
    img = img[0] # Распкаовка батча для получения матрицы изображения
    plt.imshow(np.squeeze(img)) # Преобразование матрицы в изображение и отображение ее на экране