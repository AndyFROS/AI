import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential # Последовательная сеть
# Стандартные слои нейросети
from tensorflow.keras.layers import BatchNormalization, Embedding, Dense
# Сверточные слои
from tensorflow.keras.layers import SpatialDropout1D
# Рекурентные слои
from tensorflow.keras.layers import SimpleRNN

# Самый простой вариант рекурентной сети, обучается быстро, но имеет низкие возможности обучения
# создаём последовательную модель нейросети
modelEL = Sequential()
# преобразовываем каждое слово в многомерный вектор c указанием размерности вектора и длины входных данных
modelEL.add(Embedding(max_words_count, 5, input_length=xLen))
# добавляем слой регуляризации, "выключая" 1D карты объектов из эмбединг векторов, во избежание переобучения
modelEL.add(SpatialDropout1D(0.2))
# добавляем слой нормализации данных
modelEL.add(BatchNormalization())
# Слой простой рекурентной сети
modelEL.add(SimpleRNN(4, dropout = 0.2))
# добавляем полносвязный слой на 6 нейронов, с функцией активации softmax на выходном слое
modelEL.add(Dense(6, activation = 'softmax'))

# Компиляция, составление модели с выбором алгоритма оптимизации, функции потерь и метрики точности
modelEL.compile(loss = 'categorical_crossentropy', optimizer = 'rmsprop', metrics = ['accuracy'])

# Обучаем сеть с указанием тренировочной выборки, количества эпох, размера минибатча для подачи сети, и тестовой выборки
history = modelEL.fit(xTrain,
                      yTrain,
                      batch_size=512,
                      epochs = 2,
                      validation_data = (xTest, yTest)
                     )

# Строим график для отображения динамики обучения и точности предсказания сети
plt.plot(history.history['accuracy'], 
         label='Доля верных ответов на обучающем наборе')
plt.plot(history.history['val_accuracy'], 
         label='Доля верных ответов на проверочном наборе')
plt.xlabel('Эпоха обучения')
plt.ylabel('Доля верных ответов')
plt.legend()
plt.show()

pred = recognizeMultiClass(modelEL, xTest6Classes, "SimpleRNN") #функция покажет какие классы и как распознаны верно