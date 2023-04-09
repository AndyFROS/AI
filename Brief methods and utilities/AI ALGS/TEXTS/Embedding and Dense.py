from tensorflow.keras.models import Sequential #Полносвязная модель
from tensorflow.keras.layers import BatchNormalization, Dense, Dropout, Flatten, Activation, Embedding, SpatialDropout1D #Слои
import matplotlib.pyplot as plt

'''
Подается набор токенов, каждому токену проставщяется набор векторов (матрица из веторов)
Обычный Dropout [
он-----[..X......],
пошел--[.....X...],
гулять-[..X......],
рано---[.........],
утроим-[.......X.],
]
SpatialDropout1D
[
он-----[X........],
пошел--[X........],
гулять-[X........],
рано---[X........],
утроим-[X........],
]

Для работы с последовательностями нужно верно определить длинну, до которой их можно будет соединить.
Это мождно сделать при помощи гистограммы, чтобы узнать какие длинны чаще всего используются.
'''

# Для embedding слоя, текст нужно подать в виде последовательности индексов слов, каждый из которых потом преобразуем в многомерный вектор
sequences = tokenizer.texts_to_sequences(texts) # Разбиение текстов на последовательности индексов
np_sequences = np.array(sequences) # Перевод в массив numpy

# Разбиение всех данных на обучающую и тестовую выборку при помощи метода train_test_split
X_train_e, X_test_e, y_train_e, y_test_e = train_test_split(np_sequences, y_all, test_size=0.2)
print(X_train_e.shape) # Форма текстов обучающей выборки
print(y_train_e.shape) # Форма соответствующих им классов


len_x_train_e = [len(x) for x in X_train_e]
plt.hist(len_x_train_e, 40)
plt.show()

x_len = 1000
max_words_count = [] # Тут должна быть последовательность
model = Sequential()
# Слой, который ставит соответствие слову, его вектор (обучается в составе нейросети)
# Embedding(размер словоря, размер пространства, входная длинна)
model.add(Embedding(max_words_count, 20, input_length=x_len))
# Гасит не рандомные части 
model.add(SpatialDropout1D(0.2))
# Выпремление матрицы в один вектор
model.add(Flatten())
model.add(BatchNormalization())
model.add(Dense(256, activation = 'relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(Dense(6, activation = 'softmax'))

model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

model.summary()

history = model.fit(X_train,
                    y_train,
                    epochs = 10,
                    batch_size = 128,
                    validation_data = (X_test, y_test)
                   )

plt.plot(history.history['accuracy'], label = 'Качетсво на тренировочном наборе')
plt.plot(history.history['val_accuracy'], label = 'Качетсво на тестовом наборе')
plt.xlabel('Количесво эпох')
plt.ylabel('Качество обучения')
plt.show()




