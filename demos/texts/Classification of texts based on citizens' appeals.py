import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow.keras import utils #Для работы с катеориальными данными
from tensorflow.keras.models import Sequential #Полносвязная модель
from tensorflow.keras.layers import BatchNormalization, Dense, Dropout, Flatten, Activation, Embedding, SpatialDropout1D #Слои
from tensorflow.keras.preprocessing.text import Tokenizer #Методы для роботы с текстами и преобразования их в последовательности
from tensorflow.keras.preprocessing.sequence import pad_sequences #Метод для работы с последовательностями

from sklearn.preprocessing import LabelEncoder #Метол кодирования данных
from sklearn.model_selection import train_test_split #Разделение на выборки


df = pd.read_csv('Базы/clean_data.csv')
df = df.iloc[:, :2]
df.head(10)



#Обрабатываем выборку

maxCountStrings = 400 #Задаем максимальное число строк для классов
minCountStrings = 300 #Задаем минимальное число строк для классов
df = df.dropna().reset_index() #Удаляем пустые значения и обновляем индексы в таблице

for cl in df['category'].unique(): #Проходим по всем классам
    initialLen = df[df.category == cl].shape[0]
    if(df[df.category == cl].shape[0] < minCountStrings): #Если в классе количество строк меньше minCountStrings
        df = df.drop(df[df.category == cl].index) #Удаляем данный класс
    if(df[df.category == cl].shape[0] > maxCountStrings): #Если в классе количество строк больше maxCountStrings
        df = df.drop(df[df.category == cl].index[maxCountStrings:]) #Оставляем в таблице строки количеством maxCountStrings
    print('Количество записей класса ', cl, ': ', initialLen, '. В выборку вошло: ', df[df.category == cl].shape[0], sep='')

df = df.reset_index() #Обновляем индексы в таблице

df.shape
print(df.values[0])

for cl in df['category'].unique():
    print(f'{cl} :', len(df[df['category'] == cl]))



texts = df['text'].values # Получение всехз текстов
classes = list(df['category'].values) # Получение всех соответсствующих классов
max_words_count = 60000 # Максимальное количество слов\индексов, учитываемое при обучении
n_classes = df['category'].nunique() + 1
print(n_classes)


########################################
# Токенизация
########################################

# Использование встроенной в Keras функции Tokenizers для разбиения текста и превращения в матрицу числовых значений
tokenizer = Tokenizer(
    num_words = max_words_count, # Определяем максимальное количество слов/индексов, учитываемое при обучении текстов
    filters = '!"#$%&()*+,-–—./…:;<=>?@[\\]^_`{|}~«»\t\n\xa0\ufeff', # Избавляемся от ненужных символов
    lower = True, # Приведение всех слов к нижнему регистру
    split = ' ', # Разделение слов по пробелу
    oov_token = 'unknown', # Чтобы показать, где находится пропущенное слово
    char_level = False # Не удалять однобуквенные слова
)

tokenizer.fit_on_texts(train_text) # Даем тексты в метод, который соберет словарь частотности
# Формирование матрицы по принципу bag_of_words
x_all = tokenizer.texts_to_matrix(texts) # Каждое слово из текста нашло свой индекс в векторе длинной max_words_count и отметилось в нем единичкой
print(x_all.shape)# Посмотрим на форму текстов
print(x_all[0, :20])# И отдельно на фрагмент начала вектора

# print(tokenizer.word_index.items()) #Вытаскиваем индексы слов для просмотра
print('Размер словоря - ', len(tokenizer.word_index.items()))



########################################
# Преобразование категорий в веторы
########################################
encoder = LabelEncoder() # Метод для кодирования меток
encoded_classes = encoder.fit_transform(classes) # Подгружаем категории из базы
print(encoded_classes.shape)
print(encoded_classes[:10])



y_all = utils.to_categorical(encoded_classes, n_classes) # Выводим каждую метку в виде вектора длинной 22, с 1кой в позиции соответсвующего класса и нулями
print(y_all.shape)
print(y_all[0])



########################################
# Создание обучающей и проверочной выборки
########################################

# Разбиение всех данных на обучающую и тестовую выборку при помощи метода train_test_split
X_train, X_test, y_train, y_test = train_test_split(x_all, y_all, test_size=0.2)
print(X_train.shape) # Форма текстов обучающей выборки
print(y_train.shape) # Форма соответствующих им классов


########################################
# Преобразование текста в последовательности
########################################

# Для embedding слоя, текст нужно подать в виде последовательности индексов слов, каждый из которых потом преобразуем в многомерный вектор
sequences = tokenizer.texts_to_sequences(texts) # Разбиение текстов на последовательности индексов
np_sequences = np.array(sequences) # Перевод в массив numpy

# Разбиение всех данных на обучающую и тестовую выборку при помощи метода train_test_split
X_train_e, X_test_e, y_train_e, y_test_e = train_test_split(np_sequences, y_all, test_size=0.2)
print(X_train_e.shape) # Форма текстов обучающей выборки
print(y_train_e.shape) # Форма соответствующих им классов




########################################
# Написание нейросети Texts + Dense
########################################

model = Sequential()
# Первый полносвязный слой
model.add(Dense(128, input_dim=max_words_count, activation='relu'))
# Первый слой регуляризации Dropout
model.add(Dropout(0.4))

# Второй полносвязный слой
model.add(Dense(256, activation='relu'))
# Второй слой регуляризации Dropout
model.add(Dropout(0.4))

# Третий полносвязный слой
model.add(Dense(128, activation='relu'))
# Третий слой регуляризации Dropout
model.add(Dropout(0.4))

# Выходной полносвязный слой
model.add(Dense(n_classes, activation='softmax'))

# Компиляция модели с функцией потери categorical_crossentropy оптимизатором adam и метрикой accurasy
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

# Обучение модели
history = model.fit(X_train,
                    y_train,
                    epochs = 20,
                    batch_size = 128,
                    validation_data = (X_test, y_test),
                    verbose = 1
                   )

# График отражающий точность обучения обучения
plt.plot(history.history['accuracy'], label = 'Точность не тренировочном наборе') # Точность на тренеровочном наборе
plt.plot(history.history['val_accuracy'], label = 'Точность не тестовом наборе') # Точность на тестовом наборе
plt.xlabel('Количество эпох') # Наименование оси X
plt.ylabel('Точность модели') # Наименование оси y
plt.show() # Показать полотно






########################################
# Написание нейросети Embedding + Dense
# Для работы с последовательностями нужно верно определить длинну, до которой их можно будет соединить.
# Это мождно сделать при помощи гистограммы, чтобы узнать какие длинны чаще всего используются.
########################################

len_x_train_e = [len(x) for x in X_train_e]
plt.hist(len_x_train_e, 40)
plt.show()




# Как можно увидеть выше, самая оптимальная длинна - 400, все что будет больше, обрежется, меньше дополнится при помощи padding

max_len = 400
# Преобразование входных векторов
X_train_e = pad_sequences(X_train_e, maxlen = max_len) 
X_test_e = pad_sequences(X_test_e, maxlen = max_len)

# Построение модели
model = Sequential()

# Слой, который ставит соответствие слову, его вектор (обучается в составе нейросети)
# Embedding(размер словоря, размер пространства, входная длинна)
model.add(Embedding(max_words_count, 50, input_length=max_len))
# Гасит не рандомные части 
model.add(SpatialDropout1D(0.2))
# Выпремление матрицы в один вектор
model.add(Flatten())

#Слой пакетной нормализации
model.add(BatchNormalization())
# Первый полносвязный слой
model.add(Dense(128, input_dim=max_words_count, activation='relu'))
# Первый слой регуляризации Dropout
model.add(Dropout(0.2))

# Второй полносвязный слой
model.add(Dense(256, activation='relu'))
# Второй слой регуляризации Dropout
model.add(Dropout(0.4))

# Третий полносвязный слой
model.add(Dense(128, activation='relu'))
# Третий слой регуляризации Dropout
model.add(Dropout(0.2))

# Выходной полносвязный слой
model.add(Dense(n_classes, activation='softmax'))

# Компиляция модели с функцией потери categorical_crossentropy оптимизатором adam и метрикой accurasy
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

# Обучение модели
history = model.fit(X_train_e,
                    y_train_e,
                    epochs = 20,
                    batch_size = 128,
                    validation_data = (X_test_e, y_test_e),
                    verbose = 1
                   )

# График отражающий точность обучения обучения
plt.plot(history.history['accuracy'], label = 'Точность не тренировочном наборе') # Точность на тренеровочном наборе
plt.plot(history.history['val_accuracy'], label = 'Точность не тестовом наборе') # Точность на тестовом наборе
plt.xlabel('Количество эпох') # Наименование оси X
plt.ylabel('Точность модели') # Наименование оси y
plt.show() # Показать полотно



















