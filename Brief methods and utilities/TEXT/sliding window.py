import numpy as np
from tensorflow import keras
from keras import utils #Для работы с катеориальными данными


# Формирование обучающей и тестовой выборки по листу индексов слов
# Разделение на короткие векторы
def get_set_from_index(word_indexes, x_len, step): # Функция принимает последовательности индексов, размер окна, шаг окна
    x_sample = [] # Обьявление списка для векторов
    words_len = len(word_indexes) # Считаем количество слов
    index = 0 # Начальный индекс
    
    while (index + x_len <= words_len): # Идем по всей длинге вектора индексов
        x_sample.append(word_indexes[index:index+x_len]) # Откусываем вектор длинны x_len
        index += step # Смещение вперед на step\
    return x_sample



# Формирование обучающей и проверочной выборки
# Из 2х листов индексов от двух классво
def create_set_multiclasses(word_indexes, x_len, step): # Функция принимает последовательность индексов, размер окна, шаг окна
    # Для каждого из 6ти классов 
    # Создаем обучающую/тестовую выборку из индексов
    n_classes = len(word_indexes) # Задаем количество классов выборки
    classes_x_samples = [] # Здесь будет список размером "колво классов * колво окон в тексте * длину окна (например 6 по 1341*1000)"
    for wI in word_indexes: # Для каждого текста выборки из последовательности индексов
        classes_x_samples.append(get_set_from_index(wI, x_len, step)) # Добавление в список текст индексов, разбитый на колво окон * длину окна
            
    # Формируем один общий xSamples
    x_samples = [] # Здесь будет список размером "суммарное кол-во окон во всех текстах*длину окна (например, 15779*1000)"
    y_samples = [] # Здесь будет список размером "суммарное кол-во окон во всех текстах*вектор длиной 6"
    
    for t in range(n_classes): # В диапозоне количества классов (6)
        xT = classes_x_samples[t] # Берем текст вида "колво окон в тексте * длину окна (например 6 по 1341*1000)"
        for i in range(len(xT)): # И каждое его окно
            x_samples.append(xT[i]) # Добавляем в общий список выборки
            y_samples.append(utils.to_categorical(t, n_classes)) # Добавляем соответствующий вектро класса
            
    x_samples = np.array(x_samples) # Перевод в массив numpy для подачи в нейронку
    y_samples = np.array(y_samples) # Перевод в массив numpy для подачи в нейронку
    return (x_samples, y_samples)


x_len = 1000
step = 100

train_word_indexes = [] # Тут должна быть последовательность (sequnces)
test_word_indexes = [] # Тут должна быть последовательность (sequnces)
X_train, y_train = create_set_multiclasses(train_word_indexes, x_len, step)
X_test, y_test = create_set_multiclasses(test_word_indexes, x_len, step)