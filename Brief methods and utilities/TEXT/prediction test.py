import numpy as np


class_name = ['О. Генри', 'Стругацкие', 'Булгаков', 'Саймак', 'Фрай', 'Брэдберри'] # Все классы
n_classes = len(class_name) # Количество класов


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



# Представляем тестовую выборку в удобных для распознавания размерах
def createTestMultiClasses(wordIndexes, x_len, step): #функция принимает последовательность индексов, размер окна, шаг окна
    #Для каждого из 6 классов
    #Создаём тестовую выборку из индексов
    n_classes = len(wordIndexes) #Задаем количество классов
    xTest6Classes01 = []               #Здесь будет список из всех классов, каждый размером "кол-во окон в тексте * 20000 (при maxWordsCount=20000)"
    xTest6Classes = []                 #Здесь будет список массивов, каждый размером "кол-во окон в тексте * длину окна"(6 по 420*1000)
    for wI in wordIndexes:                       #Для каждого тестового текста из последовательности индексов
        sample = (get_set_from_index(wI, x_len, step)) #Тестовая выборка размером "кол-во окон*длину окна"(например, 420*1000)
        xTest6Classes.append(sample)              # Добавляем в список
        xTest6Classes01.append(tokenizer.sequences_to_matrix(sample)) #Трансформируется в Bag of Words в виде "кол-во окон в тексте * 20000"
    xTest6Classes01 = np.array(xTest6Classes01)                     #И добавляется к нашему списку, 
    xTest6Classes = np.array(xTest6Classes)                     #И добавляется к нашему списку, 
  
    return xTest6Classes01, xTest6Classes  #функция вернёт тестовые данные: TestBag 6 классов на n*20000 и xTestEm 6 по n*1000

# Распознаём тестовую выборку и выводим результаты
def recognizeMultiClass(model, xTest, modelName):
    print("НЕЙРОНКА: ", modelName)
    print()
  
    totalSumRec = 0 # Сумма всех правильных ответов
  
    #Проходим по всем классам
    for i in range(n_classes):
        #Получаем результаты распознавания класса по блокам слов длины xLen
        currPred = model.predict(xTest[i])
        #Определяем номер распознанного класса для каждохо блока слов длины xLen
        currOut = np.argmax(currPred, axis=1)

        evVal = []
        for j in range(n_classes):
            evVal.append(len(currOut[currOut==j])/len(xTest[i]))

        totalSumRec += len(currOut[currOut==i])
        recognizedClass = np.argmax(evVal) #Определяем, какой класс в итоге за какой был распознан

        #Выводим результаты распознавания по текущему классу
        isRecognized = "Это НЕПРАВИЛЬНЫЙ ответ!"
        if (recognizedClass == i):
            isRecognized = "Это ПРАВИЛЬНЫЙ ответ!"
        str1 = 'Класс: ' + class_name[i] + " " * (11 - len(class_name[i])) + str(int(100*evVal[i])) + "% сеть отнесла к классу " + class_name[recognizedClass]
        print(str1, " " * (55-len(str1)), isRecognized, sep='')
  
    #Выводим средний процент распознавания по всем классам вместе
    print()
    sumCount = 0
    for i in range(n_classes):
        sumCount += len(xTest[i])
    print("Средний процент распознавания ", int(100*totalSumRec/sumCount), "%", sep='')

    print()
  
    return totalSumRec/sumCount


x_len = 1000
step = 100
test_word_indexes = [] # Тут должна быть последовательность

xTest6Classes01, x2 = createTestMultiClasses(test_word_indexes, x_len, step) #Преобразование тестовой выборки
#Проверяем точность нейронки обученной на bag of words
pred = recognizeMultiClass(model, xTest6Classes01, "Тексты 01 + Dense")