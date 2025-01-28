# Импорт необходимых библиотек
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Input
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Загрузка данных
df = pd.read_csv(r'F:\urban_diploma\pythonProject\data_regression\financial_regression.csv')

# Преобразование данных
df = df.drop(df[df['gold_close'].isna()].index)
df = df['gold_close'].values.astype(float)

print(df)   # [.........]

# Нормализация данных
scaler = MinMaxScaler(feature_range=(0, 1))

data_normalized = scaler.fit_transform(df.reshape(-1, 1))  # [[.][.][.]]


# Создание обучающих и тестовых выборок
def create_dataset(data, time_step=1):
    X, y = [], []
    for i in range(len(data) - time_step - 1):
        print('X', data[i:(i + time_step), 0])
        print('y', data[i + time_step, 0])
        X.append(data[i:(i + time_step), 0])   # range от нулевого элемента до 0+time_step, не включая timestep
        y.append(data[i + time_step, 0])       # элемент с индексом timestep(на 1 больше range)
    return np.array(X), np.array(y)


time_step = 5
X, y = create_dataset(data_normalized, time_step)   # shape X : (3898, 5), shape y: (3898,) 3898 одномерных массивов по
                                                    # 5(1) элементу в массиве  X shape = [[.....][.....][.....]]

# Разделение на обучающие и тестовые данные
train_size = int(len(X) * 0.8)  # 0.8 от кол.-ва элементов в Х

test_size = len(X) - train_size  # 0.2 от кол.-ва элементов в Х
X_train, X_test = X[0:train_size], X[train_size:]  # разделение массивов
y_train, y_test = y[0:train_size], y[train_size:]


# Изменение формы данных для LSTM [samples, time steps, features]
#
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1) # X_train.shape =[ [[.....]] [[.....]] [[.....]] ]
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Построение модели LSTM
model = Sequential()
model.add(Input(shape=(time_step, 1)))  # step = 5
model.add(LSTM(50, return_sequences=True))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(60, activation='linear'))
model.add(Dense(40))
model.add(Dense(1))
#
#
# # Компиляция и обучение модели
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, batch_size=1, epochs=30)
#
# # Предсказание и оценка модели
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)
#среднеквадратичная ошибка (MSE)
real_data = data_normalized[-len(test_predict):]
#
print(f'MAE - {mean_absolute_error(real_data, test_predict)}')
print(f'MSE - {mean_squared_error(real_data, test_predict)}')
#
# # Обратное преобразование предсказанных значений к исходной шкале
train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)
y_train = scaler.inverse_transform([y_train])
y_test = scaler.inverse_transform([y_test])

#
#
# # Визуализация результатов
plt.figure(figsize=(10, 6))
plt.plot(scaler.inverse_transform(data_normalized), label='Исходные данные')
#
# # Построение прогноза для обучающей выборки
print('shape: ', test_predict.shape)

plt.plot(np.arange(time_step, train_size + time_step), train_predict, label='Прогноз (обучение)')
# numpy.arange() — это встроенная в библиотеку NumPy функция, которая возвращает объект типа ndarray, содержащий
# равномерно расположенные значения внутри заданного интервала
#
# # Построение прогноза для тестовой выборки
plt.plot(np.arange(train_size + time_step, train_size + time_step + len(test_predict)), test_predict, label='Прогноз (тест)')
#
plt.legend()
plt.xlabel('Дни')
plt.ylabel('Цена')
plt.title('Прогнозирование цены золота с помощью Tensorflow LSTM')
plt.show()
