from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# -----------------------------------------------------------------------------------------------------------------
# ПОДГОТОВКА ДАННЫХ(НОРМАЛИЗАЦИЯ, УДАЛЕНИЕ ПУСТЫХ ЗНАЧЕНИЙ)
df = pd.read_csv(r'F:\urban_diploma\pythonProject\data_regression\financial_regression.csv')
# Преобразование данных
df = df.drop(df[df['gold_close'].isna()].index)
df = df['gold_close'].values.astype(float)
scaler = MinMaxScaler(feature_range=(0, 1))
data_normalized = scaler.fit_transform(df.reshape(-1, 1))  # [[.][.][.]]
# ----------------------------------------------------------------------------------------------------------------
# ПОДГОТОВКА ОБУЧАЮЩЕГО И ВАЛИДАЦИОННОГО ДАТАСЕТА


def create_dataset(data, time_step=1):
    X, y = [], []
    for i in range(len(data) - time_step - 1):
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
# -------------------------------------------------------------------------------------------------------------------
# ИНИЦИАЛИЗАЦИЯ, ОБУЧЕНИЕ МОДЕЛИ
model = LinearRegression()
model.fit(X_train, y_train)
# -------------------------------------------------------------------------------------------------------------------
# ВАЛИДАЦИЯ МОДЕЛИ
x_pred = model.predict(X_train)
y_pred = model.predict(X_test)
print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred):.5f}")
print(f'Mean Absolute Error: {mean_absolute_error(y_test, y_pred):.5f}')

# # Обратное преобразование предсказанных значений к исходной шкале
test_predict = np.reshape(y_pred,(-1,1))
train_predict = np.reshape(x_pred,(-1,1))
train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)
y_train = scaler.inverse_transform([y_train])
y_test = scaler.inverse_transform([y_test])
# # Построение прогноза для обучающей выборки

plt.plot(scaler.inverse_transform(data_normalized), label='Исходные данные')
plt.plot(np.arange(time_step, train_size + time_step), train_predict, label='Прогноз (обучение)')
# numpy.arange() — это встроенная в библиотеку NumPy функция, которая возвращает объект типа ndarray, содержащий
# равномерно расположенные значения внутри заданного интервала
#
# # Построение прогноза для тестовой выборки
plt.plot(np.arange(train_size + time_step, train_size + time_step + len(test_predict)),
         test_predict, label='Прогноз (тест)')
plt.legend()
plt.xlabel('Дни')
plt.ylabel('Цена')
plt.title('Прогнозирование цены золота с помощью SKLearn LinearRegression')
plt.show()

