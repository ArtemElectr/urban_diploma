import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import torch.nn.functional as F


df = pd.read_csv(r'F:\urban_diploma\pythonProject\data_regression\financial_regression.csv')
# Преобразование данных
df = df.drop(df[df['gold_close'].isna()].index)
df = df['gold_close'].values.astype(float)
scaler = MinMaxScaler(feature_range=(0, 1))
data_normalized = scaler.fit_transform(df.reshape(-1, 1))  # [[.][.][.]]


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

# Convert data to PyTorch tensors
X_train_ts = torch.from_numpy(X_train.astype(np.float32))
y_train_ts = torch.from_numpy(y_train.astype(np.float32)).view(-1, 1)
x_test_ts = torch.from_numpy(X_test.astype(np.float32))


# Define the linear regression model
class LinearRegressionModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_size, 200)
        self.l2 = nn.Linear(200, 400)
        self.l3 = nn.Linear(400, output_size)

    def forward(self, x):
        x = F.relu(self.linear(x))
        x = F.relu(self.l2(x))
        return self.l3(x)


# Model parameters and hyperparameters
input_size = X_train_ts.shape[1]
output_size = 1
learning_rate = 0.01
num_epochs = 1000

# Instantiate the model, loss function, and optimizer
print(input_size, ' ', output_size)
model = LinearRegressionModel(input_size, output_size)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)#SGD(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    # Forward pass
    y_predicted = model(X_train_ts)

    # Compute loss
    loss = criterion(y_predicted, y_train_ts)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Print loss every 10 epochs
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')


# Evaluation
with torch.no_grad():
    train_pred = model(X_train_ts).detach().numpy()
    test_pred = model(x_test_ts).detach().numpy()

# Calculate Mean Squared Error
mse = mean_squared_error(y_test, test_pred)
mae = mean_absolute_error(y_test, test_pred)
print("Mean Squared Error: %.4f" % mse)
print("Mean Absolute Error: %.4f" % mae)


# # Обратное преобразование предсказанных значений к исходной шкале
train_predict = scaler.inverse_transform(train_pred)
test_predict = scaler.inverse_transform(test_pred)
y_train = scaler.inverse_transform([y_train])
y_test = scaler.inverse_transform([y_test])
# # Построение прогноза для обучающей выборки
plt.plot(scaler.inverse_transform(data_normalized), label='Исходные данные')
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
plt.title('Прогнозирование цены золота с помощью Pytorch LinearRegressionModel')
plt.show()