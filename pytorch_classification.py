import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import os
import matplotlib.pyplot as plt
from PIL import Image
from random import randint

EPOCH = 15
# labels = ['buildings', 'forest', 'glacier', 'mountain','sea', 'street']
IMG_SIZE = 150
BATCH_SIZE = 32

# ----------------------------------------------------------------------------------------------------
# НАБИРАЕМ СПИСКИ С ПУТЯМИ К ИЗОБРАЖЕНИЯМИ И МЕТКАМИ КЛАССА ИЗОБРАЖЕНИЯ
image_dir = 'archive/seg_train/seg_train'
test_image_dir = 'archive/seg_test/seg_test'
pred_image_dir = 'archive/seg_pred/seg_pred'
categories = os.listdir(image_dir)

train_image_paths = []
train_labels = []
test_image_paths = []
test_labels = []


def set_dataset(list_images, list_labels, image_dir):
    for label, category in enumerate(categories):
        category_dir = os.path.join(image_dir, category)  # к пути добавляется название папки

        for filename in os.listdir(category_dir):  # для всех файлов в папке
            if filename.endswith('.jpg'):  # если это картинка(.jpg)
                list_images.append(os.path.join(category_dir, filename))  # в массив с путями добавить
                list_labels.append(label)  # в массив с метками добавить


def set_pred_dataset(image_dir):
    return [os.path.join(image_dir, filename) for filename in os.listdir(image_dir) if filename.endswith('.jpg')]


set_dataset(train_image_paths, train_labels, image_dir)  # обучающий датасет
set_dataset(test_image_paths, test_labels, test_image_dir)  # тестовый датасет
predict_images = set_pred_dataset(pred_image_dir)  # датасет для предсказания(не размеченный)
# ----------------------------------------------------------------------------------------------------------
# НОРМАЛИЗАЦИЯ, ПОДГОТОВКА ДАННЫХ К МОДЕЛИ
label_encoder = LabelEncoder()  # Закодируйте целевые метки со значением от 0 до n_классов-1(в ).
# Этот преобразователь следует использовать для кодирования целевых значений, т. е. y, а не входных X.
# .fit_transform() Установите кодировщик меток и верните закодированные меток.
train_labels = label_encoder.fit_transform(train_labels)
test_labels = label_encoder.transform(test_labels)

transform_train = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.CenterCrop(150),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])


# ----------------------------------------------------------------------------------------------------------
# ПОДГОТОВКА ДАТАСЕТОВ ДЛЯ ОБУЧЕНИЯ, ВАЛИДАЦИИ И ПРЕДСКАЗАНИЙ


class IntelImageDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx])
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


class IntelImagePredDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx])
        label = randint(0, 5)

        if self.transform:
            image = self.transform(image)

        return image, label


train_dataset = IntelImageDataset(train_image_paths, train_labels, transform=transform_train)
val_dataset = IntelImageDataset(test_image_paths, test_labels, transform=transform_train)
pred_dataset = IntelImagePredDataset(predict_images, transform=transform_train)
# ----------------------------------------------------------------------------------------------------------
#   ПОДГОТОВКА ЗАГРУЗЧИКА ДЛЯ МОДЕЛИ
# (Загрузчик данных объединяет набор данных и выборку и предоставляет итератор для заданного набора данных.)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
pred_loader = DataLoader(pred_dataset, shuffle=True, num_workers=0)


# ----------------------------------------------------------------------------------------------------------
# ОПИСАНИЕ МОДЕЛИ ОБУЧЕНИЯ


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        # Conv2d Применяет двумерную свёртку к входному сигналу, состоящему из нескольких входных плоскостей.
        # Параметры: 3 - Количество каналов во входном изображении, 6 - количество каналов, создаваемых свёрткой,
        # 5 - размер ядра свёртки
        self.conv1 = nn.Conv2d(3, 32, 5)

        # MaxPool2 Применяет 2D-объединение по максимуму к входному сигналу, состоящему из нескольких входных
        # плоскостей. # Параметры: kernel_size (Union[int, Tuple[int, int]]) — размер окна для вычисления максимума,
        # stride (Union[int, Tuple[int, int]]) –  шаг окна. Значение по умолчанию — kernel_size
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 5)
        # Применяет аффинное линейное преобразование к входящим данным: y=xA^T+b. # Параметры in_features (int) – размер
        # каждой входной выборки, out_features (int) – размер каждой выходной выборки,смещение (bool) — если установлено
        # значение False, слой не будет обучаться с учётом смещения. По умолчанию: True
        self.fc1 = nn.Linear(64 * 289 * 4, 1200)
        self.fc2 = nn.Linear(1200, 1200)
        self.fc3 = nn.Linear(1200, 1200)
        self.fc4 = nn.Linear(1200, 6)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = x.view(-1, 64 * 289 * 4)  # изменить shape тензора
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


# ИНИЦИАЛИЗАЦИЯ МОДЕЛИ
net = Net()
# Оптимизатор PyTorch нужен для того, чтобы помочь в процессе обучения модели машинного обучения. Он регулирует
# параметры модели во время обучения, чтобы минимизировать ошибку между предсказанным и фактическим выходом. Оптимизатор
# использует математический алгоритм для определения лучших настроек регулировки параметров. Этот алгоритм основан на
# ошибке и градиентах (показателе направления наиболее резкого увеличения или уменьшения) функции потерь по отношению к
# параметрам.
import torch.optim as optim

criterion = nn.CrossEntropyLoss()  #  это функция для вычисления кросс-энтропийных потерь между входным и целевым
# значением в библиотеке PyTorch
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)  # lr=0.001
# ---------------------------------------------------------------------------------------------
# ОБУЧЕНИЕ МОДЕЛИ
for epoch in range(EPOCH):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        optimizer.zero_grad()  # Сбрасывает градиенты всех оптимизированных тензоров
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()  # вычисления градиентов с помощью, например, backward()
        optimizer.step()  # Все оптимизаторы реализуют метод step() для обновления параметров.

        # print statistics
        running_loss += loss.item()
        if i % 200 == 199:  # print every 200 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 200:.3f}')
            running_loss = 0.0

print('Finished Training')

# Давайте быстро сохраним нашу обученную модель:

PATH = './intel_image_v_0_1.pth'
torch.save(net.state_dict(), PATH)
# ЗАГРУЗКА СОХРАНЕННОЙ МОДЕЛИ
net = Net()
net.load_state_dict(torch.load(PATH, weights_only=True))


# ФУНКЦИЯ ПОДГОТАВЛИВАЕТ ИЗОБРАЖЕНИЕ
def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))


# --------------------------------------------------------------------------------------------------
# ВАЛИДАЦИЯ МОДЕЛИ (ПРОВЕРКА НА ВАЛИДАЦИОННЫХ РАЗМЕЧЕННЫХ ДАННЫХ)
correct = 0
total = 0

# поскольку мы не тренируемся, нам не нужно вычислять градиенты для наших выходных данных
with torch.no_grad():
    # torch.no_grad() — это контекстный менеджер в PyTorch, который отключает вычисление градиентов.
    # Он полезен на этапах оценки или тестирования модели, когда не нужно вычислять градиенты, что
    # экономит память и вычислительные ресурсы.
    list_predict = []
    list_labels = []

    for data in val_loader:
        images, labels = data
        # вычисляйте выходные данные, прогоняя изображения по сети
        outputs = net(images)
        # класс с самой высокой энергией - это то, что мы выбираем в качестве прогноза
        _, predicted = torch.max(outputs, 1)  # predicted - tensor, в котором предсказанные индексы классов
        # predicted, label - тензоры с одним измерением( [., ., ...] )

        list_predict.extend(predicted)
        list_labels.extend(labels)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    print(f'ACCURACY IS {100 * accuracy_score(list_labels, list_predict):.2f} %')
    print(f'RECALL  IS {100 * recall_score(list_labels, list_predict, average="weighted"):.2f} %')
    print(f'PRECISION IS {100 * precision_score(list_labels, list_predict, average="weighted"):.2f} %')
    print(f'F1 score  IS {100 * f1_score(list_labels, list_predict, average="weighted"):.2f} %')
    accuracy = 100 * correct / total
    print(f'REAL ACCURACY is {accuracy:.2f}')

# приготовьтесь подсчитывать прогнозы для каждого класса
correct_pred = {classname: 0 for classname in categories}  # словарная сборка
total_pred = {classname: 0 for classname in categories}

# again no gradients needed
with torch.no_grad():
    for data in val_loader:
        images, labels = data
        outputs = net(images)
        _, predictions = torch.max(outputs, 1)

        # collect the correct predictions for each class
        for label, prediction in zip(labels, predictions):  # Функция zip в Python — это инструмент, объединяющий
            # элементы нескольких итерируемых объектов (кортежей, списков, строк) в пары. Она возвращает итератор, где
            # каждый кортеж состоит из элементов, взятых по одному из каждого исходного объекта.
            if label == prediction:
                correct_pred[categories[label]] += 1
            total_pred[categories[label]] += 1

# print accuracy for each class
for classname, correct_count in correct_pred.items():
    accuracy = 100 * float(correct_count) / total_pred[classname]
    print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')

# --- ПРЕДСКАЗАНИЕ МОДЕЛЬЮ ИЗОБРАЖЕНИЙ ----------------------------------------------------------------------------
count = 0
with torch.no_grad():
    net.eval()  # отключить обучение модели, включить оценку
    plt.figure(figsize=(10, 8))
    for data in pred_dataset:
        images, labels = data
        outputs = net(images)
        ax = plt.subplot(4, 5, count + 1)
        _, predictions = torch.max(outputs, 1)
        plt.title(f'{categories[predictions]}')
        plt.axis("off")
        imshow(images)

        count += 1
        if count > 19:
            break
    plt.show()
# -----------------------------------------------------------------------------------------------------------------

