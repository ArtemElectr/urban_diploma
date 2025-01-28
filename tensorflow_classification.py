import matplotlib.pyplot as plt
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # ERROR 'oneDNN custom operations are on'
import numpy as np
import pathlib
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from sklearn.metrics import f1_score, precision_score, recall_score,accuracy_score

EPOCHS = 15

data_dir = r"F:\urban_diploma\pythonProject\archive\seg_train\seg_train"
test_data_dir = r'F:\urban_diploma\pythonProject\archive\seg_train\seg_train'
pred_dir = r'F:\urban_diploma\pythonProject\archive\seg_pred\seg_pred'
data_dir = pathlib.Path(data_dir)
batch_size = 32
img_height = 150
img_width = 150
class_names = os.listdir(data_dir)

# -------------------------------------------------------------------------------------------------------------------
# ПОДГОТОВКА ТРЕНИРОВОЧНОГО, ПРОВЕРОЧНОГО И ОЦЕНОЧНОГО ДАТАСЕТОВ


train_ds = tf.keras.utils.image_dataset_from_directory(
    directory=data_dir,
    labels='inferred',  #  либо «inferred» (метки генерируются из структуры директории), либо NULL (без меток), либо
    # список или кортеж целочисленных меток такого же размера, как количество файлов изображений в директории. Метки
    # должны быть отсортированы в алфавитном порядке путей к файлам изображений.
    validation_split=0.2,
    # Необязательный параметр с плавающей точкой от 0 до 1, доля данных,резервируемых для проверки
    label_mode='int',
    subset="training",
    class_names=class_names,  # class_names — только если метки «inferred». Это явный список имён классов (должен
    # соответствовать именам поддиректорий). Используется для контроля порядка классов (иначе используется алфавитный
    # порядок)
    seed=123,  # необязательное случайное семя для перетасовки и преобразований.
    image_size=(img_height, img_width),
    batch_size=batch_size,
    shuffle=True  # Перетасовывать ли данные. По умолчанию — True. Если установить False, сортирует данные в алфавитном
    # порядке
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    directory=test_data_dir,
    labels='inferred',
    label_mode='int',
    validation_split=0.2,
    subset="validation",
    class_names=class_names,
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    directory=pred_dir,
    labels=None,
    label_mode='int',
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

# -------------------------------------------------------------------------------------------------------------------
# БУФЕРИЗАЦИЯ ДАТАСЕТОВ
AUTOTUNE = tf.data.AUTOTUNE
# Prefetching in TensorFlow — это оптимизация, которая позволяет перекрывать преобработку данных и обучение модели.
# .prefetch использует фоновый поток и внутренний буфер, чтобы prefetch элементы из входного датасета до того, как они
# будут запрошены
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)
# -------------------------------------------------------------------------------------------------------------------
# НОРМАЛИЗАЦИЯ ДАННЫХ(ДЛЯ МОДЕЛИ)
# Rescaling(scale, offset) Уровень предварительной обработки, который масштабирует входные значения в новый диапазон.
# Этот слой изменяет масштаб каждого значения входных данных (часто изображения), умножая на scale и добавляя offset.
# Чтобы преобразовать ввод в диапазоне [0, 255] в диапазон [0, 1] нужно передать scale=1./255.
normalization_layer = layers.Rescaling(1. / 255)
normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))

num_classes = len(class_names)

data_augmentation = keras.Sequential(
    [
        # .RandomFlip(mode) - Слой предварительной обработки, который случайным образом переворачивает изображения во
        # время обучения. Этот слой будет переворачивать изображения по горизонтали и/или вертикали в зависимости от
        # атрибута mode .RandomRotation(factor) - Слой предварительной обработки, который случайным образом поворачивает
        # изображения во время обучения.  factor=(0.1) приводит к повороту на случайное значение в диапазоне [10% *
        # 2pi, 10% * 2pi] против часовой стрелки как для верхней так и для нижней границы. .RandomZoom(height_factor) -
        # Слой предварительной обработки, который случайным образом увеличивает масштаб изображений во время обучения.
        layers.RandomFlip("horizontal",
                          input_shape=(img_height, img_width, 3)),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
    ]
)
# ------------------------------------------------------------------------------------------------------------------
# ИНИЦИАЛИЗАЦИЯ МОДЕЛИ ОБУЧЕНИЯ
# Conv2D - 2D-слой свертки. Этот слой создаёт ядро свёртки, которое свёртывается с входными данными слоя по одному
# пространственному (или временному) измерению для получения выходного тензора. Если use_bias равно True, создаётся
# вектор смещения, который добавляется к выходным данным. Наконец, если activation не равно None, он также применяется
# к выходным данным. MaxPool2D - Максимальная операция объединения для 2D пространственных данных. Уменьшает размерность
# входного сигнала по пространственным измерениям (высоте и ширине), беря максимальное значение во входном окне
# (размером, определяемым pool_size), для каждого канала входного сигнала. Окно сдвигается на strides по каждому
# измерению. Dropout(rate) - Применяет отсев к входным данным. Слой Dropout случайным образом устанавливает входные
# единицы в 0 с частотой rate на каждом шаге во время обучения, что помогает предотвратить переобучение. Входные данные,
# не установленные в 0, масштабируются на 1 / (1 - rate) таким образом, чтобы сумма всех входных данных оставалась
# неизменной. Flatten Выравнивает входные данные. Не влияет на размер пакета. Dense(units, activation) - Просто ваш
# обычный плотно связанный слой NN. Dense реализует операцию: output = activation(dot(input, kernel) + bias) где
# activation — поэлементная функция активации, переданная в качестве аргумента activation , kernel — матрица весов,
# созданная слоем, а bias — вектор смещения, созданный слоем (применимо только в том случае, если use_bias равно True).
model = Sequential([
    data_augmentation,
    layers.Rescaling(1. / 255, input_shape=(img_height, img_width, 3)),
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes)
])
# НАСТРОЙКА МОДЕЛИ ДЛЯ ОБУЧЕНИЯ
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.summary()  # Просмотрите все слои сети, используя метод модели Model.summary

# ОБУЧЕНИЕ МОДЕЛИ
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS
)

# Визуализируйте результаты тренировок. Создайте графики потерь и точности на обучающих и проверочных наборах:
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
print(model.get_metrics_result())
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(EPOCHS)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')

plt.show()
# -----------------------------------------------------------------------------------------------------------
# ВАЛИДАЦИЯ(ПРОВЕРКА МОДЕЛИ НА ПОМЕЧЕННЫХ ДАННЫХ)
pred_ = model.predict(val_ds)
val_ds_ = val_ds.unbatch()  # сбросить разделение датасета на batch(32)

list_labels = [int(labels) for _, labels in val_ds_]
#for _, labels in val_ds_:
#    list_labels.append(int(labels))
list_predict_labels = [int(np.argmax(tf.nn.softmax(label))) for label in pred_]
#for label in pred_:
#    score_ = tf.nn.softmax(label)
#    list_predict_labels.append(int(np.argmax(score_)))

print(f'Accuracy is {100 * float(accuracy_score(list_labels, list_predict_labels)):.2f} %')
print(f'Recall is {100 * recall_score(list_labels, list_predict_labels, average="weighted"):.2f} %')
print(f'Precision is {100 * precision_score(list_labels, list_predict_labels, average="weighted"):.2f} %')
print(f'f1_score is {100 * f1_score(list_labels, list_predict_labels, average="weighted"):.2f} %')

# ------------------------------------------------------------------------------------------------------------------
# ПРЕДСКАЗАНИЕ ИЗОБРАЖЕНИЙ МОДЕЛЬЮ (НЕПОМЕЧЕННЫЕ ДАННЫЕ)
predict_data = model.predict(test_ds)

for images in test_ds.take(1):
    plt.figure(figsize=(10, 8))
    for i in range(0, 20):
        ax = plt.subplot(4, 5, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        score_ = tf.nn.softmax(predict_data[i])
        plt.title(f'{class_names[np.argmax(score_)]} - {100 * np.max(score_):.0f} %')
        plt.axis("off")
        print(
            "This image most likely belongs to {} with a {:.2f} percent confidence."
            .format(class_names[np.argmax(score_)], 100 * np.max(score_))
        )
    plt.show()

# отдельная картинка
# sunflower_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/592px-Red_sunflower.jpg"
# sunflower_path = tf.keras.utils.get_file('Red_sunflower', origin=sunflower_url)
#
# img = tf.keras.utils.load_img(
#     sunflower_path, target_size=(img_height, img_width)
# )
# img_array = tf.keras.utils.img_to_array(img)
# img_array = tf.expand_dims(img_array, 0) # Create a batch
