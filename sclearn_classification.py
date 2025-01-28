import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from PIL import Image

img_height = 30
img_width = 30

train_image_dir = 'archive/seg_train/seg_train'
test_image_dir = 'archive/seg_test/seg_test'
categories = os.listdir(train_image_dir)


def set_dataset(image_dir):
    np_arr = np.zeros((1, img_height * img_width * 3))
    label_arr = []

    for label, category in enumerate(categories):
        category_dir = os.path.join(image_dir, category)  # к пути добавляется название папки

        for i, filename in enumerate(os.listdir(category_dir)):  # для всех файлов в папке
            if filename.endswith('.jpg'):  # если это картинка(.jpg)
                im = Image.open(os.path.join(category_dir, filename))
                resize_im = im.resize((img_height, img_width))
                pil_im = np.asarray(resize_im)

                if pil_im.shape[0] != img_height or pil_im.shape[1] != img_width:
                    continue

                reshape_pil = np.ravel(pil_im)
                reim = pil_im.reshape(1, reshape_pil.shape[0])
                np_arr = np.append(np_arr, reim, axis=0)
                label_arr.append(label)
    np_arr = np.delete(np_arr, 0,0)

    return np_arr, label_arr


ds, label_arr = set_dataset(train_image_dir)
print("np_arr_shape: ", ds.shape)
print("np_arr_size: ", ds.size)
print('label_arr', len(label_arr))

X, y = ds / 255.0, label_arr
print("X: ", X)
print("X_shape: ", ds.shape)
print("X_size: ", ds.size)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#X, y = image_paths / 255.0, test_image_paths.astype(int)
#X_train, X_test, y_train, y_test = train_test_split(image_paths, image_paths, test_size=0.2, random_state=42)
# Загрузка данных MNIST
mnist = fetch_openml('mnist_784', version=1)
#print(mnist.data)
#X, y = mnist.data / 255.0, mnist.target.astype(int)

# Разделение данных на обучающую и тестовую выборки
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



# Создание и обучение модели k-NN
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Прогнозирование на тестовой выборке
y_pred_knn = knn.predict(X_test)
accuracy_knn = accuracy_score(y_test, y_pred_knn)
f1_sc = f1_score(y_test, y_pred_knn, average='macro')
precision_sc = precision_score(y_test, y_pred_knn, average='macro')
recall_sc = recall_score(y_test, y_pred_knn, average='macro')
confusion_mt = confusion_matrix(y_test, y_pred_knn)

print(f'Accuracy of k-NN: {accuracy_knn:.4f}')
print(f'Precision of k-NN: {precision_sc:.4f}')
print(f'Recall of k-NN: {recall_sc:.4f}')
print(f'confusion_matrix: {confusion_mt}')
