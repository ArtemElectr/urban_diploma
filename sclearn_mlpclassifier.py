import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, f1_score
from PIL import Image
import matplotlib.pyplot as plt


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
X, y = ds / 255.0, label_arr

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Создание и обучение модели k-NN
mlp = MLPClassifier(hidden_layer_sizes=(200,), activation='relu', solver='adam',
                    max_iter=200, shuffle=True, early_stopping=True)
mlp.fit(X_train, y_train)

# Прогнозирование на тестовой выборке
y_pred_mlp = mlp.predict(X_test)
print(y_pred_mlp)
accuracy_mlp = accuracy_score(y_test, y_pred_mlp)
f1_sc = f1_score(y_test, y_pred_mlp, average='weighted')
precision_sc = precision_score(y_test, y_pred_mlp, average='weighted')
recall_sc = recall_score(y_test, y_pred_mlp, average='weighted')


print(f'Accuracy of k-NN: {100 * accuracy_mlp:.2f} %')
print(f'Precision of k-NN: {100 * precision_sc:.2f} %')
print(f'Recall of k-NN: {100 * recall_sc:.2f} %')
print(f'f1 of k-NN: {100 * f1_sc:.2f} %')

plt.figure(figsize=(10, 8))
count = 0
for i, category in enumerate(categories):
    category_dir = os.path.join('archive/seg_test/seg_test', category)  # к пути добавляется название папки

    for j, filename in enumerate(os.listdir(category_dir)):  # для всех файлов в папке
        if filename.endswith('.jpg'):
            im = Image.open(os.path.join(category_dir, filename))
            resize_im = im.resize((img_height, img_width))
            pil_im = np.asarray(resize_im)
            reshape_pil = np.ravel(pil_im)
            reim = pil_im.reshape(1, reshape_pil.shape[0])

            predicted = mlp.predict(reim)
            ax = plt.subplot(5, 5, count + 1)
            count += 1
            plt.imshow(im)
            plt.title(f'{categories[predicted[0]]} ')
            plt.axis("off")
            if j == 3:
                break
plt.show()



