# =====================================================
# Загрузка и распаковка датасета
# =====================================================

import os
import zipfile
import random
import numpy as np
import pandas as pd

from PIL import Image

import matplotlib.pyplot as plt
from google.colab import files

print("Загрузите архив с датасетом")
uploaded = files.upload()

zip_filename = next(iter(uploaded.keys()))

EXTRACT_PATH = "/content/flowers_data"
os.makedirs(EXTRACT_PATH, exist_ok=True)

with zipfile.ZipFile(zip_filename, "r") as zip_ref:
    zip_ref.extractall(EXTRACT_PATH)

print("Архив распакован")


# =====================================================
# Поиск рабочей папки
# =====================================================

def find_dataset_root(root):

    dirs = [d for d in os.listdir(root)
            if os.path.isdir(os.path.join(root, d))]

    for d in dirs:

        path = os.path.join(root, d)

        subdirs = [
            x for x in os.listdir(path)
            if os.path.isdir(os.path.join(path, x))
        ]

        if len(subdirs) >= 3:
            return path

    return root


DATA_PATH = find_dataset_root(EXTRACT_PATH)

print("Рабочая папка:", DATA_PATH)

classes = sorted([
    d for d in os.listdir(DATA_PATH)
    if os.path.isdir(os.path.join(DATA_PATH, d))
])

print("Классы:", classes)


# =====================================================
# Сбор информации о датасете
# =====================================================

widths = []
heights = []
brightness = []

class_counts = {}

image_paths = []

for class_name in classes:

    class_dir = os.path.join(DATA_PATH, class_name)

    files_list = [
        f for f in os.listdir(class_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    class_counts[class_name] = len(files_list)

    for file_name in files_list:

        path = os.path.join(class_dir, file_name)

        image_paths.append((class_name, path))

        try:

            img = Image.open(path).convert("RGB")

            width, height = img.size

            widths.append(width)
            heights.append(height)

            img_np = np.array(img)

            brightness.append(img_np.mean())

        except:
            pass


print("Всего изображений:", len(widths))


# =====================================================
# Таблица 1
# Количество изображений по классам
# =====================================================

table_classes = pd.DataFrame({
    "Класс": list(class_counts.keys()),
    "Количество": list(class_counts.values())
})

print("\nКоличество изображений по классам:")
display(table_classes)


# =====================================================
# Рисунок 1
# Распределение классов
# =====================================================

plt.figure(figsize=(7,5))

plt.bar(
    class_counts.keys(),
    class_counts.values()
)

plt.title("Распределение изображений по классам")
plt.xlabel("Класс")
plt.ylabel("Количество изображений")

plt.tight_layout()

plt.show()


# =====================================================
# Рисунок 2
# Примеры изображений
# =====================================================

fig, axes = plt.subplots(
    len(classes),
    2,
    figsize=(8, 10)
)

for row, class_name in enumerate(classes):

    class_images = [
        p for c, p in image_paths
        if c == class_name
    ]

    samples = random.sample(class_images, 2)

    for col in range(2):

        img = Image.open(samples[col])

        axes[row, col].imshow(img)

        axes[row, col].set_title(class_name)

        axes[row, col].axis("off")

plt.tight_layout()

plt.show()


# =====================================================
# Таблица 2
# Размеры изображений
# =====================================================

stats_df = pd.DataFrame({
    "Параметр": ["Ширина", "Высота"],
    "Минимум": [
        min(widths),
        min(heights)
    ],
    "Максимум": [
        max(widths),
        max(heights)
    ],
    "Среднее": [
        round(np.mean(widths), 2),
        round(np.mean(heights), 2)
    ]
})

print("\nСтатистика размеров изображений:")
display(stats_df)


# =====================================================
# Рисунок 3
# Ширина изображений
# =====================================================

plt.figure(figsize=(8,5))

plt.hist(widths, bins=20)

plt.title("Распределение ширины изображений")

plt.xlabel("Ширина (px)")
plt.ylabel("Количество")

plt.grid(True)

plt.show()


# =====================================================
# Рисунок 4
# Высота изображений
# =====================================================

plt.figure(figsize=(8,5))

plt.hist(heights, bins=20)

plt.title("Распределение высоты изображений")

plt.xlabel("Высота (px)")
plt.ylabel("Количество")

plt.grid(True)

plt.show()


# =====================================================
# Рисунок 5
# Средняя яркость
# =====================================================

plt.figure(figsize=(8,5))

plt.hist(brightness, bins=20)

plt.title("Распределение средней яркости изображений")

plt.xlabel("Средняя яркость")
plt.ylabel("Количество")

plt.grid(True)

plt.show()


# =====================================================
# Дополнительная проверка разметки
# =====================================================

print("\nИнформация о датасете")

print(f"Всего изображений: {len(widths)}")

for cls, count in class_counts.items():
    print(f"{cls}: {count}")