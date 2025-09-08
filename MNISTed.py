import time
import numpy as np
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import RMSprop
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# MNISTデータのロード
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 保存したモデルを読み込む
loaded_model = load_model('mnist_model.keras')

# ランダムに5枚選ぶ
np.random.seed(42)  # 再現性のため
indices = np.random.choice(len(x_test), 5, replace=False)

for i, idx in enumerate(indices):
    img = x_test[idx]
    img_input = img.reshape(1, 28, 28, 1)
    predicted_label = loaded_model.predict(img_input)
    predicted_class = predicted_label.argmax()
    plt.subplot(1, 5, i+1)
    plt.imshow(img, cmap='gray')
    plt.title(f"P:{predicted_class}\nT:{y_test[idx]}")
    plt.axis('off')

plt.tight_layout()
plt.show()