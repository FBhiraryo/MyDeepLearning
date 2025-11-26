import cv2
import matplotlib.pyplot as plt
import numpy as np
from math import pi

img1 = cv2.imread("figs/velagit pull origin main.png")
h1, w1 = img1.shape[:2]

# --- 余白を追加（上下左右5.3%ずつ） ---
margin = 0.053
pad_h = int(h1 * margin)
pad_w = int(w1 * margin)
img1_pad = cv2.copyMakeBorder(img1, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_CONSTANT, value=0)

# 新しいサイズ
h1p, w1p = img1_pad.shape[:2]
cx = w1p // 2
cy = h1p // 2

# --- モルワイデ→正距円筒図法変換 ---
w2 = w1p
h2 = w1p // 2

x_eq = np.linspace(0, w2 - 1, w2)
y_eq = np.linspace(0, h2 - 1, h2)
X_eq, Y_eq = np.meshgrid(x_eq, y_eq)

lambda_ = (X_eq / w2) * 2 * pi - pi
phi = (0.5 - Y_eq / h2) * pi

def theta_from_phi(phi):
    theta = phi.copy()
    for _ in range(5):
        theta = theta - (2 * theta + np.sin(2 * theta) - pi * np.sin(phi)) / (2 + 2 * np.cos(2 * theta))
    return theta

theta = theta_from_phi(phi)
a = w1p / (2 * pi)
x_moll = 2 * np.sqrt(2) * a * lambda_ * np.cos(theta) / pi + cx
y_moll = -a * np.sqrt(2) * np.sin(theta) + cy

map_x = x_moll.astype(np.float32)
map_y = y_moll.astype(np.float32)

img_warp = cv2.remap(img1_pad, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)

plt.figure(figsize=(w2/100, h2/100), dpi=100)
plt.imshow(cv2.cvtColor(img_warp, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()

