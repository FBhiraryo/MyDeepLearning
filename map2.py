import cv2
import matplotlib.pyplot as plt
import numpy as np
from math import pi

img1 = cv2.imread("figs/vela.png")
h1, w1 = img1.shape[:2]
cx = w1 // 2
cy = h1 // 2

# 出力画像サイズ（2:1比率に固定）
w2 = w1
h2 = w1 // 2

# 出力画像の各ピクセル座標
x_eq = np.linspace(0, w2 - 1, w2)
y_eq = np.linspace(0, h2 - 1, h2)
X_eq, Y_eq = np.meshgrid(x_eq, y_eq)

# 経度λ: -π ～ π
lambda_ = (X_eq / w2) * 2 * pi - pi
# 緯度φ: π/2 ～ -π/2（上から下へ）
phi = (0.5 - Y_eq / h2) * pi

# φ, λ からモルワイデ図法のx, y座標を計算
def theta_from_phi(phi):
    theta = phi.copy()
    for _ in range(5):
        theta = theta - (2 * theta + np.sin(2 * theta) - pi * np.sin(phi)) / (2 + 2 * np.cos(2 * theta))
    return theta

theta = theta_from_phi(phi)
a = w1 / (2 * pi)
x_moll = 2 * np.sqrt(2) * a * lambda_ * np.cos(theta) / pi + cx
y_moll = -a * np.sqrt(2) * np.sin(theta) + cy

map_x = x_moll.astype(np.float32)
map_y = y_moll.astype(np.float32)

img_warp = cv2.remap(img1, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)

plt.figure(figsize=(12, 6))
plt.imshow(cv2.cvtColor(img_warp, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()