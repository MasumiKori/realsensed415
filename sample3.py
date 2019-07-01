import cv2
import numpy as np
import matplotlib.pyplot as plt

def img4show(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

src = cv2.imread("data/use.jpg")
srcHeight, srcWidth = src.shape[:2]

fovw = 69.4
fovh = 42.5


#出力画像サイズ
dstWidth, dstHeight = 640, 360

#視線方向（正距円筒座標で指定）
theta = np.radians(-117)
phi = np.radians(-3)

fov_radw = np.radians(fovw)
fov_radh = np.radians(fovh)
fx = dstWidth / (2*np.tan(fov_radw/2))
fy = dstHeight / (2*np.tan(fov_radh/2))
cameraMatrix = np.array([
    [fx, 0, dstWidth/2],
    [0, fy, dstHeight/2],
    [0, 0, 1]
])

rotX_phi = np.array([
    [1, 0, 0],
    [0, np.cos(phi), -np.sin(phi)],
    [0, np.sin(phi), np.cos(phi)]
])

rotY_theta = np.array([
    [np.cos(theta), 0, np.sin(theta)],
    [0, 1, 0],
    [-np.sin(theta), 0, np.cos(theta)]
])

rot = rotY_theta @ rotX_phi

# 移り先の画像座標（透視投影画像）
dstCoord = np.meshgrid(np.arange(dstWidth), np.arange(dstHeight))
dstCoord = np.c_[dstCoord[0].reshape(-1, 1),
                       dstCoord[1].reshape(-1, 1),
                       np.ones(dstWidth * dstHeight)].T

# 正規化画像座標
normalizedDstCoord = np.linalg.inv(cameraMatrix)@dstCoord

# 球面座標
sphereCoord = normalizedDstCoord / np.linalg.norm(normalizedDstCoord, axis=0)
sphereCoord = rot @ sphereCoord

# 正距円筒座標
equiCoord = np.c_[
    np.arctan2(sphereCoord[0], sphereCoord[2]),
    -np.arcsin(sphereCoord[1])
].T

# 元の画像座標 (正距円筒画像)
srcCoord = np.c_[
    srcWidth * (equiCoord[0] / (2 * np.pi)) + srcWidth / 2,
    -srcHeight * (equiCoord[1] / np.pi) + srcHeight / 2
].T

mapX = srcCoord[0].reshape(dstHeight, dstWidth).astype(np.float32)
mapY = srcCoord[1].reshape(dstHeight, dstWidth).astype(np.float32)

# 元の画像から画素情報を移す
dst = cv2.remap(src, mapX, mapY, cv2.INTER_LINEAR)

plt.figure(figsize=(8, 6))
cv2.imwrite('per.jpg',dst)
#plt.imshow(img4show(dst))
plt.show()