import os
import cv2
import math
import glob
import numpy as np
import pyrealsense2 as rs
import matplotlib.pyplot as plt

# img1 = cv2.imread('data/FLIR0221.jpg')
img2 = cv2.imread('data/snapshot12_Color.png')  #realsense 16:9   点群　4:3
img1 = cv2.imread('data/test22.jpg')           #全天球
yy,xx,cc=img2.shape

# img1 = color_rgb
# img2 = dst_change

# A-KAZE検出器の生成
detector = cv2.AKAZE_create()

# 特徴量の検出と特徴量ベクトルの計算
kp1, des1 = detector.detectAndCompute(img1, None)
kp2, des2 = detector.detectAndCompute(img2, None)

# Brute-Force Matcherの生成
bf = cv2.BFMatcher()

# 特徴量ベクトル同士をBrute-Force＆KNNでマッチング
matches = bf.knnMatch(des1, des2, k=2)

# データを間引く
ratio = 0.2
good = []
for m, n in matches:
    if m.distance < ratio * n.distance:
        good.append([m])

# 特徴量をマッチング状況に応じてソート
good = sorted(matches, key = lambda x : x[1].distance)

# 対応する特徴点同士を描画
img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good[:2], None, flags=2)

fig = plt.figure(figsize=(16,9))
plt.imshow(img3)
plt.show

q_kp = []
t_kp = []
index1_0 = []
index1_1 = []
index2_0 = []
index2_1 = []
for p in good[:2]:
    for px in p:
        q_kp.append(kp1[px.queryIdx])
        t_kp.append(kp2[px.trainIdx])

q_x1, q_y1 = q_kp[0].pt     #全天球
q_x2, q_y2 = q_kp[-1].pt
q_deg = math.atan2(q_y2 - q_y1, q_x2 - q_x1) * 180 / math.pi
q_len = math.sqrt((q_x2 - q_x1) ** 2 + (q_y2 - q_y1) ** 2)

t_x1, t_y1 = t_kp[0].pt     #realsense
t_x2, t_y2 = t_kp[-1].pt

t_deg = math.atan2(t_y2 - t_y1, t_x2 - t_x1) * 180 / math.pi
t_len = math.sqrt((t_x2 - t_x1) ** 2 + (t_y2 - t_y1) ** 2)

print(t_len/q_len)
print(q_deg,t_deg)

x, y, c = img1.shape
size = (x, y)

# 回転の中心位置
center = (q_x1, q_y1)

# 回転角度
# angle = q_deg - t_deg
angle = 0

# サイズ比率
scale = 1.0

# 回転変換行列の算出
# rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)

# アフィン変換
# img_rot = cv2.warpAffine(img1, rotation_matrix, size, flags=cv2.INTER_CUBIC)

"""
#img2のサイズに縮小
#reduction_ratio = 0.7
reduction_ratio = t_len/q_len

q1x = int(q_x1*reduction_ratio)
q1y = int(q_y1*reduction_ratio)
q2x = int(q_x2*reduction_ratio)
q2y = int(q_y2*reduction_ratio)

t1x = int(t_x1)
t1y = int(t_y1)
t2x = int(t_x2)
t2y = int(t_y2)


img_re = cv2.resize(img1,((int(y*reduction_ratio)),int(x*reduction_ratio)))

height, width, channels = img2.shape
yy=int(q1y-t1y)
xx=int(q1x-t1x)
img_dst = img_re[yy : yy+height, xx : xx+width]
"""

#img1のサイズに拡大
reduction_ratio = q_len/t_len
print(reduction_ratio)
q1x = int(q_x1)
q1y = int(q_y1)
q2x = int(q_x2)
q2y = int(q_y2)

t1x = int(t_x1*reduction_ratio)
t1y = int(t_y1*reduction_ratio)
t2x = int(t_x2*reduction_ratio)
t2y = int(t_y2*reduction_ratio)


img_re = img1.copy()

# fig = plt.figure(figsize=(16,9))
# plt.imshow(img_re)
# plt.show()

height, width, channels = img2.shape
yy=int(q1y-t1y)
xx=int(q1x-t1x)
print(height,width)
print(reduction_ratio)
print(yy,xx)
height2 = int(height*reduction_ratio)
width2 = int(width*reduction_ratio)
img_dst = img_re[yy : yy+height2, xx : xx+width2]

print("Y_top:",yy)
print("Y_under:",yy+height2)
print("X_left:",xx)
print("X_right:",xx+width2)


img_dst2 = cv2.cvtColor(img_dst,cv2.COLOR_BGR2RGB)
img2_cvt = cv2.cvtColor(img2,cv2.COLOR_BGR2RGB)
fig = plt.figure(figsize=(16,9))
plt.subplot(1, 2, 1)
plt.imshow(img_dst2)
plt.subplot(1, 2, 2)
plt.imshow(img2_cvt)
plt.show()
prin
