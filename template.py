import cv2
import numpy as np

img = cv2.imread("data/color_Color.png",0)
temp = cv2.imread("data/per.jpg",0)

result = cv2.matchTemplate(img,temp,cv2.TM_CCOEFF_NORMED)
min_val,max_val,min_loc,max_loc = cv2.minMaxLoc(result)
top_left = max_loc
w,h = temp.shape[::-1]
bottom_right = (top_left[0] + w, top_left[1] + h)
#検出領域を四角で囲んで保存
result = cv2.imread("data/color_Color.png")
result1 = result[top_left[1]:bottom_right[1],top_left[0]:bottom_right[0]]
result2 = cv2.imread("data/depth_Depth.png")
result3 = result2[top_left[1]:bottom_right[1],top_left[0]:bottom_right[0]]
cv2.imwrite("data/result.png", result1)
cv2.imwrite("data/result_depth.png",result3)
