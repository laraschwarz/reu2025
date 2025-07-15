import cv2

img = cv2.imread('img_path')
h, w = img.shape[:2]

cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
cv2.imshow('Image', img)
cv2.waitKey(1)

# this returns x, y, width, height of the *window* in screen‑points:
_, _, win_w, win_h = cv2.getWindowImageRect('Image')

scale_x = w / win_w
scale_y = h / win_h

def click_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        true_x = int(x * scale_x)
        true_y = int(y * scale_y)
        print(f"pixel coords = {true_x, true_y}")
