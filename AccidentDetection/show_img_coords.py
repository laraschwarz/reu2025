import cv2
import sys

if len(sys.argv) < 2:
    print("Usage: python show_img_coords.py <image_path>")
    exit()

img_path = sys.argv[1]
img = cv2.imread(img_path)
if img is None:
    print(f"Error: Could not open {img_path}")
    exit()

def show_xy(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        temp = img.copy()
        cv2.putText(temp, f"({x}, {y})", (x+10, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        cv2.imshow("Image", temp)

cv2.namedWindow("Image")
cv2.setMouseCallback("Image", show_xy)

while True:
    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
        break

cv2.destroyAllWindows()