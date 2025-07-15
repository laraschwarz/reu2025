from shapely.geometry import Polygon
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# 1. Load your background image
img = mpimg.imread('stmarc_cover.png')

# 2. Define your polygon in pixel coordinates
west_box = Polygon([(376, 1224), (0, 945), (0, 55), (1020, 479)])
east_box = Polygon([(1911, 845), (1533, 1431), (2561, 1450), (2561, 1210)])
x1, y1 = west_box.exterior.xy
x2, y2 = east_box.exterior.xy

# 3. Plot the image and then overlay the polygon
fig, ax = plt.subplots()
ax.imshow(img)                      # draw image
ax.plot(x1, y1, linewidth=2)          # polygon border (default color=blue)
ax.fill(x1, y1, alpha=0.3)            # translucent fill
ax.plot(x2, y2, linewidth=2)          # polygon border (default color=blue)
ax.fill(x2, y2, alpha=0.3)     
ax.set_xlim(0, img.shape[1])        # match axes to image size
ax.set_ylim(img.shape[0], 0)        # flip Y so (0,0) is top-left
ax.set_axis_off()                   # optional: hide axes
plt.show()
