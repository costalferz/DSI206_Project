import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import imutils
from PIL import Image, ImageDraw, ImageFont



image_fn = 'receipt-scanned.jpg'
img = cv2.imread(image_fn)

AngleofRotation = "No change"
Mirror = "No"

if AngleofRotation=="No change":
    imgrot = img
elif AngleofRotation == "90":
    imgrot = cv2.rotate(img,cv2.ROTATE_90_CLOCKWISE)
elif AngleofRotation == "180":
    imgrot = cv2.rotate(img,cv2.ROTATE_180)
elif AngleofRotation == "270":
    imgrot = cv2.rotate(img,cv2.ROTATE_90_COUNTERCLOCKWISE)
if Mirror == "Yes":
    imgrot = cv2.flip(imgrot, 1)

height, width, channels = imgrot.shape
imgrgb = cv2.cvtColor(imgrot, cv2.COLOR_BGR2RGB) 
gray = cv2.cvtColor(imgrot, cv2.COLOR_RGB2GRAY)

ksize = 3 
kernel = np.ones((ksize , ksize), np.uint8) 
erosion = cv2.erode(gray, kernel, iterations=1)
opening = cv2.morphologyEx(erosion, cv2.MORPH_OPEN, kernel)
closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

if ksize > 3:

    ksize = ksize-2
else:
    ksize = 1

blur_img = cv2.GaussianBlur(closing,(ksize,ksize),0)

edges = cv2.Canny(blur_img,30, 250) 
thresh = cv2.adaptiveThreshold(edges, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C , cv2.THRESH_BINARY_INV, 11, 3)  

contours = cv2.findContours(thresh , cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
all_contours = imutils.grab_contours(contours)
all_contours = sorted(all_contours, key=cv2.contourArea, reverse=True)[:1]
contour_perimeter = cv2.arcLength(all_contours[0], True) 
approximated_poly = cv2.approxPolyDP(all_contours[0], 0.02 * contour_perimeter, True)

draw = cv2.drawContours(imgrgb.copy(), [approximated_poly], -1, (0,255,0), 4)

try:
    approximated_poly = np.array(approximated_poly.reshape(4, 2))
    rectangle = np.zeros((4, 2), dtype="float32")
    (tl, tr, br, bl) = rectangle                
    s = np.sum(approximated_poly, axis=1)
    rectangle[0] = approximated_poly[np.argmin(s)]
    rectangle[2] = approximated_poly[np.argmax(s)]

    diff = np.diff(approximated_poly, axis=1)
    rectangle[1] = approximated_poly[np.argmin(diff)]
    rectangle[3] = approximated_poly[np.argmax(diff)]

    # Calculate width
    width_a = np.sqrt((tl[0] - tr[0])**2 + (tl[1] - tr[1])**2 )
    width_b = np.sqrt((bl[0] - br[0])**2 + (bl[1] - br[1])**2 )
    max_width = max(int(width_a), int(width_b))

    # Calculate height
    height_a = np.sqrt((tl[0] - bl[0])**2 + (tl[1] - bl[1])**2 )
    height_b = np.sqrt((tr[0] - br[0])**2 + (tr[1] - br[1])**2 )
    max_height = max(int(height_a), int(height_b))
    destinations = np.array([ 
        [0,0],                                  # Top left point
        [max_width - 1, 0],                     # Top right point
        [max_width - 1, max_height - 1],        # Bottom right point
        [0, max_height - 1]],                   # Bottom left point
        dtype="float32")                        # Data type
    transform_matrix = cv2.getPerspectiveTransform(rectangle, destinations)
    scan = cv2.warpPerspective(imgrgb, transform_matrix, (max_width, max_height))
except ValueError as rectangle:
    print("ไม่สามารถหา ขอบ 4 ด้านของเอกสารได้ ลองเปลี่ยนค่า Kernel อีกครั้งหนึ่งหรือลองใช้รูปเอกสารรูปอื่นดู")


image_scan_watermark = Image.fromarray(scan)

draw_watermark = ImageDraw.Draw(image_scan_watermark)

text = "DSI206"
font = ImageFont.truetype('Kanit-ExtraLight.ttf', 36)
imgw, imgh = image_scan_watermark.size
textwidth, textheight = draw_watermark.textsize(text, font)

# calculate the x,y coordinates of the text
margin = 10
x = imgw - textwidth - margin
y = imgh - textheight - margin

# draw watermark in the bottom right corner
draw_watermark.text((x, y), text, font=font)

#Save Scanned image
Image.fromarray(scan).save('scan.jpg')

#Save watermarked image
image_scan_watermark.save('watermark.jpg')

plt.figure(figsize=(40, 20) ,  dpi=200)

plt.subplot(1,8,1)
plt.axis('off')
plt.imshow(img)
plt.title('Original', fontsize='14')

plt.subplot(1,8,2)
plt.axis('off')
plt.imshow(imgrot)
plt.title('Rotate and Mirror', fontsize='14')

plt.subplot(1,8,3)
plt.axis('off')
plt.imshow(gray,cmap='gray')
plt.title('Covert to Gray image', fontsize='14')

plt.subplot(1,8,4)
plt.axis('off')
plt.imshow(blur_img,cmap='gray')
plt.title('Denoise', fontsize='14')

plt.subplot(1,8,5)
plt.axis('off')
plt.imshow(thresh , cmap='gray')
plt.title('Edges + Adaptive Threshold', fontsize='14')

plt.subplot(1,8,6)
plt.axis('off')
plt.imshow(draw)
plt.title('Draw Contour', fontsize='14')

plt.subplot(1,8,7)
plt.axis('off')
plt.imshow(scan)
plt.title('Scaned', fontsize='14')

plt.subplot(1,8,8)
plt.axis('off')
plt.imshow(image_scan_watermark)
plt.title('Watermark', fontsize='14')

plt.show()
