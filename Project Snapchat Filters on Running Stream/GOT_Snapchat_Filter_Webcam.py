
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt


eyes_cascade = cv2.CascadeClassifier("Train/third-party/frontalEyes35x16.xml")
nose_cascade = cv2.CascadeClassifier("Train/third-party/Nose18x15.xml")


spec_img = cv2.imread("Train/glasses.png", -1)
print("Spectacles Image Shape:", spec_img.shape)
mus_img =  cv2.imread("Train/mustache.png", -1)
print("Mustache Image Shape:", mus_img.shape)

cap = cv2.VideoCapture(0)

def overlay_image_alpha(img, img_overlay, pos, alpha_mask):
    """Overlay img_overlay on top of img at the position specified by
    pos and blend using alpha_mask.

    Alpha mask must contain values within the range [0, 1] and be the
    same size as img_overlay.
    """

    x, y = pos

    # Image ranges
    y1, y2 = max(0, y), min(img.shape[0], y + img_overlay.shape[0])
    x1, x2 = max(0, x), min(img.shape[1], x + img_overlay.shape[1])

    # Overlay ranges
    y1o, y2o = max(0, -y), min(img_overlay.shape[0], img.shape[0] - y)
    x1o, x2o = max(0, -x), min(img_overlay.shape[1], img.shape[1] - x)

    # Exit if nothing to do
    if y1 >= y2 or x1 >= x2 or y1o >= y2o or x1o >= x2o:
        return

    channels = img.shape[2]

    alpha = alpha_mask[y1o:y2o, x1o:x2o]
    alpha_inv = 1.0 - alpha

    for c in range(channels):
        img[y1:y2, x1:x2, c] = (alpha * img_overlay[y1o:y2o, x1o:x2o, c] +
                                alpha_inv * img[y1:y2, x1:x2, c])



ret,frame = cap.read()
eyes  = eyes_cascade.detectMultiScale(frame,1.1,5)
nose  = nose_cascade.detectMultiScale(frame,1.1,5)

while True:

    ret,frame = cap.read()

    if ret == False:
        continue

    eyes  = eyes_cascade.detectMultiScale(frame,1.1,5)

    for (x,y,w,h) in eyes:
        s_img = cv2.resize(spec_img, (w,h), interpolation = cv2.INTER_AREA)
        overlay_image_alpha(frame,s_img[:, :, 0:3],(x-5, y+5),s_img[:, :, 3] / 255.0)

    nose  = nose_cascade.detectMultiScale(frame,1.1,5)
    for (x,y,w,h) in nose:
        s_img = cv2.resize(mus_img, (2*w,h), interpolation = cv2.INTER_AREA)
        w = int(w/2)
        h = int(h/2)
        overlay_image_alpha(frame,s_img[:, :, 0:3],(x-w, y+h),s_img[:, :, 3] / 255.0)

    cv2.imshow("Video Frame",frame)

    key_pressed = cv2.waitKey(1) & 0xFF
    if key_pressed == ord('q'):
        break


    
cap.release()    
cv2.destroyAllWindows()