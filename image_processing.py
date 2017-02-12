import numpy as np
import random
import cv2

def region_of_interest(img):
    height = img.shape[0]
    width = img.shape[1]
    vertices = np.array([[(0, height-15), (0, height/2-10),
                          (width, height/2-10), (width, height-15)]],
                        dtype=np.int32)
    #defining a blank mask to start with
    mask = np.zeros_like(img)
    channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
    ignore_mask_color = (255,) * channel_count
    #filling pixels inside the polygon defined by \"vertices\" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    #returning the image only where mask pixels are nonzero\n",
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def rezise(img):
    return cv2.resize(img, (75,48))

def change_brightness(image):
    # Randomly select a percent change
    change_pct = random.uniform(0.3, 1.0)
    
    # Change to HSV to change the brightness V
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    hsv[:,:,2] = hsv[:,:,2] * change_pct
    
    #Convert back to RGB 
    img_brightness = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return img_brightness
	
def preprocess_image(img):
    img = region_of_interest(img)
    img = rezise(img)
    return img

