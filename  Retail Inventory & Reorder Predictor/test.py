import cv2
import numpy as np
import pandas as pd

# Read image
image = cv2.imread("image.jpg")

# Grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Log transformation
c = 255 / np.log(1 + np.max(gray))
log_image = c * np.log(1 + gray)
log_image = np.uint8(log_image)

# Negative transformation
negative = 255 - gray

# Display all images
cv2.imshow("Original Image", image)
cv2.imshow("Grayscale Image", gray)
cv2.imshow("Log Transformation", log_image)
cv2.imshow("Negative Transformation", negative)

# Pandas example
data = {
    "Transformation": ["Original", "Grayscale", "Log", "Negative"],
    "Height": [
        image.shape[0],
        gray.shape[0],
        log_image.shape[0],
        negative.shape[0]
    ],
    "Width": [
        image.shape[1],
        gray.shape[1],
        log_image.shape[1],
        negative.shape[1]
    ]
}

df = pd.DataFrame(data)
print(df)

cv2.waitKey(0)
cv2.destroyAllWindows()