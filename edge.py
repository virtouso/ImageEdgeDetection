import cv2
import numpy as np

# Load the image
image = cv2.imread('imgs/1.png', cv2.IMREAD_GRAYSCALE)


# Add Gaussian noise to the image
noise = np.random.normal(0, 25, image.shape)  # Mean = 0, Standard deviation = 25
noisy_image = np.clip(image.astype(np.float64) + noise, 0, 255).astype(np.uint8)


_, blurred = cv2.threshold(noisy_image, 128, 255, cv2.THRESH_BINARY)

#blurred = cv2.fastNlMeansDenoising(noisy_image, None, h=10, templateWindowSize=15, searchWindowSize=21)

# Apply Gaussian blur to reduce noise
#blurred = cv2.GaussianBlur(noisy_image, (3, 3), 0)

# Apply Prewitt operator for horizontal edges
prewitt_horizontal = cv2.filter2D(blurred, -1, np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]))

# Apply Prewitt operator for vertical edges
prewitt_vertical = cv2.filter2D(blurred, -1, np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]]))

# Combine horizontal and vertical edges
prewitt_combined = cv2.addWeighted(cv2.convertScaleAbs(prewitt_horizontal), 0.5,
                                   cv2.convertScaleAbs(prewitt_vertical), 0.5, 0)

# Display the results
cv2.imshow('Original Image', image)
cv2.imshow('Noisy  Image', noisy_image)
cv2.imshow('Blurred Image', blurred)
cv2.imshow('Prewitt Horizontal Edges', prewitt_horizontal)
cv2.imshow('Prewitt Vertical Edges', prewitt_vertical)
cv2.imshow('Prewitt Combined Edges', prewitt_combined)
cv2.waitKey(0)
cv2.destroyAllWindows()