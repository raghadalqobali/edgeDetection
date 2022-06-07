import numpy as np
import cv2
import matplotlib.pyplot as plt

# read image
image = cv2.imread('cameraman.jpg', cv2.IMREAD_GRAYSCALE)

# display the image
plt.imshow(image, cmap='gray')
plt.show()

# define horizontal and Vertical sobel operators
Sx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
Sy = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])


# define convolution function
# Inputs: img: image, Filter:filter
def convolution(img, filter):
    # height and width of the image
    img_height = img.shape[0]
    img_width = img.shape[1]

    # height and width of the filter
    filter_height = filter.shape[0]
    filter_width = filter.shape[1]

    H = (filter_height - 1) // 2
    W = (filter_width - 1) // 2



    # output image
    out = np.zeros((img_height, img_width))
    # loop over all the pixels of image 
    for i in np.arange(H, img_height - H):
        for j in np.arange(W, img_width - W):
            sum = 0
            # loop over the filter
            for k in np.arange(-H, H + 1):
                for l in np.arange(-W, W + 1):
                    # get the corresponding value from image and filter
                    a = img[i + k, j + l]
                    w = filter[H + k, W + l]
                    sum += (w * a)
            out[i, j] = sum
    return out


# calculate Gx
Gx = convolution(image, Sx)
plt.imshow(Gx, cmap='gray')
plt.show()

# calculate Gy
Gy = convolution(image, Sy)
plt.imshow(Gy, cmap='gray')
plt.show()

# calculate the gradient magnitude G
G = np.sqrt(np.power(Gx, 2) + np.power(Gy, 2))

# Thresholding
ret, G = cv2.threshold(G, 250, 255, cv2.THRESH_BINARY)

# show output image
plt.imshow(G, cmap='gray')
plt.show()
