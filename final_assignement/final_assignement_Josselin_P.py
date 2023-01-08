import time
import numpy as np
from numba import cuda, jit
import cv2
import numba
from math import sqrt


@cuda.jit
def rgb_to_hsv_kernel(input_image, output_image):
    # Get the number of rows and columns in the image
    rows, cols = input_image.shape[:2]

    # Get the current thread's position in the grid
    x, y = numba.cuda.grid(2)

    # Check if the current thread is within the bounds of the image
    if x < rows and y < cols:
        # Get the RGB values of the pixel at (x, y)
        r, g, b = input_image[x, y]

        # Convert the pixel from RGB to HSV
        c_max = max(r, g, b)
        c_min = min(r, g, b)
        c_delta = c_max - c_min

        # Calculate the hue value
        if c_max == r:
            h = 60 * ((g - b) / c_delta % 6)
        elif c_max == g:
            h = 60 * ((b - r) / c_delta + 2)
        elif c_max == b:
            h = 60 * ((r - g) / c_delta + 4)

        # Calculate the saturation value
        s = c_delta / (c_max) if c_max != 0 else 0

        # Calculate the value value
        v = c_max

        # Store the HSV values in the output image
        output_image[x, y, 0] = (h / 255) * 100
        output_image[x, y, 1] = (s / 255) * 100
        output_image[x, y, 2] = (v / 255) * 100


@cuda.jit
def kuwahara_filter_kernel(input_rgb, input_hsv, output, window):

    x = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    y = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y

    row_len = input_rgb.shape[0]
    col_len = input_rgb.shape[1]
    frame = 10

    # if pixel is in frame, then assign its original value
    if x < frame or x > row_len - frame or y < frame or y > col_len - frame:
        output[x, y, 0] = input_rgb[x, y, 0]
        output[x, y, 1] = input_rgb[x, y, 1]
        output[x, y, 2] = input_rgb[x, y, 2]
        return

    radius = window // 2

    # ===================================================================================
    # ===================================================================================
    # SD FORMULA : sqrt(mean((values-mean(values))^2)) (pseudo code)
    #       to compute mean(values), we need to sum every value of each subwindow,
    #       and then divide them by (radius+1)**2 <--> number of values subvalues

    ######################################################################################
    #                                                                   xx xx xx __ __
    # top-left (tl) subwindow :         for x : x - radius -> x+1       xx xx xx __ __
    #                                   for y : y - radius -> y+1       xx xx xx __ __
    #                                                                   __ __ __ __ __
    #                                                                   __ __ __ __ __
    sum = np.float32(0)
    for i in range(x - radius, x + 1):
        for j in range(y - radius, y + 1):
            sum += input_hsv[i, j, 2]
    mean_hsv = sum / (radius + 1) ** 2

    temp_calc = np.float32(0)
    for i in range(x - radius, x + 1):
        for j in range(y - radius, y + 1):
            temp_calc += (input_hsv[i, j, 2] - mean_hsv) ** 2

    std_tl = sqrt(temp_calc / (radius + 1) ** 2)

    ######################################################################################
    #                                                                   __ __ xx xx xx
    # top-right (tr) subwindow :        for x : x - radius -> x + 1     __ __ xx xx xx
    #                                   for y : y -> y + radius + 1     __ __ xx xx xx
    #                                                                   __ __ __ __ __
    #                                                                   __ __ __ __ __
    sum = np.float32(0)
    for i in range(x - radius, x + 1):
        for j in range(y, y + radius + 1):
            sum += input_hsv[i, j, 2]
    mean_hsv = sum / (radius + 1) ** 2

    temp_calc = np.float32(0)
    for i in range(x - radius, x + 1):
        for j in range(y, y + radius + 1):
            temp_calc += (input_hsv[i, j, 2] - mean_hsv) ** 2

    std_tr = sqrt(temp_calc / (radius + 1) ** 2)

    ######################################################################################
    #                                                                   __ __ __ __ __
    # bottom-left (bl) subwindow :      for x : x -> x + radius + 1     __ __ __ __ __
    #                                   for y : y - radius -> y + 1     xx xx xx __ __
    #                                                                   xx xx xx __ __
    #                                                                   xx xx xx __ __
    sum = np.float32(0)
    for i in range(x, x + radius + 1):
        for j in range(y - radius, y + 1):
            sum += input_hsv[i, j, 2]
    mean_hsv = sum / (radius + 1) ** 2

    temp_calc = np.float32(0)
    for i in range(x, x + radius + 1):
        for j in range(y - radius, y + 1):
            temp_calc += (input_hsv[i, j, 2] - mean_hsv) ** 2

    std_bl = sqrt(temp_calc / (radius + 1) ** 2)

    ######################################################################################
    #                                                                   __ __ __ __ __
    # bottom-right (br) subwindow :     for x : x -> x + radius + 1     __ __ __ __ __
    #                                   for y : y -> y + radius + 1     __ __ xx xx xx
    #                                                                   __ __ xx xx xx
    #                                                                   __ __ xx xx xx
    count = 0.0
    sum = np.float32(0)
    for i in range(x, x + radius + 1):
        for j in range(y, y + radius + 1):
            sum += input_hsv[i, j, 2]
            count += 1

    mean_hsv = sum / (radius + 1) ** 2

    temp_calc = 0
    for i in range(x, x + radius + 1):
        for j in range(y, y + radius + 1):
            temp_calc += (input_hsv[i, j, 2] - mean_hsv) ** 2

    std_br = sqrt(temp_calc / (radius + 1) ** 2)

    # ===================================================================================
    # Using the calculated stds to choose the min and assign new color to output image
    # ===================================================================================
    std_tuple = (std_tl, std_tr, std_bl, std_br)
    min_std_value = min(std_tuple)

    mean_r = 0.0
    mean_g = 0.0
    mean_b = 0.0

    # Determine the range of indices to iterate over based on the min_std_value
    if min_std_value == std_tl:
        x_range = range(x - radius, x + 1)
        y_range = range(y - radius, y + 1)
    elif min_std_value == std_tr:
        x_range = range(x - radius, x + 1)
        y_range = range(y, y + radius + 1)
    elif min_std_value == std_bl:
        x_range = range(x, x + radius + 1)
        y_range = range(y - radius, y + 1)
    elif min_std_value == std_br:
        x_range = range(x, x + radius + 1)
        y_range = range(y, y + radius + 1)

    # Iterate over the specified range of indices and sum the red, green, and blue values
    for i in x_range:
        for j in y_range:
            mean_r += input_rgb[i, j, 0]
            mean_g += input_rgb[i, j, 1]
            mean_b += input_rgb[i, j, 2]

    # Calculate the average red, green, and blue values by dividing the sums by the total number of pixels
    mean_r /= (radius + 1) ** 2
    mean_g /= (radius + 1) ** 2
    mean_b /= (radius + 1) ** 2

    # assigning pixel color to the output picture
    output[x, y, 0] = mean_r
    output[x, y, 1] = mean_g
    output[x, y, 2] = mean_b


# Reading the input image
# img = cv2.imread("house.png")
input_img = cv2.imread("montains.jpg")

# creating arrays "on" gpu using cuda
hsv_img = cuda.device_array((input_img.shape[0], input_img.shape[1], 3), input_img.dtype)
rgb_final_img = cuda.device_array((input_img.shape[0], input_img.shape[1], 3), input_img.dtype)

# defining kernel blocksize, gridsize
block_size = (8, 8)
grid_size = (int(input_img.shape[0] / block_size[0]) + 1, int(input_img.shape[1] / block_size[1]) + 1)

# running the kernel successively
start_time = time.time()
rgb_to_hsv_kernel[grid_size, block_size](input_img, hsv_img)
kuwahara_filter_kernel[grid_size, block_size](input_img, hsv_img, rgb_final_img, 17)
print("Execution time for a 5472 x 3648 siza image :\n--- %s seconds ---" % (time.time() - start_time))

# copying the hsv and final images to make sure they are correct and saving the output as an image
hsvvv = hsv_img.copy_to_host()
imggg = rgb_final_img.copy_to_host()
cv2.imwrite("output_kuwahara.png", imggg)
