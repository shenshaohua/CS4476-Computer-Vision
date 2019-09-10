#!/usr/bin/python3

import numpy as np


def create_Gaussian_kernel(cutoff_frequency):
  """
  Returns a 2D Gaussian kernel using the specified filter size standard
  deviation and cutoff frequency.

  The kernel should have:
  - shape (k, k) where k = cutoff_frequency * 4 + 1
  - mean = floor(k / 2)
  - standard deviation = cutoff_frequency
  - values that sum to 1

  Args:
  - cutoff_frequency: an int controlling how much low frequency to leave in
    the image.
  Returns:
  - kernel: numpy nd-array of shape (k, k)

  HINT:
  - The 2D Gaussian kernel here can be calculated as the outer product of two
    vectors with values populated from evaluating the 1D Gaussian PDF at each
    corrdinate.
  """

  ############################
  ### TODO: YOUR CODE HERE ###
  k = int(cutoff_frequency * 4 + 1)
  mean = np.floor(k/2)
  vector = np.array([])
  for x in range(k):
    vector = np.append(vector, np.exp(-1/(2 * np.square(cutoff_frequency)) * np.square(x - mean))/(np.sqrt(2 * np.pi) * cutoff_frequency))  
  kernel = np.outer(vector,vector)
  kernel = kernel/np.sum(kernel)

  ### END OF STUDENT CODE ####
  ############################

  return kernel

def my_imfilter(image, filter):
  """
  Apply a filter to an image. Return the filtered image.

  Args
  - image: numpy nd-array of shape (m, n, c)
  - filter: numpy nd-array of shape (k, j)
  Returns
  - filtered_image: numpy nd-array of shape (m, n, c)

  HINTS:
  - You may not use any libraries that do the work for you. Using numpy to work
   with matrices is fine and encouraged. Using OpenCV or similar to do the
   filtering for you is not allowed.
  - I encourage you to try implementing this naively first, just be aware that
   it may take an absurdly long time to run. You will need to get a function
   that takes a reasonable amount of time to run so that the TAs can verify
   your code works.
  """

  assert filter.shape[0] % 2 == 1
  assert filter.shape[1] % 2 == 1

  ############################
  ### TODO: YOUR CODE HERE ###
  height = len(image)
  color = len(image[0][0])
  width = len(image[0])

  fHeight = len(filter)
  fWidth = len(filter[0])

  filtered_image = np.zeros((height, width, color))
  fw = int((fWidth - 1) / 2)
  fh = int((fHeight - 1) / 2)
  pw = ((fh, fh), (fw, fw), (0, 0))
  pad = np.pad(image, pw, 'symmetric')

  for x in range(color):
    for y in range(width):
      for z in range(height):
        s = pad[z : z + fHeight, y : y + fWidth, x]
        result = np.multiply(s, filter)
        result = np.sum(result)
        filtered_image[z][y][x] = result
  
  ### END OF STUDENT CODE ####
  ############################

  return filtered_image

def create_hybrid_image(image1, image2, filter):
  """
  Takes two images and a low-pass filter and creates a hybrid image. Returns
  the low frequency content of image1, the high frequency content of image 2,
  and the hybrid image.

  Args
  - image1: numpy nd-array of dim (m, n, c)
  - image2: numpy nd-array of dim (m, n, c)
  - filter: numpy nd-array of dim (x, y)
  Returns
  - low_frequencies: numpy nd-array of shape (m, n, c)
  - high_frequencies: numpy nd-array of shape (m, n, c)
  - hybrid_image: numpy nd-array of shape (m, n, c)

  HINTS:
  - You will use your my_imfilter function in this function.
  - You can get just the high frequency content of an image by removing its low
    frequency content. Think about how to do this in mathematical terms.
  - Don't forget to make sure the pixel values of the hybrid image are between
    0 and 1. This is known as 'clipping'.
  - If you want to use images with different dimensions, you should resize them
    in the notebook code.
  """

  assert image1.shape[0] == image2.shape[0]
  assert image1.shape[1] == image2.shape[1]
  assert image1.shape[2] == image2.shape[2]
  assert filter.shape[0] <= image1.shape[0]
  assert filter.shape[1] <= image1.shape[1]
  assert filter.shape[0] % 2 == 1
  assert filter.shape[1] % 2 == 1

  ############################
  ### TODO: YOUR CODE HERE ###
  f = my_imfilter(image2, filter)
  low_frequencies = my_imfilter(image1, filter)
  high_frequencies = image2 - f
  hybrid_image = np.clip(low_frequencies + high_frequencies, 0, 1)
  ### END OF STUDENT CODE ####
  ############################

  return low_frequencies, high_frequencies, hybrid_image
