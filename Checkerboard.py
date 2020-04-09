"""Script which generates a RGB picture of chessboard, using python libraries - NumPy and Matplotlib."""

import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy

# Size of the chessboard.
height = 8
width = 8

# Creating a chessboard as a numpy array filled with 0.
chessboard = np.zeros((width*100, height*100)).astype(np.uint8)

# Creating a white tiles on the chessboard
for i in range(0, width, 2):
    for j in range(0, height, 2):
        chessboard[0 + i * 100:i * 100 + 100:1, 0 + j * 100:j * 100 + 100:1] = 255
        chessboard[100 + i * 100:i * 100 + 200:1, 100 + j * 100:j * 100 + 200:1] = 255

# Creating a copy of a chessboard array, for every RGB color
red_channel = deepcopy(chessboard)
green_channel = deepcopy(chessboard)
blue_channel = deepcopy(chessboard)

# Creating a chessboard with a RGB colors
chessboard = np.array([red_channel, green_channel, blue_channel]).astype(np.uint8)

# Plotting classical chessboard using matplotlib.
_ = plt.imshow(np.moveaxis(chessboard, 0, 2))
plt.show()

# Changing the red channel values to 0
red_channel[::] = 0

# Creating a modified chessboard
modified_chessboard = np.array([red_channel, green_channel, blue_channel]).astype(np.uint8)

# Plotting modified chessboard
_ = plt.imshow(np.moveaxis(modified_chessboard, 0, 2))
plt.show()
