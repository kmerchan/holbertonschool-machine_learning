#!/usr/bin/env python3
""" plots y as a line graph """
import numpy as np
import matplotlib.pyplot as plt

y = np.arange(0, 11) ** 3

plt.plot([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], y, 'r-')
plt.show()
