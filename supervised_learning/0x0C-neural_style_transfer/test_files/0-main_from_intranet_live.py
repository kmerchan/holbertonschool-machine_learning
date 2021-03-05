#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
NST = __import__('0-neural_style_live_code').NST

if __name__ == '__main__':
      np.random.seed(2)
      tf.enable_eager_execution()
      image = np.random.uniform(0, 256, size=(512, 512, 3))
      scaled = NST.scale_image(image)
      print(tf.reduce_min(scaled))
      print(tf.reduce_max(scaled))
