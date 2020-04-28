import numpy as np
import tensorflow as tf

import sys


ckpt = tf.train.NewCheckpointReader(sys.argv[1])
var_to_shape_map = ckpt.get_variable_to_shape_map()

total = 0
for k in sorted(var_to_shape_map.keys()):
    v = var_to_shape_map[k]
    print(k, v)
    total += np.prod(v)

print("Total: ", total)
