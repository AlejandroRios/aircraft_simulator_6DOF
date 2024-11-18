import numpy as np

# globals.py
g = 9.80665
aircraft = {
    'm': 55788 * g / g,
    'J_O_b': np.array([
        [821466, 0, -178919],
        [0, 3343669, 0],
        [-178919, 0, 4056813]
    ]),
    'rC_b': np.array([0, 0, 0]),
    'b': 32.757,
    'S': 116,
    'c': 3.862,
    'hex': 160,
    'r_pilot_b': np.array([15, 0, 0])
}
