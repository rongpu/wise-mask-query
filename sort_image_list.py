# Produce sorted WISE image list

from __future__ import division, print_function
import numpy as np
from astropy.table import Table
import matplotlib.pyplot as plt
import sys, os

path_in = '/Users/roz18/git/wise-mask-query/misc/wise_mask_image_list.txt'
path_out = '/Users/roz18/git/wise-mask-query/misc/wise_mask_image_list_sorted.txt'
with open(path_in, 'r') as f:
    lines = np.array(map(str.split, map(str.rstrip, f.readlines())))

# Sort by filename
lines = lines[np.argsort(lines[:, 1])]

with open(path_out, 'w') as f:
    for line in lines:
        f.write('{:8}{}\n'.format(line[0], line[1]))
