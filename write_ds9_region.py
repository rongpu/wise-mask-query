from __future__ import division, print_function
import sys, os
import numpy as np

def write_ds9_region(fn, ra, dec):

    # Region file format: DS9 version 4.1
    header = '# Region file format: DS9 version 4.1\nglobal color=green dashlist=8 3 width=1 font="helvetica 10 normal roman" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1\nfk5\n'

    with open(fn, 'w+') as f:
        f.write(header)
        for index in range(len(ra)):
            line = 'point({:.7f},{:.7f}) # point=x\n'.format(ra[index], dec[index])
            f.write(line)