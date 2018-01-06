from __future__ import division, print_function
import numpy as np
from astropy.table import Table
import sys, os
from astropy import wcs
from astropy.io import fits

t = Table.read('/Users/roz18/git/wise-mask-query/misc/astrom-atlas.fits')
print(len(t))

ra_center, dec_center = np.zeros([2, len(t)])
ra_corners = np.zeros([len(t), 4])
dec_corners = np.zeros([len(t), 4])

for coadd_index in range(len(t)):
# for coadd_index in range(10):

    w = wcs.WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.cd = t['CD'][coadd_index]
    w.wcs.crval = t['CRVAL'][coadd_index]
    w.wcs.crpix = t['CRPIX'][coadd_index]
    w.wcs.lonpole = t['LONGPOLE'][coadd_index]
    w.wcs.latpole = t['LATPOLE'][coadd_index]
    w.wcs.equinox = 2000.
    
    pixcrd = np.array([[1024.5, 1024.5]], dtype=float)
    # Convert pixel coordinates to world coordinates
    world = w.wcs_pix2world(pixcrd, True)

    # print(t['COADD_ID'][coadd_index])
    # print('{:7.3f}   {:7.3f}'.format(*world[0]))
    
    ra_center[coadd_index], dec_center[coadd_index] = world[0]
    
    # RA/Dec of four corners
    pixcrd = np.array([[0.5, 0.5], [2048.5, 0.5], [2048.5, 2048.5], [0.5, 2048.5]], dtype=float)
    world = w.wcs_pix2world(pixcrd, True)
    ra_corners[coadd_index], dec_corners[coadd_index] = world.transpose()

t['ra_center'] = ra_center
t['dec_center'] = dec_center
t['ra_corners'] = ra_corners
t['dec_corners'] = dec_corners

t.write('/Users/roz18/git/wise-mask-query/misc/astrom-atlas_radec_added1.fits')