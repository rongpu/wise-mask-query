# Query the WISE catalog-based mask for LRG selection

from __future__ import division, print_function
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import sys, os
from astropy.table import Table
from astropy.coordinates import SkyCoord

sys.path.append('/global/homes/r/rongpu/git/Python/user_modules/')
import match_coord

# Maximum mask radius
search_radius = 60. # 1 arcmin

# Other parameters
a = 2.13
b = 8
c = -0.08

def catalog_mask(d2d, w1_ab):
    '''
    w1_ab: W1 magnitude in AB;
    d2d: distance in arcmin;

    Return:
    
    '''
    
    # True corresponds to contaminated
    flag = np.zeros(len(d2d), dtype=bool)

    mask = w1_ab>10.
    
    flag[mask] = d2d[mask] < (a/(w1_ab[mask]-b) + c)

    # Bright stars with W1_AB<10.0 contaminates everything within 
    # the search radius
    flag[~mask] = True
    
    return flag

def query_catalog_mask(ra, dec):
    '''
    Input:
    ra, dec: coordinates;

    Return:
    cat_flag: array of mask value; the location is masked (contaminated) if True.
    '''

    # Load trimmed WISE bright star catalog
    wisecat = Table.read('/global/homes/r/rongpu/mydesi/useful/w1_bright-13.3_trim_dr5_region.fits')

    w1_ab = np.array(wisecat['W1MPRO']) + 2.7

    # Match every LRG target to three(3) nearby bright stars
    ra2 = np.array(wisecat['RA'])
    dec2 = np.array(wisecat['DEC'])

    # 1st nearest bright star
    idx2_1, idx1_1, d2d_1, _, _ = match_coord.match_coord(ra2, dec2, ra, dec, search_radius=search_radius, 
                                    nthneighbor=1, keep_all_pairs=True, plot_q=False, verbose=False)
    # 2nd nearest bright star
    idx2_2, idx1_2, d2d_2, _, _ = match_coord.match_coord(ra2, dec2, ra, dec, search_radius=search_radius, 
                                    nthneighbor=2, keep_all_pairs=True, plot_q=False, verbose=False)
    # 3rd nearest bright star
    idx2_3, idx1_3, d2d_3, _, _ = match_coord.match_coord(ra2, dec2, ra, dec, search_radius=search_radius, 
                                    nthneighbor=3, keep_all_pairs=True, plot_q=False, verbose=False)

    cat_flag = np.zeros(len(ra), dtype=bool)

    cat_flag[idx1_1] = cat_flag[idx1_1] | catalog_mask(d2d_1*60., w1_ab[idx2_1])
    print('+1st nearest neighbor:', np.sum(cat_flag))
    cat_flag[idx1_2] = cat_flag[idx1_2] | catalog_mask(d2d_2*60., w1_ab[idx2_2])
    print('+2nd nearest neighbor:', np.sum(cat_flag))
    cat_flag[idx1_3] = cat_flag[idx1_3] | catalog_mask(d2d_3*60., w1_ab[idx2_3])
    print('+3rd nearest neighbor:', np.sum(cat_flag))

    return cat_flag