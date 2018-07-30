# Query the WISE catalog-based mask for LRG selection
# Two sets of mask values are computed separately: 
# Bright stars of W1_AB > 10 and W1_AB < 10

from __future__ import division, print_function
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import sys, os
from astropy.table import Table
from astropy.coordinates import SkyCoord
from scipy.interpolate import interp1d

sys.path.append(os.path.expanduser("~")+'/git/Python/user_modules')
import match_coord

wise_cat_path = '/Users/roz18/Documents/Data/desi_lrg_selection/wisemask/w1_bright-13.3_trim_dr5_region.fits'
# wise_cat_path = '/global/homes/r/rongpu/mydesi/useful/w1_bright-13.3_trim_dr5_region.fits'

def catalog_mask_function1(d2d, w1_ab):
    '''
    Evaluate the distance-magnitude cut for bright stars of W1_AB > 10.0.

    w1_ab: W1 magnitude in AB;
    d2d: distance in arcmin;    
    '''
    
    # parameters
    a = 2.13
    b = 8
    c = -0.08
    
    # True corresponds to contaminated
    flag = np.zeros(len(d2d), dtype=bool)

    # mask = w1_ab>10.
    # flag[mask] = d2d[mask] < (a/(w1_ab[mask]-b) + c)
    # # Bright stars with W1_AB<10.0 contaminates everything within 
    # # the search radius
    # flag[~mask] = True

    flag = d2d < (a / (w1_ab - b) + c)
    
    return flag


def query_catalog_mask1(ra, dec):
    '''
    Catalog-based WISE mask for bright stars of W1_AB > 10.0.

    Input:
    ra, dec: coordinates;

    Return:
    cat_flag: array of mask value; the location is masked (contaminated) if True.
    '''

    # Maximum mask radius
    search_radius = 60. # 1 arcmin

    # Load trimmed WISE bright star catalog
    wisecat = Table.read(wise_cat_path)

    w1_ab = np.array(wisecat['W1MPRO']) + 2.7

    # Sometimes an object is contaminated by a very bright star but its closest match is a 
    # much fainter star, and a simple closest much would fail to mask the object.
    # To avoid this, match to WISE stars in magnitude bins:

    w1_bins = np.arange(10., 20., 0.5)
    cat_flag = np.zeros(len(ra), dtype=bool)

    for index in range(len(w1_bins)-1):

        mask_wise = (w1_ab>=w1_bins[index]) & (w1_ab<=w1_bins[index+1])
        print(w1_bins[index], w1_bins[index+1])
        
        if np.sum(mask_wise)!=0:
        
            # Match every LRG target to three(3) nearby bright stars
            ra2 = np.array(wisecat['RA'][mask_wise])
            dec2 = np.array(wisecat['DEC'][mask_wise])

            # 1st nearest bright star
            idx2_1, idx1_1, d2d_1, _, _ = match_coord.match_coord(ra2, dec2, ra, dec, search_radius=search_radius, 
                                            nthneighbor=1, keep_all_pairs=True, plot_q=False, verbose=False)
            # 2nd nearest bright star
            idx2_2, idx1_2, d2d_2, _, _ = match_coord.match_coord(ra2, dec2, ra, dec, search_radius=search_radius, 
                                            nthneighbor=2, keep_all_pairs=True, plot_q=False, verbose=False)
            # # 3rd nearest bright star
            # idx2_3, idx1_3, d2d_3, _, _ = match_coord.match_coord(ra2, dec2, ra, dec, search_radius=search_radius, 
            #                                 nthneighbor=3, keep_all_pairs=True, plot_q=False, verbose=False)

            # Find the 3 nearest bright stars
            cat_flag[idx1_1] = cat_flag[idx1_1] | catalog_mask_function1(d2d_1*60., w1_ab[mask_wise][idx2_1])
            print('+1st nearest neighbor:', np.sum(cat_flag))
            nflag = np.sum(cat_flag)
            cat_flag[idx1_2] = cat_flag[idx1_2] | catalog_mask_function1(d2d_2*60., w1_ab[mask_wise][idx2_2])
            print('+2nd nearest neighbor:', np.sum(cat_flag)-nflag)
            # nflag = np.sum(cat_flag)
            # cat_flag[idx1_3] = cat_flag[idx1_3] | catalog_mask_function1(d2d_3*60., w1_ab[mask_wise][idx2_3])
            # print('+3rd nearest neighbor:', np.sum(cat_flag)-nflag)




    return cat_flag


def catalog_mask_function2(d2d, w1_ab):
    '''
    Evaluate the distance-magnitude cut for bright stars of W1_AB < 10.0.

    w1_ab: W1 magnitude in AB;
    d2d: distance in arcmin;    
    '''
    
    # Anchor points of the cut
    p1 = (0, 10)
    p2 = (6, 3)
    p3 = (8, 1.5)
    p4 = (10, 1.2)

    interp_func = interp1d([p1[0], p2[0], p3[0], p4[0]], [p1[1], p2[1], p3[1], p4[1]])

    # True corresponds to contaminated
    flag = np.zeros(len(d2d), dtype=bool)

    flag = d2d < interp_func(w1_ab)
    
    return flag


def query_catalog_mask2(ra, dec):
    '''
    Catalog-based WISE mask for bright stars of W1_AB < 10.0.

    Input:
    ra, dec: coordinates;

    Return:
    cat_flag: array of mask value; the location is masked (contaminated) if True.
    '''

    # Maximum mask radius
    search_radius = 600. # 10 arcmin

    # Load trimmed WISE bright star catalog
    wisecat = Table.read(wise_cat_path)

    w1_ab = np.array(wisecat['W1MPRO']) + 2.7

    # Sometimes an object is contaminated by a very bright star but its closest match is a 
    # much fainter star, and a simple closest much would fail to mask the object.
    # To avoid this, match to WISE stars in magnitude bins:

    w1_bins = np.arange(0., 10.5, 0.5)
    if w1_bins[-1]!=10.0:
        raise ValueError('W1 magnitude bins does not have the correct maximum magnitude')

    cat_flag = np.zeros(len(ra), dtype=bool)

    for index in range(len(w1_bins)-1):

        mask_wise = (w1_ab>=w1_bins[index]) & (w1_ab<=w1_bins[index+1])
        print(w1_bins[index], w1_bins[index+1])
        
        if np.sum(mask_wise)!=0:
        
            # Match every LRG target to three(3) nearby bright stars
            ra2 = np.array(wisecat['RA'][mask_wise])
            dec2 = np.array(wisecat['DEC'][mask_wise])

            # 1st nearest bright star
            idx2_1, idx1_1, d2d_1, _, _ = match_coord.match_coord(ra2, dec2, ra, dec, search_radius=search_radius, 
                                            nthneighbor=1, keep_all_pairs=True, plot_q=False, verbose=False)
            # 2nd nearest bright star
            idx2_2, idx1_2, d2d_2, _, _ = match_coord.match_coord(ra2, dec2, ra, dec, search_radius=search_radius, 
                                            nthneighbor=2, keep_all_pairs=True, plot_q=False, verbose=False)
            # # 3rd nearest bright star
            # idx2_3, idx1_3, d2d_3, _, _ = match_coord.match_coord(ra2, dec2, ra, dec, search_radius=search_radius, 
            #                                 nthneighbor=3, keep_all_pairs=True, plot_q=False, verbose=False)

            # Find the 3 nearest bright stars
            cat_flag[idx1_1] = cat_flag[idx1_1] | catalog_mask_function2(d2d_1*60., w1_ab[mask_wise][idx2_1])
            print('+1st nearest neighbor:', np.sum(cat_flag))
            nflag = np.sum(cat_flag)
            cat_flag[idx1_2] = cat_flag[idx1_2] | catalog_mask_function2(d2d_2*60., w1_ab[mask_wise][idx2_2])
            print('+2nd nearest neighbor:', np.sum(cat_flag)-nflag)
            # nflag = np.sum(cat_flag)
            # cat_flag[idx1_3] = cat_flag[idx1_3] | catalog_mask_function2(d2d_3*60., w1_ab[mask_wise][idx2_3])
            # print('+3rd nearest neighbor:', np.sum(cat_flag)-nflag)

    return cat_flag