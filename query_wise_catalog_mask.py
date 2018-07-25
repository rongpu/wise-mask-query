# Query the WISE catalog-based mask for LRG selection

from __future__ import division, print_function
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import sys, os
from astropy.table import Table
from astropy.coordinates import SkyCoord
from scipy.interpolate import interp1d
from astropy import units as u

sys.path.append(os.path.expanduser("~")+'/git/Python/')
from match_coord import search_around

wise_cat_path = '/Users/roz18/Documents/Data/desi_lrg_selection/wisemask/w1_bright-13.3_trim_dr5_region_matched.fits'
# wise_cat_path = '/global/homes/r/rongpu/mydesi/useful/w1_bright-13.3_trim_dr5_region.fits'

def circular_mask_radii_func(w1_ab):
    '''
    Evaluate the WISE bright star circular mask radius for given W1 magnitude

    Inputs
    ------
    w1_ab: W1 magnitude in AB (array);

    Output
    ------
    radii: mask radii (array)
    '''
    pa, pb, pc = 6.5, 1300, 0.14

    w1_ab = np.array(w1_ab)
    radii = np.zeros(len(w1_ab))
    mask = w1_ab>6.0 # set maximum mask radius at W1==6.0
    if np.sum(mask)>0:
        radii[mask] = pa + pb * 10**(-pc*(w1_ab[mask]))
    if np.sum(~mask)>0:
        radii[~mask] = pa + pb * 10**(-pc*(6.0))

    # mask radius in arcsec
    return radii


def ds_mask_widths_func(w1_ab):
    '''
    Define mask width for diffraction spikes
    
    Inputs
    ------
    w1_ab: W1 magnitude in AB (array);

    Output
    ------
    widths: mask widths (array)
    '''

    w1_ab = np.array(w1_ab)
    widths = np.zeros(len(w1_ab))
    mask = w1_ab<=8.0 # set maximum mask width at W1==8.0
    if np.sum(mask)>0:
        widths[mask] = 25. - (8-8)
    mask = (w1_ab>8.0) & (w1_ab<=13.0)
    if np.sum(mask)>0:
        widths[mask] = 25. - (w1_ab[mask]-8)
    mask = (w1_ab>13.0)
    if np.sum(mask)>0:
        widths[mask] = 0
    return widths


def ds_mask_radii_func(w1_ab):
    '''
    Define mask radii for diffraction spikes
    
    Inputs
    ------
    w1_ab: W1 magnitude in AB (array);

    Output
    ------
    radii: mask radii (array)
    '''

    w1_ab = np.array(w1_ab)
    radii = np.zeros(len(w1_ab))
    mask = w1_ab<=8.0 # set maximum mask radius at W1==8.0
    if np.sum(mask)>0:
        radii[mask] = np.exp(-0.41*8+8.9)
    mask = (w1_ab>8.0) & (w1_ab<=13.0)
    if np.sum(mask)>0:
        radii[mask] = np.exp(-0.41*w1_ab[mask]+8.9)
    mask = (w1_ab>13.0)
    if np.sum(mask)>0:
        radii[mask] = 0
    return radii


def ds_masking_func(d_ra, d_dec, d2d, w1_ab):
    '''
    Masking function for diffraction spikes

    Inputs
    ------
    d_ra, d_dec: (array) the differences in RA and Dec (arcsec); 
    d2d: (array) angular distances (arcsec);
    w1_ab: (array) W1 magnitude in AB;
    
    Output
    ------
    ds_flag: array of mask value; True if masked (contaminated).
    '''
    
    ds_mask_widths = ds_mask_widths_func(w1_ab)
    ds_mask_radii = ds_mask_radii_func(w1_ab)

    mask1 = d_dec > (d_ra - ds_mask_widths/np.sqrt(2))
    mask1 &= d_dec < (d_ra + ds_mask_widths/np.sqrt(2))
    mask2 = d_dec > (-d_ra - ds_mask_widths/np.sqrt(2))
    mask2 &= d_dec < (-d_ra + ds_mask_widths/np.sqrt(2))
        
    ds_flag = (mask1 | mask2) & (d2d<ds_mask_radii)
    
    return ds_flag


def query_catalog_mask(ra, dec, diff_spikes=True, return_diagnostics=False):
    '''
    Catalog-based WISE mask for bright stars of W1_AB > 10.0.

    Input:
    ra, dec: coordinates;
    diff_spikes: apply diffraction spikes masking if True;
    return_diagnostics: return disgnostic information if True;

    Return:
    cat_flag: array of mask value; the location is masked (contaminated) if True.
    '''

    # Load trimmed WISE bright star catalog
    wisecat = Table.read(wise_cat_path)

    w1_ab = np.array(wisecat['W1MPRO']) + 2.7

    # Convert to the ecliptic coordinates
    c_decals = SkyCoord(ra=ra*u.degree, dec=dec*u.degree, frame='icrs')
    temp = c_decals.barycentrictrueecliptic
    decals_lon, decals_lat = np.array(temp.lon), np.array(temp.lat)
    c_wise = SkyCoord(ra=wisecat['RA']*u.degree, dec=wisecat['DEC']*u.degree, frame='icrs')
    temp = c_wise.barycentrictrueecliptic
    wise_lon, wise_lat = np.array(temp.lon), np.array(temp.lat)

    w1_bins = np.arange(-1, 20., 0.5)
    
    # only flagged by the circular mask (True if contaminated):
    circ_flag = np.zeros(len(ra), dtype=bool)
    # flagged by the diffraction spike mask but not the circular mask (True if contaminated):
    ds_flag = np.zeros(len(ra), dtype=bool)
    # flagged in the combined masks (True if contaminated):
    cat_flag = np.zeros(len(ra), dtype=bool)

    # record the magnitude of the star that causes the contamination and distance to it
    w1_source = np.zeros(len(ra), dtype=float)
    d2d_source = np.zeros(len(ra), dtype=float)

    for index in range(len(w1_bins)-1):

        mask_wise = (w1_ab>=w1_bins[index]) & (w1_ab<=w1_bins[index+1])
        print('{:.2f} < W1mag < {:.2f}   {} WISE bright stars'.format(w1_bins[index], w1_bins[index+1], np.sum(mask_wise)))

        if np.sum(mask_wise)==0:
            print()
            continue
    
        # find the maximum mask radius for the magnitude bin        
        if not diff_spikes:
            search_radius = np.max(circular_mask_radii_func(w1_ab[mask_wise]))
        else:
            search_radius = np.max([circular_mask_radii_func(w1_ab[mask_wise]), ds_mask_radii_func(w1_ab[mask_wise])])

        # Find all pairs within the search radius
        idx_wise, idx_decals, d2d, d_ra, d_dec = search_around(wise_lon[mask_wise], wise_lat[mask_wise], decals_lon, decals_lat, search_radius=search_radius)

        # circular mask
        mask_radii = circular_mask_radii_func(w1_ab[mask_wise][idx_wise])
        # True means contaminated:
        circ_contam = d2d < mask_radii
        circ_flag[idx_decals[circ_contam]] = True

        w1_source[idx_decals[circ_contam]] = w1_ab[mask_wise][idx_wise[circ_contam]]
        d2d_source[idx_decals[circ_contam]] = d2d[circ_contam]

        if diff_spikes:

            ds_contam = ds_masking_func(d_ra, d_dec, d2d, w1_ab[mask_wise][idx_wise])
            ds_flag[idx_decals[ds_contam]] = True

            # combine the two masks
            cat_flag[idx_decals[circ_contam | ds_contam]] = True

            w1_source[idx_decals[ds_contam]] = w1_ab[mask_wise][idx_wise[ds_contam]]
            d2d_source[idx_decals[ds_contam]] = d2d[ds_contam]

            print('{} objects masked by circular mask'.format(np.sum(circ_flag)))
            print('{} additionally objects masked by diffraction spikes mask'.format(np.sum(cat_flag)-np.sum(circ_flag)))
            print('{} objects masked by the combined masks'.format(np.sum(cat_flag)))
            print()

        else:

            print('{} objects masked'.format(np.sum(circ_flag)))
            print()

    if not diff_spikes:
        cat_flag = circ_flag

    if not return_diagnostics:
        return cat_flag
    else:
        # package all the extra info
        more_info = {}
        more_info['w1_source'] = w1_source
        more_info['d2d_source'] = d2d_source
        more_info['circ_flag'] = circ_flag
        more_info['ds_flag'] = ds_flag

        return cat_flag, more_info

