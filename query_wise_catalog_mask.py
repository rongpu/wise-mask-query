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

sys.path.append(os.path.expanduser("~")+'/git/Python/user_modules/')
from match_coord import search_around

wise_cat_path_default = '/Users/roz18/Documents/Data/desi_lrg_selection/wisemask/w1_bright-13.3_trim_dr7_region_matched.fits'
# wise_cat_path = '/global/homes/r/rongpu/mydesi/useful/w1_bright-13.3_trim_dr5_region.fits'


# Define the radius mask of the circular mask
x, y = np.transpose([[4.5, 210.], [5.5, 200.], [6.25, 150.], [6.75, 125.], [7.25, 120.], [7.75, 110.], [8.25, 100.], [8.75,  75.], [9.25,  60.], [9.75,  55.], [ 10.25,  50.], [ 10.75,  48.], [ 11.25,  40.], [ 11.75,  37.], [ 12.25,  25.], [ 12.75,  20.], [ 13.25,  18.], [ 13.75,  16.], [ 14.25,  12.], [ 14.75,  11.], [ 15.25,  11.], [ 15.75,  10.]])
circular_mask_radii_func = interp1d(x, y, bounds_error=False, fill_value=(y[0], y[-1]))

# Define length for diffraction spikes mask
x, y = np.transpose([[4.5, 600.], [5.5, 600.], [6.25, 540.], [6.75, 520.], [7.25, 500.], [7.75, 320.], [8.25, 300.], [8.75, 290.], [9.25, 160.], [9.75, 150.], [ 10.25, 140.], [ 10.75, 130.], [ 11.25, 130.], [ 11.75, 100.], [ 12.25, 60.], [ 12.75, 40.], [ 13., 40.]])
ds_mask_length_func = interp1d(x, y, bounds_error=False, fill_value=(y[0], 0))


# Define width for diffraction spikes mask
x, y = np.transpose([[8., 25.], [13., 16.]])
ds_mask_width_func = interp1d(x, y, bounds_error=False, fill_value=(y[0], 0))


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
    
    ds_mask_widths = ds_mask_width_func(w1_ab)
    ds_mask_length = ds_mask_length_func(w1_ab)

    mask1 = d_dec > (d_ra - ds_mask_widths/np.sqrt(2))
    mask1 &= d_dec < (d_ra + ds_mask_widths/np.sqrt(2))
    mask1 &= (d_dec < -d_ra + ds_mask_length/np.sqrt(2)) & (d_dec > -d_ra - ds_mask_length/np.sqrt(2))

    mask2 = d_dec > (-d_ra - ds_mask_widths/np.sqrt(2))
    mask2 &= d_dec < (-d_ra + ds_mask_widths/np.sqrt(2))
    mask2 &= (d_dec < +d_ra + ds_mask_length/np.sqrt(2)) & (d_dec > +d_ra - ds_mask_length/np.sqrt(2))

    ds_flag = (mask1 | mask2)
    
    return ds_flag


def query_catalog_mask(ra, dec, diff_spikes=True, return_diagnostics=False, wise_cat_path=None):
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
    if wise_cat_path is None:
        wise_cat_path = wise_cat_path_default
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

    ra2, dec2 = map(np.copy, [decals_lon, decals_lat])
    sky2 = SkyCoord(ra2*u.degree,dec2*u.degree, frame='icrs')

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
            search_radius = np.max([circular_mask_radii_func(w1_ab[mask_wise]), 0.5*ds_mask_length_func(w1_ab[mask_wise])])

        # Find all pairs within the search radius
        ra1, dec1 = map(np.copy, [wise_lon[mask_wise], wise_lat[mask_wise]])
        sky1 = SkyCoord(ra1*u.degree,dec1*u.degree, frame='icrs')
        idx_wise, idx_decals, d2d, _ = sky2.search_around_sky(sky1, seplimit=search_radius*u.arcsec)
        print('%d nearby objects'%len(idx_wise))
        
        # convert distances to numpy array in arcsec
        d2d = np.array(d2d.to(u.arcsec))

        d_ra = (ra2[idx_decals]-ra1[idx_wise])*3600.    # in arcsec
        d_dec = (dec2[idx_decals]-dec1[idx_wise])*3600. # in arcsec
        ##### Convert d_ra to actual arcsecs #####
        mask = d_ra > 180*3600
        d_ra[mask] = d_ra[mask] - 360.*3600
        mask = d_ra < -180*3600
        d_ra[mask] = d_ra[mask] + 360.*3600
        d_ra = d_ra * np.cos(dec1[idx_wise]/180*np.pi)
        ##########################################

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

            print('{} objects masked by circular mask'.format(np.sum(circ_contam)))
            print('{} additionally objects masked by diffraction spikes mask'.format(np.sum(circ_contam | ds_contam)-np.sum(circ_contam)))
            print('{} objects masked by the combined masks'.format(np.sum(circ_contam | ds_contam)))
            print()

        else:

            print('{} objects masked'.format(np.sum(circ_contam)))
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
