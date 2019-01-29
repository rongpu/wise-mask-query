from __future__ import division, print_function
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import sys, os
from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy import wcs

coadd_fn = '/global/project/projectdirs/desi/users/rongpu/useful/astrom-atlas_radec_added.fits'
# coadd_dir = '/project/projectdirs/cosmo/data/unwise/neo2/unwise-coadds/fulldepth/'
# coadd_dir = '/global/cscratch1/sd/ameisner/brightmask_lrg'
coadd_dir = '/global/projecta/projectdirs/cosmo/work/wise/brightmask_latent_size'

def create_wcs(coadd, coadd_id):
    '''
    Create Astropy WCS object from the coadd table.

    Inputs
    ------
    coadd: coadd table;
    coadd_id: index in coadd table;

    Output
    ------
    w: WCS object.
    '''

    w = wcs.WCS(naxis=2)
    # Set up gnomonic projection
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    # WCS parameters
    w.wcs.cd = coadd['CD'][coadd_id]
    w.wcs.crval = coadd['CRVAL'][coadd_id]
    w.wcs.crpix = coadd['CRPIX'][coadd_id]
    # CDELT is ignored since cd is present
    # # w.wcs.cdelt = coadd['CDELT'][coadd_id]
    w.wcs.lonpole = coadd['LONGPOLE'][coadd_id]
    w.wcs.latpole = coadd['LATPOLE'][coadd_id]
    # Epoch must be defined
    w.wcs.equinox = 2000.
    
    return w

def query_wise_coadd(ra, dec, n_match, coadd_fn=coadd_fn, verbose=True):
    '''
    Find which WISE coadd each object belongs to and compute 
    the pixel coordinatess.
    
    Inputs
    ------
    ra, dec: coordinates of the objects;
    n_match: (int between 1 and 6) number of nearest coadds to match the coordinates to, 
    use 6 if the coordinates are within 1 degree of the poles, otherwise 4 is enough;
    coadd_fn: location of the coadd table;

    Output
    ------
    coadd_idx_final: (array of int) the coadd index for each object;
    pixcrd_x_final, pixcrd_y_final: (array of float) the pixel coordinates, 
    note that the centers of each pixel have integer coordinate value starting
    from 1, and the smallest pixel coordinate value is 0.5.

    '''

    coadd = Table.read(coadd_fn)

    if n_match<4:
        raise ValueError('n_match should be no less than 4')

    if verbose: print('Matching to the nearest WISE coadds\n')

    ra1 = np.array(coadd['ra_center'])
    dec1 = np.array(coadd['dec_center'])
    ra2 = np.array(ra)
    dec2 = np.array(dec)
    skycat1 = SkyCoord(ra1*u.degree,dec1*u.degree, frame='icrs')
    skycat2 = SkyCoord(ra2*u.degree,dec2*u.degree, frame='icrs')

    # Find the nearest coadds for each object
    coadd_idx = np.zeros([n_match, len(ra)], dtype=int)
    for index in range(n_match):
        coadd_idx[index], _, _ = skycat2.match_to_catalog_sky(skycat1, nthneighbor=index+1)

    # Pixel coordinates (in range of (0.5, 2048.5))
    pixcrd_x = np.zeros([n_match, len(ra)])
    pixcrd_y = np.zeros([n_match, len(ra)])

    # Check if objects are inside the nearest coadds
    # and compute the distances to the nearest boundary
    inside = np.zeros([n_match, len(ra)], dtype=bool)
    order_str = ['nearest', 'second nearest', 'third nearest', 'fourth nearest', 'fifth nearest', 'sixth nearest']
    d2b = -1.*np.ones([n_match, len(ra)])
    for index1 in range(n_match):
        if verbose:
            print('Finding the {} coadd'.format(order_str[index1]))
        idx_unique = np.unique(coadd_idx[index1])
        # Loop over unique coadds
        for index2 in range(len(idx_unique)):
            if verbose and (len(idx_unique)>=10):
                if index2%(len(idx_unique)//10)==0:
                    print('{:.0f}%'.format((index2)/len(idx_unique)*100))
            elif verbose:
                    print('{:.0f}%'.format((index2)/len(idx_unique)*100))
            # Create Astropy WCS object
            coadd_id = idx_unique[index2]
            w = create_wcs(coadd, coadd_id)
            # select objects that are matched to that coadd
            cat_mask = coadd_idx[index1]==coadd_id
            # Convert RA/Dec to pixel coordinates
            world = np.column_stack([ra2[cat_mask], dec2[cat_mask]])
            pixcrd = w.wcs_world2pix(world, True)
            # [0.5, 0.5] and [2048.5, 2048.5] are the corners of the image
            mask = (pixcrd[:, 0]>0.5) & (pixcrd[:, 0]<2048.5)
            mask &= (pixcrd[:, 1]>0.5) & (pixcrd[:, 1]<2048.5)
            inside[index1, cat_mask] = mask
            pixcrd_x1, pixcrd_y1 = pixcrd.transpose()
            d2b_all_boundaries = np.stack((pixcrd_x1 - 0.5, 2048.5-pixcrd_x1, pixcrd_y1 - 0.5, 2048.5-pixcrd_y1))
            d2b[index1][cat_mask] = np.min(d2b_all_boundaries, axis=0)
            pixcrd_x[index1][cat_mask] = pixcrd_x1
            pixcrd_y[index1][cat_mask] = pixcrd_y1            
            
        if verbose:
            print('{} ({:.1f}%) objects inside the {} coadd\n'
              .format(np.sum(inside[index1]), np.sum(inside[index1])/len(ra)*100., order_str[index1]))

    if np.sum((~inside[0]) & (~inside[1]) & (~inside[2]))!=0:
        raise ValueError('ERROR: WISE mask coadd not found!')

    # Identify objects that only appear in one coadd
    inside_only_one_coadd = np.zeros([n_match, len(ra)], dtype=bool)
    for index1 in range(n_match):
        inside_only_one_coadd[index1] = inside[index1]
        for index2 in range(n_match):
            if index1!=index2:
                inside_only_one_coadd[index1] &= (~inside[index2])

    # for index in range(3):
    #     if verbose:
    #         print('{} ({:.1f}%) objects only inside the {} coadd'
    #               .format(np.sum(inside_only_one_coadd[index]), 
    #                 np.sum(inside_only_one_coadd[index])/len(ra)*100., 
    #                 order_str[index]))

    # Assign each object to a coadd index
    coadd_idx_final = -1*np.ones(len(ra), dtype=int)
    # Record which of the three coadds is chosen
    choice_id = np.zeros(len(ra), dtype=int)
    # Pixel coordinates
    pixcrd_x_final = np.zeros(len(ra))
    pixcrd_y_final = np.zeros(len(ra))

    # Assign coadd index to objects only inside one coadd
    for index in range(n_match):
        coadd_idx_final[inside_only_one_coadd[index]] = coadd_idx[index][inside_only_one_coadd[index]]
        pixcrd_x_final[inside_only_one_coadd[index]] = pixcrd_x[index][inside_only_one_coadd[index]]
        pixcrd_y_final[inside_only_one_coadd[index]] = pixcrd_y[index][inside_only_one_coadd[index]]
        choice_id[inside_only_one_coadd[index]] = index

    mask_overlap = (coadd_idx_final==-1)

    # Assign negative distance to irrelavant coadds
    # so that they will not be chosen
    for index in range(n_match):
        d2b[index][~inside[index]] = -1.
        
    # For objects inside more than one coadd, choose the coadd whose
    # boundaries are farthest away from the object
    argmax = np.argmax(d2b, axis=0)
    for index in range(n_match):
        mask = mask_overlap & (argmax==index)
        coadd_idx_final[mask] = coadd_idx[index][mask]
        pixcrd_x_final[mask] = pixcrd_x[index][mask]
        pixcrd_y_final[mask] = pixcrd_y[index][mask]
        choice_id[mask] = index

    if verbose:
        for index in range(3):
            print('{} ({:.1f}%) objects belong to the {} coadd'.format(
                np.sum(choice_id==index), np.sum(choice_id==index)/len(ra)*100., 
                order_str[index]))
        print()

    return coadd_idx_final, pixcrd_x_final, pixcrd_y_final

def query_mask_value(ra, dec, n_match, coadd_fn=coadd_fn, coadd_dir=coadd_dir, verbose=True):
    '''
    Query WISE mask value at each object loation.
    
    Inputs
    ------
    ra, dec: coordinates of the objects;
    n_match: (int between 1 and 6) number of nearest coadds to match the coordinates to, 
    use 6 if the coordinates are within 1 degree of the poles, otherwise 4 is enough;
    the coordinates are within 1 degree of the poles, otherwise 4 is enough;
    coadd_fn: location of the coadd table;
    coadd_dir: directory of the WISE mask images;
    
    Output
    ------
    mask_value: the WISE mask value at each object loation.

    Note: the following script converts mask_value to wisemask_w1 and wisemask_w2 in the
    DECaLS catalogs:
        wisemask_w1 = mask_value%4
        wisemask_w2 = mask_value//4
    '''

    coadd_idx, pixcrd_x, pixcrd_y = query_wise_coadd(ra, dec, n_match, coadd_fn)
    mask_found = coadd_idx>=0

    # Convert the coordinates to integers with zero-based numbering of Python
    pixcrd_x = np.round(pixcrd_x-1.).astype(int)
    pixcrd_y = np.round(pixcrd_y-1.).astype(int)

    coadd = Table.read(coadd_fn)
    idx_unique = np.unique(coadd_idx[mask_found])
    mask_value = -1 * np.ones(len(ra), dtype=int)

    if verbose: print('Obtaining mask values from images')
    for index in range(len(idx_unique)):
        if verbose and (len(idx_unique)>=10):
            if index%(len(idx_unique)//10)==0:
                print('{:.0f}%'.format((index)/len(idx_unique)*100))
        elif verbose:
                print('{:.0f}%'.format((index)/len(idx_unique)*100))
        coadd_index = idx_unique[index]
        img_path = os.path.join(coadd_dir, 
            coadd['COADD_ID'][coadd_index][:3], 
            'unwise-{}-msk.fits.gz'.format(coadd['COADD_ID'][coadd_index]))
        if os.path.exists(img_path):
            # print('YES!')
            data = fits.getdata(img_path)
            cat_mask = coadd_idx==coadd_index
            mask_value[cat_mask] = data[pixcrd_y[cat_mask], pixcrd_x[cat_mask]]
        else:
            print('NOT FOUND:', img_path)

    return mask_value
