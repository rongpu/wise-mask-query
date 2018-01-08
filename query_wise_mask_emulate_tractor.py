# Emulate tractor's choice of best WISE coadd

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
coadd_dir = '/project/projectdirs/cosmo/data/unwise/neo2/unwise-coadds/fulldepth/'

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

def query_wise_coadd(ra, dec, coadd_fn=coadd_fn, verbose=True):
    '''
    Find which WISE coadd each object belongs to and compute 
    the pixel coordinatess.
    
    Inputs
    ------
    ra, dec: coordinates of the objects;
    coadd_fn: location of the coadd table;

    Output
    ------
    coadd_idx_final: (array of int) the coadd index for each object;
    pixcrd_x_final, pixcrd_y_final: (array of float) the pixel coordinates, 
    note that the centers of each pixel have integer coordinate value starting
    from 1, and the smallest pixel coordinate value is 0.5.

    '''

    coadd = Table.read(coadd_fn)

    if verbose: print('Matching to the nearest WISE coadds\n')

    ra1=np.array(coadd['ra_center'])
    dec1=np.array(coadd['dec_center'])
    ra2=np.array(ra)
    dec2=np.array(dec)
    skycat1=SkyCoord(ra1*u.degree,dec1*u.degree, frame='icrs')
    skycat2=SkyCoord(ra2*u.degree,dec2*u.degree, frame='icrs')

    # Find the nearest 3 coadds for each object
    coadd_idx = np.zeros([3, len(ra)], dtype=int)
    coadd_idx[0], d2d_1, _ = skycat2.match_to_catalog_sky(skycat1, nthneighbor=1)
    coadd_idx[1], d2d_2, _ = skycat2.match_to_catalog_sky(skycat1, nthneighbor=2)
    coadd_idx[2], d2d_3, _ = skycat2.match_to_catalog_sky(skycat1, nthneighbor=3)

    # Pixel coordinates (in range of (0.5, 2048.5))
    pixcrd_x = np.zeros([3, len(ra)])
    pixcrd_y = np.zeros([3, len(ra)])

    # Check if objects are the 3 nearest coadds
    # and compute the distances to the nearest boundary
    inside = np.zeros([3, len(ra)], dtype=bool)
    order_str = ['nearest', 'second nearest', 'third nearest']
    d2b = -1.*np.ones([3, len(ra)])

    for index1 in range(3):
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
            pixcrd_x[index1][cat_mask] = pixcrd_x1
            pixcrd_y[index1][cat_mask] = pixcrd_y1            
            
        if verbose:
            print('{} ({:.1f}%) objects inside the {} coadd\n'
              .format(np.sum(inside[index1]), np.sum(inside[index1])/len(ra)*100., order_str[index1]))

    if np.sum((~inside[0]) & (~inside[1]) & (~inside[2]))!=0:
        raise ValueError('EROR: coadd not found!')

    inside_only_1 = inside[0] & (~inside[1]) & (~inside[2])
    inside_only_2 = (~inside[0]) & inside[1] & (~inside[2])
    inside_only_3 = (~inside[0]) & (~inside[1]) & (inside[2])
    if verbose:
        print('{} ({:.1f}%) objects only inside the nearest coadd'
              .format(np.sum(inside_only_1), np.sum(inside_only_1)/len(ra)*100.))
        print('{} ({:.1f}%) objects only inside the second nearest coadd'
              .format(np.sum(inside_only_2), np.sum(inside_only_2)/len(ra)*100.))
        print('{} ({:.1f}%) objects only inside the third nearest coadd\n'
              .format(np.sum(inside_only_3), np.sum(inside_only_3)/len(ra)*100.))

    # Assign each object to a coadd index
    coadd_idx_final = -1*np.ones(len(ra), dtype=int)
    # Record which of the three coadds is chosen
    choice_id = np.zeros(len(ra), dtype=int)
    # Pixel coordinates
    pixcrd_x_final = np.zeros(len(ra))
    pixcrd_y_final = np.zeros(len(ra))

    # Objects that are not ambiguous
    coadd_idx_final[inside_only_1] = coadd_idx[0][inside_only_1]
    pixcrd_x_final[inside_only_1] = pixcrd_x[0][inside_only_1]
    pixcrd_y_final[inside_only_1] = pixcrd_y[0][inside_only_1]
    choice_id[inside_only_1] = 0
    coadd_idx_final[inside_only_2] = coadd_idx[1][inside_only_2]
    pixcrd_x_final[inside_only_2] = pixcrd_x[1][inside_only_2]
    pixcrd_y_final[inside_only_2] = pixcrd_y[1][inside_only_2]
    choice_id[inside_only_2] = 1
    coadd_idx_final[inside_only_3] = coadd_idx[2][inside_only_3]
    pixcrd_x_final[inside_only_3] = pixcrd_x[2][inside_only_3]
    pixcrd_y_final[inside_only_3] = pixcrd_y[2][inside_only_3]
    choice_id[inside_only_3] = 2

    mask_overlap = (coadd_idx_final==-1)

    # Assign negative coadd index number to irrelavant coadds
    for index in range(3):
        mask = (~inside[index])
        coadd_idx[index][mask] = -1
        
    # For objects inside 2 or 3 coadds, choose the coadd with the 
    # largest index number
    coadd_idx_final[mask_overlap] = np.max(coadd_idx, axis=0)[mask_overlap]
    choice_id[mask_overlap] = np.argmax(coadd_idx, axis=0)[mask_overlap]
    for index in range(3):
        mask = choice_id==index
        pixcrd_x_final[mask] = pixcrd_x[index][mask]
        pixcrd_y_final[mask] = pixcrd_y[index][mask]

    if verbose:
        print('{} ({:.1f}%) objects belong to the nearest coadd'
              .format(np.sum(choice_id==0), np.sum(choice_id==0)/len(ra)*100.))
        print('{} ({:.1f}%) objects belong to the second nearest coadd'
              .format(np.sum(choice_id==1), np.sum(choice_id==1)/len(ra)*100.))
        print('{} ({:.1f}%) objects belong to the third nearest coadd\n'
              .format(np.sum(choice_id==2), np.sum(choice_id==2)/len(ra)*100.))

    return coadd_idx_final, pixcrd_x_final, pixcrd_y_final

def query_mask_value(ra, dec, coadd_fn=coadd_fn, coadd_dir=coadd_dir, verbose=True):
    '''
    Query WISE mask value at each object loation.
    
    Inputs
    ------
    ra, dec: coordinates of the objects;
    coadd_fn: location of the coadd table;
    coadd_dir: directory of the WISE mask images;
    
    Output
    ------
    wise_mask: the WISE mask value at each object loation.
    '''

    coadd_idx, pixcrd_x, pixcrd_y = query_wise_coadd(ra, dec, coadd_fn)

    # Convert the coordinates to integers with zero-based numbering of Python
    pixcrd_x = np.round(pixcrd_x-1.).astype(int)
    pixcrd_y = np.round(pixcrd_y-1.).astype(int)

    coadd = Table.read(coadd_fn)
    idx_unique = np.unique(coadd_idx)
    mask_value = np.zeros(len(ra), dtype=int)

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
            coadd['COADD_ID'][coadd_index], 
            'unwise-{}-msk.fits.gz'.format(coadd['COADD_ID'][coadd_index]))
        data = fits.getdata(img_path)
        cat_mask = coadd_idx==coadd_index
        mask_value[cat_mask] = data[pixcrd_y[cat_mask], pixcrd_x[cat_mask]]

    return mask_value
