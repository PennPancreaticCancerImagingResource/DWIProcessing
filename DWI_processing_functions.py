#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#@title
# 10/15/20 Modify for recent mouse diffusion data
# Order:
##begin view
##    begin b-values # Two per b-value (opposite polarity - take the geometric mean)
##        begin slices
##            read
##        end slices
##    end b-value
##end view
# 11/23/20 Add more functions to allow more automated processing (Correct freq/center k-space; One slice/all b-value; All slices/one b-value, etc)

# my_function_file
# HKS 5/12/20 All my functions for radial recon

import numpy as np
import matplotlib.pyplot as plt
#from scipy.fftpack import fft
import scipy.special as sp
import math
from scipy import signal
from scipy.optimize import curve_fit
import cmath
from scipy.spatial import Voronoi, voronoi_plot_2d
from scipy.spatial import ConvexHull, convex_hull_plot_2d

PI = np.pi

# =====================================================================================================================
# Find the off-resonance frequency
# =====================================================================================================================

def correct_frequency(rawdata_all, xres, views, slices):

    # Let's average many views to get an average correction
    num_views_correction = views//2 - 5 # Must be less than 1/2 of total views
    ZF_fact = 8
    profile_tot_0 = np.zeros(ZF_fact*xres)
    profile_tot_180rev = np.zeros(ZF_fact*xres)

    for jj in range(0, num_views_correction): # Look at opposing projections to estimate shift
        
        first_view = jj
        ro_0 = rawdata_all[first_view,0,slices//2,0:xres] # Consider just one (center) slice, lowest b-value
        ro_180 = rawdata_all[first_view+views//2+1,0,slices//2,0:xres]

        # Zerofill for finer resolution
        tmp = np.zeros(xres*ZF_fact, dtype = complex)
        tmp[(xres*ZF_fact//2 - xres//2):(xres*ZF_fact//2 + xres//2)] = ro_0
        ro_0 = tmp
        tmp = np.zeros(xres*ZF_fact, dtype = complex)
        tmp[(xres*ZF_fact//2 - xres//2):(xres*ZF_fact//2 + xres//2)] = ro_180
        ro_180 = tmp

        # Look at phase difference
        ##num_points = 31
        ##phasediff = np.zeros(num_points)
        ##for i in range(0,num_points):
        ##    phasediff[i] = np.angle(ro_0[xres//2-num_points//2+i]) - np.angle(ro_180rev[xres//2-num_points//2+i])

        profile_0 = np.fft.fftshift(np.fft.fft(np.fft.fftshift(ro_0)))
        profile_180 = np.fft.fftshift(np.fft.fft(np.fft.fftshift(ro_180)))
        profile_180rev = np.flipud(profile_180)
        #profile_180rev = np.roll(profile_0, 4) # For testing purposes

        profile_tot_0 = profile_tot_0 + abs(profile_0)
        profile_tot_180rev = profile_tot_180rev + abs(profile_180rev)

    profile_0 = profile_tot_0
    profile_180rev = profile_tot_180rev

    num_points = 50*ZF_fact
    diff = np.zeros(num_points)
    for i in range(0,num_points):
        profile_diff = abs(profile_0) - abs(np.roll(profile_180rev, (i-num_points//2)))
        diff[i] = np.sum(abs(profile_diff))

    minpoint_arr = np.where(diff == np.amin(diff)) # find the minimum location
    minpoint = minpoint_arr[0]
    offreson = (num_points/2 - minpoint)/(2*ZF_fact)
    print("minimum = ", minpoint, "  offreson = ", offreson)

    return offreson

# =====================================================================================================================
# Find k-space center
# =====================================================================================================================

def find_center(rawdata_all, xres, views, slices):

    sl = slices//2 # Just consider the center slice; slice number starts at 1

    sl_order = ((sl-1)//2) + (sl%2)*(slices+1)//2 # Reorder since slices were 2 x interleaved (e.g., 1-3-5-7-2-4-6)
            
    rawdata = rawdata_all[:,0,sl_order,0:xres] # order: views, b-values, slices, xres
     
    # Look at the data or find the center
    avgmax = 0.0
    for i in range(0, views):
        tmp = rawdata[i,:]
        maxindex_arr = np.where(np.absolute(tmp) == np.amax(np.absolute(tmp))) # find the peak location
        maxindex = maxindex_arr[0]
        # print(maxindex)
        avgmax += 1.0*maxindex/views
    print("average peak location = ", avgmax)
    #plt.figure(1)
    #plt.imshow(abs(rawdata), cmap='gray')
    #plt.show()

    return avgmax

# =====================================================================================================================
# Process one slice, all b-values
# =====================================================================================================================

def recon_one_slice(rawdata_all, xres, xres_ro, yres, views, slices, bvalues, sl, angl, M, L, beta, zerofill, peak_location, offreson, weights):

    Narr = np.arange(xres) - xres/2
    imgarr = np.zeros((bvalues,xres,xres)) # Array for storage and saving; default type = 
    imgcxarr = np.zeros((bvalues,(zerofill+1)*xres,(zerofill+1)*xres), dtype=complex) # Save the k-space and image-space cx data
    kgridarr = np.zeros((bvalues,(zerofill+1)*xres,(zerofill+1)*xres), dtype=complex)    

    for bval in range(0, bvalues):

        sl_order = ((sl-1)//2) + ((sl-1)%2)*(slices+1)//2 # Reorder since slices were 2 x interleaved (e.g., 1-3-5-7-2-4-6)
        
        rawdata = np.copy(rawdata_all[:,bval,sl_order,0:xres]) # order: views, b-values, slices, xres
        #print("info: ", type(rawdata), rawdata.shape)
        #peak_location = 47 # 96 acquired points, but 128 saved; rest zerofilled

        # Look at the data or find the center
##        avgmax = 0.0
##        for i in range(0, views):
##            tmp = rawdata[i,:]
##            maxindex_arr = np.where(np.absolute(tmp) == np.amax(np.absolute(tmp))) # find the peak location
##            maxindex = maxindex_arr[0]
##            print(maxindex)
##            avgmax += 1.0*maxindex/views
##        print("average peak location = ", avgmax)
##        plt.imshow(abs(rawdata), cmap='gray')
##        plt.show()
##        quit()

        # Try plotting the sinogram
##        for i in range(0, views):
##            tmp = rawdata[i,:]
##            tmp_proj = np.fft.fftshift(np.fft.fft(np.fft.fftshift(tmp)))
##            rawdata[i,:] = tmp_proj
##        plt.imshow(abs(rawdata), cmap='gray')
##        plt.show()
##        quit()
        
        # # Shift to center data
        # for i in range(0, views):
        #     tmp = rawdata[i,:]
        #     rawdata[i,:] = np.roll(tmp, (xres//2 - peak_location))

        linphase = -(xres//2 - peak_location)*2*PI/xres * Narr
        for i in range(0, views):
            tmp = rawdata[i,:]
            tmp_proj = np.fft.ifftshift(np.fft.fft(np.fft.fftshift(tmp)))
            tmp_proj = tmp_proj * (np.cos(linphase) + 1j*np.sin(linphase)) # complex() function only accepts scalar
            tmp = np.fft.ifftshift(np.fft.ifft(np.fft.fftshift(tmp_proj)))
            #tmp[(xres - (xres_ro-xres)):xres] = 2*tmp[(xres - (xres_ro-xres)):xres] # Also double the high freq component to compensate for partial echo
            rawdata[i,:] = tmp
                  
        # Phase normalize
        for i in range(0, views):
            tmp = rawdata[i,:]
            datphase = np.angle(tmp[xres//2]) # math.atan(imag(tmp[xres//2]), real(tmp[xres//2])) # Normalize to phase at center of each readout
            tmp = tmp * complex(math.cos(datphase), -math.sin(datphase))
            rawdata[i,:] = tmp

        # Off-resonance frequency correction
        #offreson = 0.4375 # Number of cycles during readout period
        linphase = offreson*2*PI/xres * Narr
        for i in range(0, views):
            tmp = rawdata[i,:]
            tmp = tmp * (np.cos(linphase) + 1j*np.sin(linphase)) # complex() function only accepts scalar
            rawdata[i,:] = tmp

        # OR Simulate the data
        #rawdata = mf.phantom(xres,views)
        #mf.fermifunc(rawdata) # apply Fermi function

        # Do the GRIDDING
        img, kgrid, imgcx = gridding(rawdata, xres, yres, angl, L, M, beta, zerofill, weights)
#        img = gridding(rawdata, xres, yres, angl, L, M, beta, zerofill, weights)
#        img = np.absolute(img)
        imgarr[bval,:,:] = img
        imgcxarr[bval,:,:] = imgcx
        kgridarr[bval,:,:] = kgrid
        print("b-value = ", bval)

    return imgarr #, kgridarr, imgcxarr

# =====================================================================================================================
# Process all slices, b-value = 0
# =====================================================================================================================

def recon_all_slices(rawdata_all, xres, yres, views, slices, angl, M, L, beta, zerofill, peak_location, offreson):

    Narr = np.arange(xres) - xres/2
    imgarr = np.zeros((slices,xres,xres)) # Array for storage and saving; default type = float64

    for sl in range(0, slices):

        sl_order = (sl//2) + (sl%2)*(slices+1)//2 # Reorder since slices were 2 x interleaved (e.g., 1-3-5-7-2-4-6)
        
        rawdata = np.copy(rawdata_all[:,0,sl_order,0:xres]) # order: views, b-values, slices, xres
        #print("info: ", type(rawdata), rawdata.shape)
        #peak_location = 47 # 96 acquired points, but 128 saved; rest zerofilled

        # Look at the data or find the center
##        avgmax = 0.0
##        for i in range(0, views):
##            tmp = rawdata[i,:]
##            maxindex_arr = np.where(np.absolute(tmp) == np.amax(np.absolute(tmp))) # find the peak location
##            maxindex = maxindex_arr[0]
##            print(maxindex)
##            avgmax += 1.0*maxindex/views
##        print("average peak location = ", avgmax)
##        plt.imshow(abs(rawdata), cmap='gray')
##        plt.show()
##        quit()

        # Try plotting the sinogram
##        for i in range(0, views):
##            tmp = rawdata[i,:]
##            tmp_proj = np.fft.fftshift(np.fft.fft(np.fft.fftshift(tmp)))
##            rawdata[i,:] = tmp_proj
##        plt.imshow(abs(rawdata), cmap='gray')
##        plt.show()
##        quit()
        
        # Shift to center data
        for i in range(0, views):
            tmp = rawdata[i,:]
            rawdata[i,:] = np.roll(tmp, (xres//2 - peak_location))
                  
        # Phase normalize
        for i in range(0, views):
            tmp = rawdata[i,:]
            datphase = np.angle(tmp[xres//2]) # math.atan(imag(tmp[xres//2]), real(tmp[xres//2])) # Normalize to phase at center of each readout
            tmp = tmp * complex(math.cos(datphase), -math.sin(datphase))
            rawdata[i,:] = tmp

        # Off-resonance frequency correction
        #offreson = 0.4375 # Number of cycles during readout period
        linphase = offreson*2*PI/xres * Narr
        for i in range(0, views):
            tmp = rawdata[i,:]
            tmp = tmp * (np.cos(linphase) + 1j*np.sin(linphase)) # complex() function only accepts scalar
            rawdata[i,:] = tmp

        # OR Simulate the data
        #rawdata = mf.phantom(xres,views)
        #mf.fermifunc(rawdata) # apply Fermi function

        # Do the GRIDDING
        img = gridding(rawdata, xres, yres, angl, L, M, beta, zerofill) 
        img = np.absolute(img)
        imgarr[sl,:,:] = img
        print("slice = ", sl)

    return imgarr


# =====================================================================================================================
# Read in raw data
# =====================================================================================================================

def read_raw_data(input_file, xres, views, slices, bvalues, offset1, downsample, downsample_factor):

    tmpdat = np.fromfile(input_file, dtype=np.int32, count = 2*xres*views*slices*bvalues, offset=offset1)
#    tmpdat = np.reshape(tmpdat,(slices,bvalues,views,xres,2)) # For default "C" like read, order is "backwards"
    tmpdat = np.reshape(tmpdat,(views,bvalues,slices,xres,2)) # For default "C" like read, order is "backwards"
    #tmpdat = tmpdat.byteswap(True) # swap_endian?? Not sure this is correct call
    rawdata = tmpdat[:,:,:,:,0] + 1j*tmpdat[:,:,:,:,1] # Convert from long int to complex
    if downsample == 1:
      rawdata = rawdata[::downsample_factor,:,:,:]
    
    views = rawdata.shape[0]

    print("rawdata input: (after downsampling, if downsampled)", type(rawdata), rawdata.shape)

    # Try zerofilling in case k-space center is in between 2 points? In mouse data, maybe helped SNR ~ 10-15%
    #rawdata = zf_func(rawdata) # zerofill by x2 if desired
    #xres = xres * 2

    return rawdata, views

# =====================================================================================================================
# Kaiser-Bessel function for convolution kernel 
# =====================================================================================================================

def kb(M, beta): 
    kb_func = signal.kaiser(M, beta)
    return kb_func

# =====================================================================================================================
# Compute the de-apodizing filter
# =====================================================================================================================

def deapodize(L, beta, xres, yres):

    filterx = np.arange(1.0*xres) # float array
    for i in range(0,xres):
        d2 = (PI*L*(i-xres/2)/xres)**2 - beta**2
        if d2 > 0.0:
            d2 = math.sqrt(d2)
            filterx[i] = math.sin(d2)/d2
        elif d2 < 0.0:
            d2 = complex(0,math.sqrt(-d2))
            filterx[i] = abs(cmath.sin(d2)/d2)
        else: filterx[i] = 1
        
    filtery = np.arange(1.0*yres)
    for i in range(0,yres):
        d2 = (PI*L*(i-yres/2)/yres)**2 - beta**2
        if d2 > 0.0:
            d2 = math.sqrt(d2)
            filtery[i] = math.sin(d2)/d2
        elif d2 < 0.0:
            d2 = complex(0,math.sqrt(-d2))
            filtery[i] = abs(cmath.sin(d2)/d2)
        else: filtery[i] = 1

    filter = np.zeros((yres,xres))
    for i in range(0,xres):
        for j in range(0,yres):
            filter[j][i] = filterx[i] * filtery[j]

    return filter

# =====================================================================================================================
# Do the gridding here 
# =====================================================================================================================

def gridding(rawdat, xres, yres, angl, L, M, beta, zerofill, weights):

    # print("Enter gridding")

    #xres = len(rawdat[0]) # if zerofilling, then want to change xres and yres
    #yres = xres
    views = len(rawdat)

    if zerofill == 1:
        rawdat = zf_func(rawdat) # zerofill by x2 if desired
        xres = xres * 2
        yres = yres * 2
        

    # print("Gridding ", xres, yres)
    
    #angl = (math.sqrt(5)-1)/2 * PI # golden angle in radians
    kgrid = np.zeros((xres, yres), dtype=complex) # Initialize the cumulative k-space grid
    kbfunc = kb(M,beta) # load the 1D Kaiser-Bessel kernel

    # Density compensation
    #simple_ramp = abs(np.arange(xres) - xres//2)
    #for j in range(0,views):
    #    rawdat[j,:] = rawdat[j,:]*simple_ramp

    # voronoi and ramlak perform similarly even for golden angle
    #rawdat = ramlak(rawdat) # Do ramlak filter for density compensation
    # rawdat = voronoi_density_comp(rawdat, xres, views, angl) # Use voronoi
    rawdat = rawdat*weights # Seems like computing VORONOI takes vast majority of time; try reading it in


    # Gridding
    for j in range(0,views):
        ang = j * angl
        #ang = j*2*PI/views # try equal spacing
        #print(j, ang*180/PI)
        
        for i in range(0,xres):
            x = (i - xres/2)*math.cos(ang) + xres/2 # x,y positions on Cartesian grid centered at xres/2, yres/2
            y = (i - xres/2)*math.sin(ang) + xres/2

            x1 = math.ceil(x - L/2); x2 = math.floor(x + L/2) # range of points affected by convolution
            y1 = math.ceil(y - L/2); y2 = math.floor(y + L/2)
            #print(j, x1, x2, y1, y2)
            
            if x1 <= 0: x1 = 0 # make sure we don't exceed k-space range
            if x2 >= (xres-1): x2 = xres-1 
            if y1 <= 0: y1 = 0
            if y2 >= (yres-1): y2 = yres-1

            yy = y1 # start the convolution
            while yy <= y2:
                xx = x1
                ay = round(abs(y - yy)*M/L + M/2) # convert from k-space coordinate to Kaiser-Besel index
                
                while xx <= x2:
                    ax = round(abs(x - xx)*M/L + M/2)
                    #print(ay, ax)
                    if ay > 500: ay = 500
                    if ax > 500: ax = 500
                    kgrid[yy][xx] += rawdat[j][i] * kbfunc[ax] * kbfunc[ay] # j = row (view) number; i = column number
                    xx += 1
                yy += 1
                
                    
    img = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(kgrid))) # FFT the gridded data
    fconv = deapodize(L, beta, xres, yres)
    imgcx = img/fconv # deapodize

    if zerofill == 1: # return to original matrix size (and FOV)
        img2 = np.zeros((xres//2, yres//2), dtype=complex)
        img2 = img[(yres//4):(3*yres//4),(xres//4):(3*xres//4)]
        xres = xres//2
        yres = yres//2
        img = np.absolute(img2)

    return img, kgrid, imgcx # Return complex k-space and image-space data, as well as (cropped if ZF=1) final image (float)

# =====================================================================================================================
# Ram-Lak filter for equi-angluar radial data
# =====================================================================================================================

def ramlak(rawdat):

    xres = len(rawdat[0])
    views = len(rawdat)
    phi = np.zeros(xres)

    print("ramlak density compensation")
    
    for i in range(0, xres):
        j = i - xres//2
        
        if j == 0: phi[i] = PI/2
        elif abs(j % 2) == 1: phi[i] = -2.0/(PI*j*j) # Odd numbers

    rl_ramp = abs(np.fft.fftshift(np.fft.fft(np.fft.fftshift(phi))))/(2*PI)

    for i in range(0, views):
        rawdat[i,:] *= rl_ramp

    #plt.plot(rl_ramp)
    #plt.show()

    return rawdat
        
# =====================================================================================================================
# Fermi Function - new for Python - make the filter a function of the image matrix size
# =====================================================================================================================

def fermifunc(rawdat):

    xres = len(rawdat[0])
    views = len(rawdat)
    
    #sizef = 150.0 # This term (and the if loop below) was needed to avoid calculation error due to too large or
                   # too small numbers in exponent. For Python, not needed.
    width = xres*0.05 # Let's make the width about 45% of the matrix size (0.05 x 9)
                 # Width is "proportional" to the transition region - lower value = sharper boundary; Previously set to 10.0
                 # 5% to 95% of the transition occurs within 9*width
    halfpoint = xres*0.15 # Let's make the transition region about 15% from the edges
    # halfpoint = 30 # number of points from the edge of kspace where fermi = 0.5 (transition center)
    
    ferm_arr = np.zeros(xres)
    
    for i in range(0, xres):        
        dist = abs(i - xres/2)
        ilimit = dist - (xres/2 - halfpoint)
        ferm_arr[i] = 1.0/(np.exp(ilimit/width) + 1)
        #if ilimit > (-sizef): ferm_arr[i] = 1.0/(np.exp(ilimit/width) + 1)
        #else: ferm_arr[i] = 1

    for i in range(0, views):
        rawdat[i,:] *= ferm_arr

    #plt.plot(ferm_arr)
    #plt.show()

# =====================================================================================================================
# Zerofill
# =====================================================================================================================

def zf_func(rawdat):

    xres = len(rawdat[0])
    views = len(rawdat)

    # print("Zerofill")
    
    xres2 = 2*xres
    dat2 = np.zeros((views,xres2),dtype = complex)
    tmp2 = np.zeros(xres2, dtype = complex)

    for i in range(0, views):
        tmp = rawdat[i,:]
        tmp = np.fft.fftshift(np.fft.fft(np.fft.fftshift(tmp)))
        tmp2 = np.zeros(xres2, dtype = complex)
        tmp2[(xres2//2-xres//2):(xres2//2+xres//2)] = tmp
        tmp2 = np.fft.fftshift(np.fft.fft(np.fft.fftshift(tmp2)))
        dat2[i,:] = tmp2

    return dat2


# =====================================================================================================================
# Simulate radial, golden angle circular phantom data here. Returns complex radial data 
# =====================================================================================================================

def phantom(xres, views):
    
    w = 32 # width of simulated object
    Narr = np.arange(xres) - xres/2
    Narr[xres//2] = 0.00000001 # Avoid divide by 0

    # sinc function
    # a = np.sinc(Narr*w/xres) # sinc(x) = sin(PI*x)/(PI*x)
    # fa = np.roll(fft(np.roll(a, xres//2)), xres//2)

    # jinc function and projection of a circular object
    a = sp.j1(Narr*w*PI/xres)/(2*Narr)
    #fa = np.roll(np.fft.fft(np.roll(a, xres//2)), xres//2)
    rawdata = np.ones((views,1)) * a # for a single circular phantom at center, just repeat for all views

    return rawdata

# =====================================================================================================================
# VORONOI density compensation 
# =====================================================================================================================

def voronoi_volumes(points):
    v = Voronoi(points)
    vol = np.zeros(v.npoints)
    for i, reg_num in enumerate(v.point_region):
        indices = v.regions[reg_num]
        if -1 in indices: # some regions can be opened
            vol[i] = np.inf
        else:
            vol[i] = ConvexHull(v.vertices[indices]).volume
    return vol

def voronoi_density_comp(rawdat, xres, views, ang, downsample):

    # print("Voronoi density compensation")

    #ang = (math.sqrt(5)-1)/2 * PI # golden angle in radians
    #ang = 2*PI/views
    if downsample == 0: angle_arr = ang * np.arange(views) # all angles
    elif downsample == 1: 
      angle_arr = ang* np.arange(views, step=4)
      views = angle_arr.size

    x = np.arange(xres) - xres//2

    kx_all = []
    ky_all = []

    for i in range(0, views):
        kx = x * np.cos(angle_arr[i])
        ky = x * np.sin(angle_arr[i])
        kx_all = kx_all + list(kx)
        ky_all = ky_all + list(ky)

    kxy = list(zip(kx_all, ky_all))
    A = np.asarray(kxy)

    #plt.plot(kx_all, ky_all, '.', c='black') # pixel (",") or dot (".") markers
    #plt.show()

    vor = Voronoi(A)
    vol = voronoi_volumes(A) # one long array
    weights = np.reshape(vol,(views,xres)) # reshape to k-space dimension

    # need to divide center by number of views (since points overlap) and set the edge weights
    weights[:,(xres//2)] = weights[:,(xres//2)]/views
    weights[:,0] = weights[:,1] + (weights[:,1] - weights[:,2]) # use slope to determine end weights
    weights[:,(xres-1)] = weights[:,(xres-2)] + (weights[:,(xres-2)] - weights[:,(xres-3)])

    rawdat = rawdat * weights

    return rawdat

    #voronoi_plot_2d(vol) # for some reason, gives error

    ##plt.plot(weights[views//2,:])
    ##plt.plot(weights[0,:])
    ##plt.show()

    # Next few lines used just to test/understand voronoi
    ##x=[0,1,0,1,0,1,0,1,0,1]
    ##y=[0,0,1,1,2,2,3,3.5,4,4.5]
    ##
    ##points = list(zip(x,y))
    ##A = np.array(points)
    ##points = np.array([[0, 0], [0, 1], [0, 2], [0, 3], \
    ##                   [1, 0], [1, 1], [1, 2], [1, 3], \
    ##                   [2, 0], [2, 1], [2, 2], [2, 3], \
    ##                   [3, 0], [3, 1], [3, 2], [3, 3]])
    ##
    ##vor=Voronoi(points)
    ##
    ##def voronoi_volumes(points):
    ##    v = Voronoi(points)
    ##    vol = np.zeros(v.npoints)
    ##    for i, reg_num in enumerate(v.point_region):
    ##        indices = v.regions[reg_num]
    ##        if -1 in indices: # some regions can be opened
    ##            vol[i] = np.inf
    ##        else:
    ##            vol[i] = ConvexHull(v.vertices[indices]).volume
    ##    return vol
    ##
    ##vol = voronoi_volumes(points)
    ##print("volumes: ", vol)
    ##voronoi_plot_2d(vor)
    ##plt.show()


# =====================================================================================================================
# Compute weights for VORONOI density compensation, which can be used later instead of repeatedly using function above
# =====================================================================================================================

def voronoi_density_comp_weights(xres, views, ang, downsample, downsample_factor):

    # print("Voronoi density compensation")

    #ang = (math.sqrt(5)-1)/2 * PI # golden angle in radians
    #ang = 2*PI/views
    if downsample == 0: angle_arr = ang * np.arange(views) # all angles
    elif downsample == 1: 
      angle_arr = ang* np.arange(views, step=downsample_factor)
      views = angle_arr.size

    x = np.arange(xres) - xres//2

    kx_all = []
    ky_all = []

    for i in range(0, views):
        kx = x * np.cos(angle_arr[i])
        ky = x * np.sin(angle_arr[i])
        kx_all = kx_all + list(kx)
        ky_all = ky_all + list(ky)

    kxy = list(zip(kx_all, ky_all))
    A = np.asarray(kxy)

    #plt.plot(kx_all, ky_all, '.', c='black') # pixel (",") or dot (".") markers
    #plt.show()

    vor = Voronoi(A)
    vol = voronoi_volumes(A) # one long array
    weights = np.reshape(vol,(views,xres)) # reshape to k-space dimension

    # need to divide center by number of views (since points overlap) and set the edge weights
    weights[:,(xres//2)] = weights[:,(xres//2)]/views
    weights[:,0] = weights[:,1] + (weights[:,1] - weights[:,2]) # use slope to determine end weights
    weights[:,(xres-1)] = weights[:,(xres-2)] + (weights[:,(xres-2)] - weights[:,(xres-3)])

    return weights

# =====================================================================================================================
# Open and create mask array (shape [nROI, slice, x, y]) from raw mask file (shape [slices, x, y])
# =====================================================================================================================

def open_mask(mask_file, slices, yres, xres):
    mask_raw = np.fromfile(mask_file, dtype= np.int8, count=xres*yres*slices)
    mask_raw = np.reshape(mask_raw, (slices, yres, xres))

    n_ROI = np.amax(mask_raw)
    mask_all = np.ones((n_ROI, slices, yres, xres), dtype = np.uint8)
    mask_all = mask_all*np.nan

    for iROI in range(n_ROI):
        for islice in range(slices):
            for ix in range(xres):
                for iy in range(yres):
                      if mask_raw[islice, iy, ix] == iROI+1:
                           mask_all[iROI, islice, iy, ix] = 1

    return mask_raw, mask_all

# ============ Reconstruct image =========
def reconstruct_radial_DWIs(input_file, img_dim, kspace_dim, zerofill=1, downsample_factor=0, offset=0, save_dat=True):

    print("Start reconstruction program!")
    PI = np.pi

    xres_ro = kspace_dim[0]
    views = kspace_dim[1]
    angl = kspace_dim[2]
    slices = img_dim[0]
    bvalues = img_dim[1]
    yres = img_dim[2]
    xres = img_dim[3]

    # Raw data file info here
    print('zerofill = ' + str(zerofill))

    downsample = 0
    if downsample_factor != 0: downsample = 1
    else: print('downsample = ', str(downsample), 'by a factor of ', str(downsample_factor))

    # Constants for the convolution kernel
    M = 501  # M only determines how fine to sample, not beta.
    L = 4  # Keep same definition for beta as I did for IDL for the purpose of computing beta.
    beta = PI * L / 2

    Narr = np.arange(xres) - xres / 2

    weights = voronoi_density_comp_weights(xres * (zerofill + 1), views, angl, downsample, downsample_factor)

    # Read in data
    rawdata_all, views = read_raw_data(input_file, xres_ro, views, slices, bvalues, offset, downsample,
                                       downsample_factor)
    if downsample == 1:
        angl = angl * downsample_factor

    # Find the frequency offset
    offreson = correct_frequency(rawdata_all, xres, views, slices)

    # Find the k-space center
    peak_location_arr = find_center(rawdata_all, xres, views, slices)
    peak_location = peak_location_arr.item(0)  # Don't round

    # Recon all slices, all b-values
    allslices = np.zeros((slices, bvalues, xres, xres))
    for sl in range(1, (slices + 1)):  # Slice starts at 1
        one_slice_arr = recon_one_slice(rawdata_all, xres, xres_ro, yres, views, slices, bvalues, sl, angl, M, L, beta,
                                        zerofill, peak_location, offreson, weights)
        allslices[sl - 1, :, :, :] = one_slice_arr
        # allkgridarr[sl-1,:,:,:] = kgridarr
        # allimgcxarr[sl-1,:,:,:] = imgcxarr
        allslices[sl - 1, :, :, :] = one_slice_arr
        print("******* Slice = ", sl, " *************")
        if sl == 1:
            plt.figure(1)
            maxintensity = np.amax(one_slice_arr)
            plt.imshow(one_slice_arr[0, :, :], cmap='gray', vmin=0, vmax=maxintensity, origin='upper')
            plt.show()

    # Save data
    out_file = "AllSlicesBvalues.bin"
    if downsample == 1: out_file = "AllSlicesBvalues_" + str(downsample_factor) + "x_downsampled.bin"
    if save_dat == 1: allslices.tofile(out_file)

    print("Done!")
    return allslices

# ============= DWI parameters from fit ========
def diffusion_fit(b_array, dwis=None, input_file="AllSlicesBvalues.bin", img_dims=None, SNR_threshold=5, noise_region=[0,10,0,10]):
    print("Start ADC fitting program! \n An overflow runtime error is expected; just means fitting does not converge in some pixels.")

    # Define the diffusion function to fit
    def func(x, a, b, c):
        return a * np.exp(-b * x) + c

    if dwis is not None:
        imgarr = dwis
        slices = dwis.shape[0]
        yres = dwis.shape[2]
        xres = dwis.shape[3]
    else:
        tmpdat = np.fromfile(input_file, dtype=np.float64)
        slices = img_dims[0]
        bvalues = img_dims[1]
        yres = img_dims[2]
        xres = img_dims[3]
        imgarr = np.reshape(tmpdat, (slices, bvalues, yres, xres))

    maxval = np.amax(np.absolute(imgarr))
    imgarr = imgarr / maxval * 100  # Normalize to 100 since absolute values are meaningless

    noise_mean_BL = np.average(imgarr[:, 0, noise_region[2]:noise_region[3], noise_region[0]:noise_region[1]])
    fit_threshold = noise_mean_BL * SNR_threshold

    # Fit the diffusion
    fitval = np.empty((3, slices, yres, xres))  # Fit all slices
    fitval[:] = np.nan
    for sl in range(0, slices):
        img = imgarr[sl, :, :, :]

        error_count = 0
        for j in range(0, yres):
            for i in range(0, xres):
                yn = img[:, j, i]
                #                yn = img[i,j,:]
                if (yn[0] > fit_threshold):
                    init_bval = -np.log(yn[1] / yn[0]) / (b_array[1] - b_array[0])
                    try:
                        popt, pcov = curve_fit(func, b_array, yn, p0=[(yn[0] - yn[4]), init_bval, yn[4]])
                        fitval[0, sl, j, i] = popt[1]
                        fitval[1, sl, j, i] = np.clip(popt[2]/ popt[0], 0, 1) # reasonable given we expect decreasing exponential
                        fitval[2, sl, j, i] = 0  # 0 if no error
                    except RuntimeError:
                        error_count += 1
                        fitval[2, sl, j, i] = 0.5  # 0.5 if SNR above threshold but fit won't converge
                else:
                    fitval[2, sl, j, i] = 1  # 1 if below threshold SNR

    # Save results to file
    out_file = "DiffusionFits.bin"
    fitval[0,:,:,:].tofile("ADCMaps.bin")
    fitval[1,:,:,:].tofile("KurtosisMaps.bin")
    fitval[2,:,:,:].tofile("ErrorMaps.bin")
    print("Done!")

    return fitval


# ============= Analyze ROIs from masks ========
def ROI_analysis(mask_file, img_dims, index=None, dwi_fits=None, ADCMaps_file="ADCMaps.bin", KurtosisMaps_file="KurtosisMaps.bin"):

    slices = img_dims[0]
    yres = img_dims[2]
    xres = img_dims[3]

    mask_raw, mask_all = open_mask(mask_file, slices, yres, xres)
    nROI = np.size(mask_all, 0)

    if dwi_fits is None:
        ADC_map = np.reshape(np.fromfile(ADCMaps_file), (slices, yres, xres))
        BLM0_map = np.reshape(np.fromfile(KurtosisMaps_file), (slices, yres, xres))
    else:
        ADC_map = dwi_fits[0,:,:,:]
        BLM0_map = dwi_fits[1,:,:,:]

    if index is None:
        index = []
        for iROI in range(nROI):
            index.append("ROI " + str(iROI+1))

    ROI_ADC_all = np.zeros_like(mask_all)
    ROI_ADC_total = np.zeros((nROI))
    ROI_ADC_std = np.zeros((nROI))
    ROI_BLM0_all = np.zeros_like(mask_all)
    ROI_BLM0_total = np.zeros((nROI))
    ROI_BLM0_std = np.zeros((nROI))
    for iROI in range(nROI):
        ROI_ADC_all[iROI, :, :, :] = np.multiply(mask_all[iROI, :, :, :], ADC_map[:, :, :])
        ROI_BLM0_all[iROI, :, :, :] = np.multiply(mask_all[iROI, :, :, :], BLM0_map[:, :, :])
        ROI_ADC_total[iROI] = np.nanmean(np.multiply(mask_all[iROI, :, :, :], ADC_map[:, :, :]))
        ROI_ADC_std[iROI] = np.nanstd(np.multiply(mask_all[iROI, :, :, :], ADC_map[:, :, :]))
        ROI_BLM0_total[iROI] = np.nanmean(np.multiply(mask_all[iROI, :, :, :], BLM0_map[:, :, :]))
        ROI_BLM0_std[iROI] = np.nanstd(np.multiply(mask_all[iROI, :, :, :], BLM0_map[:, :, :]))
        print(f"{index[iROI]} : \n ADC: {ROI_ADC_total[iROI]*1000:.2f} +/- {ROI_ADC_std[iROI]*1000:.2f}   x10^-3 mm^2/s, \n KI : {ROI_BLM0_total[iROI]*100:.2f} +/- {ROI_BLM0_std[iROI]*100:.2f}   %" )

    return [ROI_ADC_all, ROI_ADC_total, ROI_BLM0_all, ROI_BLM0_total]


  

