#!/usr/bin/python
# -*- coding: utf-8 -*-
# vi: ts=4 sw=4
'''
:mod:`SciAnalysis.XSAnalysis.Data` - Base objects for XSAnalysis
================================================
.. module:: SciAnalysis.XSAnalysis
   :synopsis: Provides base classes for doing analysis of x-ray scattering data
.. moduleauthor:: Dr. Kevin G. Yager <kyager@bnl.gov>
                    Brookhaven National Laboratory
'''

################################################################################
#  This code defines some baseline objects for x-ray analysis.
################################################################################
# Known Bugs:
#  N/A
################################################################################
# TODO:
#  Search for "TODO" below.
################################################################################



#import sys
import re # Regular expressions

import numpy as np
import pylab as plt
import matplotlib as mpl
#from scipy.optimize import leastsq
#import scipy.special
np.set_printoptions(threshold=np.nan)
import multiprocessing
from joblib import Parallel, delayed
from collections import Counter

import PIL # Python Image Library (for opening PNG, etc.)    

from .. import tools
from ..Data import *

import time



   
    
# Data2DScattering        
################################################################################    
class Data2DScattering(Data2D):
    '''Represents the data from a 2D (area) detector in a scattering measurement.'''


    
    def __init__(self, infile=None, format='auto', calibration=None, mask=None, name=None, **kwargs):
        '''Creates a new Data2D object, which stores a scattering area detector
        image.'''
        
        super(Data2DScattering, self).__init__(infile=None, **kwargs)
        
        self.set_z_display([None, None, 'gamma', 0.3])
        
        self.calibration = calibration
        self.mask = mask
        
        self.detector_data = None # Detector-specific object
        self.data = None # 2D data
        self.measure_time = 0.0
        self.maxmin_array =  None
        
        if name is not None:
            self.name = name
        elif infile is not None:
            if 'full_name' in kwargs and kwargs['full_name']:
                self.name = tools.Filename(infile).get_filename()
            else:
                self.name = tools.Filename(infile).get_filebase()            
                
        if infile is not None:
            self.load(infile, format=format)
        

    # Data loading
    ########################################

    def load(self, infile, format='auto', **kwargs):
        '''Loads data from the specified file.'''
        
        if format=='eiger' or infile[-10:]=='_master.h5':
            self.load_eiger(infile, **kwargs)
            
        elif format=='hdf5' or infile[-3:]=='.h5' or infile[-4:]=='.hd5':
            self.load_hdf5(infile)
            
        elif format=='tiff' or infile[-5:]=='.tiff' or infile[-4:]=='.tif':
            self.load_tiff(infile)
            
        elif format=='BrukerASCII' or infile[-6:]=='.ascii' or infile[-4:]=='.dat':
            self.load_BrukerASCII(infile)
            
        else:
            super(Data2DScattering, self).load(infile=infile, format=format, **kwargs)


        # Masking is now applied in the Processor's load method
        #if self.mask is not None:
            #self.data *= self.mask.data
            

        
        
    def load_tiff(self, infile):
        
        img = PIL.Image.open(infile).convert('I') # 'I' : 32-bit integer pixels
        self.data = np.copy( np.asarray(img) )
        del img
        

                
                
    # Coordinate methods
    ########################################
                    
    def get_origin(self):
        
        x0 = self.calibration.x0
        y0 = self.calibration.y0
        
        return x0, y0

        
    def _xy_axes(self):
        # TODO: test, and integrate if it makes sense
        
        dim_y,dim_x = self.data.shape
        
        x_axis = (np.arange(dim_x) - self.calibration.x0)*self.x_scale
        y_axis = (np.arange(dim_y) - self.calibration.y0)*self.y_scale
        
        return x_axis, y_axis
        
        
    # Data modification
    ########################################
        
    def threshold_pixels(self, threshold, new_value=0.0):
        
        self.data[self.data>threshold] = new_value
        
        
    def crop(self, size, shift_crop_up=0.0):
        '''Crop the data, centered about the q-origin. I.e. this throws away 
        some of the high-q information. The size specifies the size of the new
        image (as a fraction of the original full image width).
        
        shift_crop_up forces the crop to be off-center in the vertical (e.g. a
        value of 1.0 will shift it up so the q-origin is at the bottom of the
        image, which is nice for GISAXS).'''
        
        
        height, width = self.data.shape
        #self.data = self.data[ 0:height, 0:width ] # All the data
        
        x0, y0 = self.get_origin()
        xi = max( int(x0 - size*width/2), 0 )
        xf = min( int(x0 + size*width/2), width )
        yi = max( int(y0 - size*height*(0.5+0.5*shift_crop_up) ), 0 )
        yf = min( int(y0 + size*height*(0.5-0.5*shift_crop_up) ), height )
        
        
        self.data = self.data[ yi:yf, xi:xf ]
        
        
    def dezinger(self, sigma=3, tol=100, mode='median', mask=True, fill=False):
        # NOTE: This could probably be improved.
        
        if mode=='median':
            avg = ndimage.filters.median_filter(self.data, size=(sigma,sigma))
            variation = ndimage.filters.maximum_filter(avg, size=(sigma,sigma)) - ndimage.filters.minimum_filter(avg, size=(sigma,sigma))
            variation = np.where(variation > 1, variation, 1)
            idx = np.where( (self.data-avg)/variation > tol )
            
        elif mode=='gauss':
            # sigma=3, tol=1e5
            avg = ndimage.filters.gaussian_filter( self.data, sigma )
            local = avg - self.data/np.square(sigma)
            
            #dy, dx = np.gradient(self.data)
            #var = np.sqrt( np.square(dx) + np.square(dy) )
            
            idx = np.where( (self.data > avg) & (self.data > local) & (self.data-avg > tol)  )
        
        
        #self.data[idx] = 0
        if fill:
            self.data[idx] = avg[idx]
            
        if mask:
            self.mask.data[idx] = 0




    def circular_average_q_bin(self, bins_relative=1.0, error=False, **kwargs):
        '''Returns a 1D curve that is a circular average of the 2D data. The
        data is average over 'chi', so that the resulting curve is as a function
        of q.
        
        'bins_relative' controls the binning (q-spacing in data).
            1.0 means the q-spacing is (approximately) a single pixel
            2.0 means there are twice as many bins (spacing is half a pixel)
            0.1 means there are one-tenth the number of bins (i.e. each data point is 10 pixels)
            
        'error' sets whether or not error-bars are calculated.
        '''
        
        # This version uses numpy.histogram instead of converting the binning
        # into an equivalent integer list (and then using numpy.bincount).
        # This code is slightly slower (30-40% longer to run. However, this
        # version is slightly more general inasmuch as one can be more
        # arbitrary about the number of bins to be used (doesn't have to match
        # the q-spacing).

        #start = time.clock()

        if self.mask is None:
            mask = np.ones(self.data.shape)
        else:
            mask = self.mask.data
            
        # .ravel() is used to convert the 2D grids into 1D arrays.
        # This is not strictly necessary, but improves speed somewhat.
        
        data = self.data.ravel()
        #print('data raveled', data)
        pixel_list = np.where(mask.ravel()==1) # Non-masked pixels
        #print ('pixel list:', pixel_list)

        #2D map of the q-value associated with each pixel position in the detector image
        Q = self.calibration.q_map().ravel()
        #print("printing Q:")
        #print(Q)

        #delta-q associated with a single pixel
        dq = self.calibration.get_q_per_pixel()

        #lowest intensity and highest intensity
        x_range = [np.min(Q[pixel_list]), np.max(Q[pixel_list])]
        #print(x_range)

        bins = int( bins_relative * abs(x_range[1]-x_range[0])/dq ) #averaging part
        #print(bins)

        num_per_bin, rbins = np.histogram(Q[pixel_list], bins=bins, range=x_range)
        #print(num_per_bin)
        #print(rbins)

        idx = np.where(num_per_bin!=0) # Bins that actually have data


        if error:
            # TODO: Include error calculations
            
            x_vals, rbins = np.histogram( Q[pixel_list], bins=bins, range=x_range, weights=Q[pixel_list] )
            
            # Create array of the average values (mu), in the original array layout
            locations = np.digitize( Q[pixel_list], bins=rbins, right=True) # Mark the bin IDs in the original array layout
            mu = (x_vals/num_per_bin)[locations-1]
            
            weights = np.square(Q[pixel_list] - mu)
            
            x_err, rbins = np.histogram( Q[pixel_list], bins=bins, range=x_range, weights=weights )
            x_err = np.sqrt( x_err[idx]/num_per_bin[idx] )
            x_err[0] = dq/2 # np.digitize includes all the values less than the minimum bin into the first element
            
            x_vals = x_vals[idx]/num_per_bin[idx]
            
            I_vals, rbins = np.histogram( Q[pixel_list], bins=bins, range=x_range, weights=data[pixel_list] )
            I_err_shot = np.sqrt(I_vals)[idx]/num_per_bin[idx]
            
            mu = (I_vals/num_per_bin)[locations-1]
            weights = np.square(data[pixel_list] - mu)
            I_err_std, rbins = np.histogram( Q[pixel_list], bins=bins, range=x_range, weights=weights )
            I_err_std = np.sqrt( I_err_std[idx]/num_per_bin[idx] )
                
            y_err = np.sqrt( np.square(I_err_shot) + np.square(I_err_std) )
            I_vals = I_vals[idx]/num_per_bin[idx]
            
            line = DataLine( x=x_vals, y=I_vals, x_err=x_err, y_err=y_err, x_label='q', y_label='I(q)', x_rlabel='$q \, (\AA^{-1})$', y_rlabel=r'$I(q) \, (\mathrm{counts/pixel})$' )
            
            
        else:
            x_vals, rbins = np.histogram( Q[pixel_list], bins=bins, range=x_range, weights=Q[pixel_list] )
            x_vals = x_vals[idx]/num_per_bin[idx]
            I_vals, rbins = np.histogram( Q[pixel_list], bins=bins, range=x_range, weights=data[pixel_list] )
            I_vals = I_vals[idx]/num_per_bin[idx]
            
            line = DataLine( x=x_vals, y=I_vals, x_label='q', y_label='I(q)', x_rlabel='$q \, (\AA^{-1})$', y_rlabel=r'$I(q) \, (\mathrm{counts/pixel})$' )

        #print(time.clock() - start)
        #line.plot(show=True)
        return line

    def circular_average_q_bin_parallel(self, bins_relative=1.0, error=False, **kwargs):
        num_cores = multiprocessing.cpu_count()
        self.maxmin_array = np.array(num_cores*2)

        data = self.data.ravel()

        data_split = np.array_split(self.data, num_cores)
        mask_split = np.array_split(self.mask.data, num_cores)
        qmap_split = np.array_split(self.calibration.q_map(), num_cores)
        dq = self.calibration.get_q_per_pixel()

        count_pixels = Parallel(n_jobs=num_cores)(delayed(self.analyze)(data_chunk = i, mask_chunk = j, q_chunk = q) for i in data_split for j in mask_split for q in qmap_split)

        max = np.max(self.maxmin_array)
        min = np.min(self.maxmin_array)
        x_range = [min, max]
        bins = int(bins_relative * abs(max-min) / dq)
        #combined_dicts = Counter({})

        #for dictionary in count_dict:
            #combined_dicts += dictionary

        pixel_list = []

        for p_list in count_pixels:
            pixel_list += [p_list]

        Q = self.calibration.q_map().ravel()

        num_per_bin, rbins = np.histogram(Q[pixel_list], bins=bins, range=x_range)
            # print(num_per_bin)
            # print(rbins)

        idx = np.where(num_per_bin != 0)  # Bins that actually have data

        x_vals, rbins = np.histogram(Q[pixel_list], bins=bins, range=x_range, weights=Q[pixel_list])
        x_vals = x_vals[idx] / num_per_bin[idx]
        I_vals, rbins = np.histogram(Q[pixel_list], bins=bins, range=x_range, weights=data[pixel_list])
        I_vals = I_vals[idx] / num_per_bin[idx]


        #num_per_bin, rbins = np.histogram(combined_dicts.keys(), bins=bins, range=x_range)
        #idx = np.where(num_per_bin != 0)

        #x_vals, rbins = np.histogram(combined_dicts.keys(), bins=bins, range=x_range, weights=combined_dicts.keys())
        #x_vals = x_vals[idx] / num_per_bin[idx]

        #PROBLEM
        #I_vals, rbins = np.histogram(combined_dicts.keys(), bins=bins, range=x_range, weights=data[pixel_list])
        #I_vals = I_vals[idx] / num_per_bin[idx]

        line = DataLine(x=x_vals, y=I_vals, x_label='q', y_label='I(q)', x_rlabel='$q \, (\AA^{-1})$',
                        y_rlabel=r'$I(q) \, (\mathrm{counts/pixel})$')

        return line



    def analyze(self, data_chunk, mask_chunk, q_chunk):

        if mask_chunk is None:
            mask_chunk = np.ones(data_chunk.shape)

        data = data_chunk.ravel()
        pixel_list = np.where(mask_chunk.ravel() == 1)
        Q = q_chunk.ravel()
        np.append(self.maxmin_array, [np.min(Q[pixel_list]), np.max(Q[pixel_list])])
        q_dict = {}

        #for p in pixel_list:
        #    q_dict['Q[p]'] += 1

        #return Counter(q_dict)

        return pixel_list






    def linecut_angle(self, q0, dq, x_label='angle', x_rlabel='$\chi \, (^{\circ})$', y_label='I', y_rlabel=r'$I (\chi) \, (\mathrm{counts/pixel})$', **kwargs):
        '''Returns the intensity integrated along a ring of constant q.'''

        #if self.mask is None:
            #mask = np.ones(self.data.shape)
        #else:
            #mask = self.mask.data

        mask = np.ones(self.data.shape)

        data = self.data.ravel()

        pixel_list = np.where( (abs(self.calibration.q_map().ravel()-q0)<dq) & (mask.ravel()==1) )


        if 'show_region' in kwargs and kwargs['show_region']:
            region = np.ma.masked_where(abs(self.calibration.q_map()-q0)>dq, self.calibration.angle_map())
            self.regions = [region]

        #Q = self.calibration.q_map().ravel()
        dq = self.calibration.get_q_per_pixel()

        # Generate map
        M = self.calibration.angle_map().ravel()
        scale = np.degrees( np.abs(np.arctan(1.0/(q0/dq))) ) # approximately 1-pixel

        Md = (M/scale + 0.5).astype(int) # Simplify the distances to closest integers
        Md -= np.min(Md)

        num_per_m = np.bincount(Md[pixel_list])
        idx = np.where(num_per_m!=0) # distances that actually have data

        x_vals = np.bincount( Md[pixel_list], weights=M[pixel_list] )[idx]/num_per_m[idx]
        I_vals = np.bincount( Md[pixel_list], weights=data[pixel_list] )[idx]/num_per_m[idx]

        line = DataLineAngle( x=x_vals, y=I_vals, x_label=x_label, y_label=y_label, x_rlabel=x_rlabel, y_rlabel=y_rlabel )

        #line.plot(show=True)

        return line

        

    # Plotting
    ########################################

    def plot(self, save=None, show=False, ztrim=[0.02, 0.01], **kwargs):

        super(Data2DScattering, self).plot(save=save, show=show, ztrim=ztrim, **kwargs)
        

    # Plot interaction
    ########################################

    def _format_coord(self, x, y):
        
        h, w = self.data.shape
        
        #xp = (x-self.calibration.x0)/self.x_scale
        #yp = (y-self.calibration.y0)/self.y_scale
        xp = (x)/self.x_scale
        yp = (h-y)/self.y_scale
        
        col = int(xp+0.5)
        row = int(yp+0.5 - 1)
        
        numrows, numcols = self.data.shape
        if col>=0 and col<numcols and row>=0 and row<numrows:
            z = self.data[row,col]
            #z = self.Z[row,col]
            #return 'x=%1.1f, y=%1.1f, z=%1.1f'%(x, y, z)
            return 'x=%g, y=%g, z=%g'%(x, y, z)
        else:
            return 'x=%g, y=%g'%(x, y)        

    
    # End class Data2DScattering(Data2D)
    ########################################





# Mask
################################################################################    
class Mask(object):
    '''Stores the matrix of pixels to be excluded from further analysis.'''
    
    def __init__(self, infile=None, format='auto'):
        '''Creates a new mask object, storing a matrix of the pixels to be 
        excluded from further analysis.'''
        
        self.data = None
        
        if infile is not None:
            self.load(infile, format=format)
        
        
    def load(self, infile, format='auto', invert=False):
        '''Loads a mask from a a file. If this object already has some masking
        defined, then the new mask is 'added' to it. Thus, one can load multiple
        masks to exlude various pixels.'''
        
        if format=='png' or infile[-4:]=='.png':
            self.load_png(infile, invert=invert)
            
        elif format=='hdf5' or infile[-3:]=='.h5' or infile[-4:]=='.hd5':
            self.load_hdf5(infile, invert=invert)
            
        else:
            print("Couldn't identify mask format for %s."%(infile))
            
            
    def load_blank(self, width, height):
        '''Creates a null mask; i.e. one that doesn't exlude any pixels.'''
        
        # TODO: Confirm that this is the correct order for x and y.
        self.data = np.ones((height, width))
        
            
    def load_png(self, infile, threshold=127, invert=False):
        '''Load a mask from a PNG image file. High values (white) are included, 
        low values (black) are exluded.'''
        
        # Image should be black (0) for excluded pixels, white (255) for included pixels
        img = PIL.Image.open(infile).convert("L") # black-and-white
        img2 = img.point(lambda p: p > threshold and 255)
        data = np.asarray(img2)/255
        data = data.astype(int)
        
        if invert:
            data = -1*(data-1)
        
        if self.data is None:
            self.data = data
        else:
            self.data *= data
        
        
    def load_hdf5(self, infile, invert=False):
        
        with h5py.File(infile, 'r') as f:
            data = np.asarray( f['mask'] )

        if invert:
            data = -1*(data-1)
        
        if self.data is None:
            self.data = data
        else:
            self.data *= data

        
    def invert(self):
        '''Inverts the mask. Can be used if the mask file was written using the
        opposite convention.'''
        self.data = -1*(self.data-1)


    # End class Mask(object)
    ########################################
    
    
    
    
    
# Calibration
################################################################################    
class Calibration(object):
    '''Stores aspects of the experimental setup; especially the calibration
    parameters for a particular detector. That is, the wavelength, detector
    distance, and pixel size that are needed to convert pixel (x,y) into
    reciprocal-space (q) value.
    
    This class may also store other information about the experimental setup
    (such as beam size and beam divergence).
    '''
    
    def __init__(self, wavelength_A=None, distance_m=None, pixel_size_um=None):
        
        self.wavelength_A = wavelength_A
        self.distance_m = distance_m
        self.pixel_size_um = pixel_size_um
        
        self.sample_normal = None
        
        
        # Data structures will be generated as needed
        # (and preserved to speedup repeated calculations)
        self.clear_maps()
    
    
    # Experimental parameters
    ########################################
    
    def set_wavelength(self, wavelength_A):
        '''Set the experimental x-ray wavelength (in Angstroms).'''
        
        self.wavelength_A = wavelength_A
    
    
    def get_wavelength(self):
        '''Get the x-ray beam wavelength (in Angstroms) for this setup.'''
        
        return self.wavelength_A
    
        
    def set_energy(self, energy_keV):
        '''Set the experimental x-ray beam energy (in keV).'''
        
        energy_eV = energy_keV*1000.0
        energy_J = energy_eV/6.24150974e18
        
        h = 6.626068e-34 # m^2 kg / s
        c = 299792458 # m/s
        
        wavelength_m = (h*c)/energy_J
        self.wavelength_A = wavelength_m*1e+10
    
    
    def get_energy(self):
        '''Get the x-ray beam energy (in keV) for this setup.'''
        
        h = 6.626068e-34 # m^2 kg / s
        c = 299792458 # m/s
        
        wavelength_m = self.wavelength_A*1e-10 # m
        E = h*c/wavelength_m # Joules
        
        E *= 6.24150974e18 # electron volts
        
        E /= 1000.0 # keV
        
        return E
    
    
    def get_k(self):
        '''Get k = 2*pi/lambda for this setup, in units of inverse Angstroms.'''
        
        return 2.0*np.pi/self.wavelength_A
    
    
    def set_distance(self, distance_m):
        '''Sets the experimental detector distance (in meters).'''
        
        self.distance_m = distance_m
        
    
    def set_pixel_size(self, pixel_size_um=None, width_mm=None, num_pixels=None):
        '''Sets the pixel size (in microns) for the detector. Pixels are assumed
        to be square.'''
        
        if pixel_size_um is not None:
            self.pixel_size_um = pixel_size_um
            
        else:
            if num_pixels is None:
                num_pixels = self.width
            pixel_size_mm = width_mm*1./num_pixels
            self.pixel_size_um = pixel_size_mm*1000.0
        
        
    def set_beam_position(self, x0, y0):
        '''Sets the direct beam position in the detector images (in pixel 
        coordinates).'''
        
        self.x0 = x0
        self.y0 = y0
        
        
    def set_image_size(self, width, height=None):
        '''Sets the size of the detector image, in pixels.'''
        
        self.width = width
        if height is None:
            # Assume a square detector
            self.height = width
        else:
            self.height = height


    def get_width(self):
        return self.width

    def get_height(self):
        return self.height
    
    
    def get_q_per_pixel(self):
        '''Gets the delta-q associated with a single pixel. This is computed in
        the small-angle limit, so it should only be considered approximate.
        For instance, wide-angle detectors will have different delta-q across
        the detector face.'''
        
        if self.q_per_pixel is not None:
            return self.q_per_pixel
        
        c = (self.pixel_size_um/1e6)/self.distance_m
        twotheta = np.arctan(c) # radians
        
        self.q_per_pixel = 2.0*self.get_k()*np.sin(twotheta/2.0)
        
        return self.q_per_pixel
    
    def set_angles(self, sample_normal=0):
        self.sample_normal = sample_normal
    
    
    # Convenience methods
    ########################################
    def q_to_angle(self, q):
        '''Convert from q to angle (full scattering angle, 2theta, in degrees).'''
        kpre = 2.0*self.get_k()
        return np.degrees( 2.0*np.arcsin(q/kpre) )
    
    def angle_to_q(self, angle):
        '''Convert from scattering angle (full scattering angle, in degrees)
        to q-value (in inverse angstroms).'''
        kpre = 2.0*self.get_k()
        return kpre*np.sin(np.radians(angle/2))    
    
    
    # Maps
    ########################################
    
    def clear_maps(self):
        self.r_map_data = None
        self.q_per_pixel = None
        self.q_map_data = None
        self.angle_map_data = None
        
        self.qx_map_data = None
        self.qy_map_data = None
        self.qz_map_data = None
        self.qr_map_data = None

    
    def r_map(self):
        '''Returns a 2D map of the distance from the origin (in pixel units) for
        each pixel position in the detector image.'''
        
        if self.r_map_data is not None:
            return self.r_map_data

        x = np.arange(self.width) - self.x0
        y = np.arange(self.height) - self.y0
        X, Y = np.meshgrid(x, y)
        R = np.sqrt(X**2 + Y**2)
        
        self.r_map_data = R
        
        return self.r_map_data
        
    
    def q_map(self):
        '''Returns a 2D map of the q-value associated with each pixel position
        in the detector image.'''

        if self.q_map_data is not None:
            return self.q_map_data
        
        c = (self.pixel_size_um/1e6)/self.distance_m
        twotheta = np.arctan(self.r_map()*c) # radians
        
        self.q_map_data = 2.0*self.get_k()*np.sin(twotheta/2.0)
        
        return self.q_map_data
        
    
    def angle_map(self):
        '''Returns a map of the angle for each pixel (w.r.t. origin).
        0 degrees is vertical, +90 degrees is right, -90 degrees is left.'''

        if self.angle_map_data is not None:
            return self.angle_map_data

        x = (np.arange(self.width) - self.x0)
        y = (np.arange(self.height) - self.y0)
        X,Y = np.meshgrid(x,y)
        #M = np.degrees(np.arctan2(Y, X))
        # Note intentional inversion of the usual (x,y) convention.
        # This is so that 0 degrees is vertical.
        #M = np.degrees(np.arctan2(X, Y))

        # TODO: Lookup some internal parameter to determine direction
        # of normal. (This is what should befine the angle convention.)
        M = np.degrees(np.arctan2(X, -Y))


        self.angle_map_data = M

        if self.sample_normal is not None:
            self.angle_map_data += self.sample_normal


        return self.angle_map_data


    
    
    # End class Calibration(object)
    ########################################
    
    
