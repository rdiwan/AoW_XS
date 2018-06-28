#!/usr/bin/python
# -*- coding: utf-8 -*-
# vi: ts=4 sw=4
'''
:mod:`SciAnalysis.Data` - Base data objects for SciAnalysis
================================================
.. module:: SciAnalysis.Data
   :synopsis: Provides base classes for handling data
.. moduleauthor:: Dr. Kevin G. Yager <kyager@bnl.gov>
                    Brookhaven National Laboratory
'''

################################################################################
#  This code defines some baseline objects for handling data.
################################################################################
# Known Bugs:
#  N/A
################################################################################
# TODO:
#  Search for "TODO" below.
################################################################################


#import sys
import numpy as np
import pylab as plt
import matplotlib as mpl
from scipy import signal # For gaussian smoothing
from scipy import ndimage # For resize, etc.
from scipy import stats # For skew
#from scipy.optimize import leastsq
#import scipy.special

import PIL # Python Image Library (for opening PNG, etc.)

from . import tools





# DataLine
################################################################################
class DataLine(object):

    def __init__(self, infile=None, x=None, y=None, name=None, plot_args=None, **kwargs):

        if infile is None:
            self.x = x
            self.y = y
        else:
            self.load(infile, **kwargs)


        self.x_label = kwargs['x_label'] if 'x_label' in kwargs else 'x'
        self.y_label = kwargs['y_label'] if 'y_label' in kwargs else 'y'

        self.x_rlabel = kwargs['x_rlabel'] if 'x_rlabel' in kwargs else self.x_label
        self.y_rlabel = kwargs['y_rlabel'] if 'y_rlabel' in kwargs else self.y_label

        self.x_err = kwargs['x_err'] if 'x_err' in kwargs else None
        self.y_err = kwargs['y_err'] if 'y_err' in kwargs else None

        if name is not None:
            self.name = name
        elif infile is not None:
            self.name = tools.Filename(infile).get_filebase()
        else:
            self.name = None


        self.plot_valid_keys = ['color', 'linestyle', 'linewidth', 'marker', 'markerfacecolor', 'markersize', 'alpha', 'markeredgewidth', 'markeredgecolor', 'capsize', 'ecolor', 'elinewidth']
        self.plot_args = { 'color' : 'k',
                        'marker' : 'o',
                        'linewidth' : 3.0,
                        'rcParams': {'axes.labelsize': 35,
                                        'xtick.labelsize': 30,
                                        'ytick.labelsize': 30,
                                        },
                            }
        if plot_args: self.plot_args.update(plot_args)


    # Data loading
    ########################################

    def load(self, infile, format='auto', **kwargs):
        '''Loads data from the specified file.'''

        f = tools.Filename(infile)
        ext = f.get_ext()[1:]

        if format=='custom':
            x, y = self.load_custom(infile, **kwargs)
            self.x = x
            self.y = y

        elif format=='npy' or ext=='npy':
            data = np.load(infile)
            self.x = data[:,0]
            self.y = data[:,1]

        elif format in ['auto'] or ext in ['dat', 'txt']:
            data = np.loadtxt(infile)
            self.x = data[:,0]
            self.y = data[:,1]

        else:
            print("Couldn't identify data format for %s."%(infile))




    def copy_labels(self, line):
        '''Copies labels (x, y) from the supplied line into this line.'''

        self.x_label = line.x_label
        self.y_label = line.y_label
        self.x_rlabel = line.x_rlabel
        self.y_rlabel = line.y_rlabel


    # Data export
    ########################################

    def save_data(self, outfile):

        if self.x_err is None and self.y_err is None:
            data = np.dstack([self.x, self.y])[0]
            header = '%s %s' % (self.x_label, self.y_label)

        elif self.y_err is None:
            data = np.dstack([self.x, self.x_err, self.y])[0]
            header = '%s %serr %s' % (self.x_label, self.x_label, self.y_label)

        elif self.x_err is None:
            data = np.dstack([self.x, self.y, self.y_err])[0]
            header = '%s %s %serr' % (self.x_label, self.y_label, self.y_label)

        else:
            data = np.dstack([self.x, self.x_err, self.y, self.y_err])[0]
            header = '%s %serr %s %serr' % (self.x_label, self.x_label, self.y_label, self.y_label)

        np.savetxt( outfile, data, header=header )




    def sub_range(self, xi, xf):
        '''Returns a DataLine that only has a subset of the original x range.'''

        try:
            line = self.copy()
        except NotImplementedError:
            line = DataLine()
            line.x = self.x
            line.y = self.y

        line.trim(xi, xf)

        return line


    def target_x(self, target):
        '''Find the datapoint closest to the given x.'''

        self.sort_x()

        # Search through x for the target
        idx = np.where( self.x>=target )[0][0]
        xcur = self.x[idx]
        ycur = self.y[idx]

        return xcur, ycur


    def target_y(self, target):
        '''Find the datapoint closest to the given y.'''

        x = np.asarray(self.x)
        y = np.asarray(self.y)

        # Sort
        indices = np.argsort(y)
        x_sorted = x[indices]
        y_sorted = y[indices]

        # Search through y for the target
        idx = np.where( y_sorted>=target )[0][0]
        xcur = x_sorted[idx]
        ycur = y_sorted[idx]

        return xcur, ycur


    # Data modification
    ########################################
    def sort_x(self):
        '''Arrange (x,y) datapoints so that x is increasing.'''
        x = np.asarray(self.x)
        y = np.asarray(self.y)

        # Sort
        indices = np.argsort(x)
        self.x = x[indices]
        self.y = y[indices]

    def sort_y(self):
        x = np.asarray(self.x)
        y = np.asarray(self.y)

        # Sort
        indices = np.argsort(y)
        self.x = x[indices]
        self.y = y[indices]


    def trim(self, xi, xf):
        '''Reduces the data by trimming the x range.'''

        x = np.asarray(self.x)
        y = np.asarray(self.y)

        # Sort
        indices = np.argsort(x)
        x_sorted = x[indices]
        y_sorted = y[indices]

        if xi==None:
            idx_start = 0
        else:
            try:
                idx_start = np.where( x_sorted>xi )[0][0]
            except IndexError:
                idx_start = 0

        if xf==None:
            idx_end = len(x_sorted)
        else:
            try:
                idx_end = np.where( x_sorted>xf )[0][0]
            except IndexError:
                idx_end = len(x_sorted)

        self.x = x_sorted[idx_start:idx_end]
        self.y = y_sorted[idx_start:idx_end]


    def kill_x(self, x_center, x_spread):
        '''Removes some points from the line (within the specified range).'''

        x = np.asarray(self.x)
        y = np.asarray(self.y)

        # Sort
        indices = np.argsort(x)
        x_sorted = x[indices]
        y_sorted = y[indices]

        idx = np.where( abs(x_sorted-x_center)<x_spread )
        self.x = np.delete( x_sorted, idx )
        self.y = np.delete( y_sorted, idx )


    def remove_spurious(self, bins=5, tol=1e5):
        '''Remove data-points that deviate strongly from the curve.
        They are replaced with the local average.'''

        s = int(bins/2)
        for i, y in enumerate(self.y):

            sub_range = self.y[i-s:i+s]

            # average excluding point i
            avg = ( np.sum(self.y[i-s:i+s]) - y )/( len(sub_range) - 1 )

            if abs(y-avg)/avg>tol:
                self.y[i] = avg


    def smooth(self, sigma):

        self.y = ndimage.filters.gaussian_filter( self.y, sigma )




    # Data analysis
    ########################################
    def stats(self, prepend='stats_'):

        results = {}

        results[prepend+'max'] = np.max(self.y)
        results[prepend+'min'] = np.min(self.y)
        results[prepend+'average'] = np.average(self.y)
        results[prepend+'std'] = np.std(self.y)
        results[prepend+'N'] = len(self.y)
        results[prepend+'total'] = np.sum(self.y)

        results[prepend+'skew'] = stats.skew(self.y)

        results[prepend+'spread'] = results[prepend+'max'] - results[prepend+'min']
        results[prepend+'std_rel'] = results[prepend+'std'] / results[prepend+'average']

        zero_crossings = np.where(np.diff(np.signbit(self.y)))[0]
        results[prepend+'zero_crossings'] = len(zero_crossings)

        return results


    # Plotting
    ########################################

    def plot(self, save=None, show=False, plot_range=[None,None,None,None], plot_buffers=[0.2,0.05,0.2,0.05], **kwargs):
        '''Plots the scattering data.

        Parameters
        ----------
        save : str
            Set to 'None' to avoid saving to disk. Provide filename to save.
        show : bool
            Set to true to open an interactive window.
        plot_range : [float, float, float, float]
            Set the range of the plotting (None scales automatically instead).
        '''

        self._plot(save=save, show=show, plot_range=plot_range, plot_buffers=plot_buffers, **kwargs)


    def _plot(self, save=None, show=False, plot_range=[None,None,None,None], plot_buffers=[0.2,0.05,0.2,0.05], error=False, error_band=False, xlog=False, ylog=False, xticks=None, yticks=None, dashes=None, **kwargs):

        # DataLine._plot()

        plot_args = self.plot_args.copy()
        plot_args.update(kwargs)
        self.process_plot_args(**plot_args)

        self.fig = plt.figure( figsize=(10,7), facecolor='white' )
        left_buf, right_buf, bottom_buf, top_buf = plot_buffers
        fig_width = 1.0-right_buf-left_buf
        fig_height = 1.0-top_buf-bottom_buf
        self.ax = self.fig.add_axes( [left_buf, bottom_buf, fig_width, fig_height] )




        p_args = dict([(i, plot_args[i]) for i in self.plot_valid_keys if i in plot_args])
        self._plot_main(error=error, error_band=error_band, dashes=dashes, **p_args)


        plt.xlabel(self.x_rlabel)
        plt.ylabel(self.y_rlabel)

        if xlog:
            plt.semilogx()
        if ylog:
            plt.semilogy()
        if xticks is not None:
            self.ax.set_xticks(xticks)
        if yticks is not None:
            self.ax.set_yticks(yticks)


        # Axis scaling
        xi, xf, yi, yf = self.ax.axis()
        if plot_range[0] != None: xi = plot_range[0]
        if plot_range[1] != None: xf = plot_range[1]
        if plot_range[2] != None: yi = plot_range[2]
        if plot_range[3] != None: yf = plot_range[3]
        self.ax.axis( [xi, xf, yi, yf] )

        self._plot_extra(**plot_args)

        if save:
            if 'dpi' in plot_args:
                plt.savefig(save, dpi=plot_args['dpi'], transparent=True)
            else:
                plt.savefig(save, transparent=True)

        if show:
            self._plot_interact()
            plt.show()

        plt.close(self.fig.number)


    def _plot_main(self, error=False, error_band=False, dashes=None, **plot_args):

        if error_band:
            # TODO: Make this work
            l, = plt.plot(self.x, self.y, **plot_args)
            self.ax.fill_between(self.x, self.y-self.y_err, self.y+self.y_err, facecolor='0.8', linewidth=0)

        elif error:
            l = plt.errorbar( self.x, self.y, xerr=self.x_err, yerr=self.y_err, **plot_args)

        else:
            #l, = plt.plot(self.x, self.y, **plot_args)
            l, = self.ax.plot(self.x, self.y, **plot_args)


        if dashes is not None:
            l.set_dashes(dashes)


    def _plot_extra(self, **plot_args):
        '''This internal function can be over-ridden in order to force additional
        plotting behavior.'''

        pass



    def process_plot_args(self, **plot_args):

        if 'rcParams' in plot_args:
            for param, value in plot_args['rcParams'].items():
                plt.rcParams[param] = value



    # Plot interaction
    ########################################

    def _plot_interact(self):

        self.fig.canvas.set_window_title('SciAnalysis')
        #plt.get_current_fig_manager().toolbar.pan()
        #self.fig.canvas.toolbar.pan()
        self.fig.canvas.mpl_connect('scroll_event', self._scroll_event )
        #self.fig.canvas.mpl_connect('motion_notify_event', self._move_event )
        #self.fig.canvas.mpl_connect('key_press_event', self._key_press_event)

        #self.ax.format_coord = self._format_coord


    def _scroll_event(self, event):
        '''Gets called when the mousewheel/scroll-wheel is used. This activates
        zooming.'''

        if event.inaxes!=self.ax:
            return


        current_plot_limits = self.ax.axis()
        x = event.xdata
        y = event.ydata


        # The following function converts from the wheel-mouse steps
        # into a zoom-percentage. Using a base of 4 and a divisor of 2
        # means that each wheel-click is a 50% zoom. However, the speed
        # of zooming can be altered by changing these numbers.

        # 50% zoom:
        step_percent = 4.0**( -event.step/2.0 )
        # Fast zoom:
        #step_percent = 100.0**( -event.step/2.0 )
        # Slow zoom:
        #step_percent = 2.0**( -event.step/4.0 )

        xi = x - step_percent*(x-current_plot_limits[0])
        xf = x + step_percent*(current_plot_limits[1]-x)
        yi = y - step_percent*(y-current_plot_limits[2])
        yf = y + step_percent*(current_plot_limits[3]-y)

        self.ax.axis( (xi, xf, yi, yf) )

        self.fig.canvas.draw()


    # Object
    ########################################
    def copy(self):
        import copy
        return copy.deepcopy(self)


    # End class DataLine(object)
    ########################################



# DataLineAngle
################################################################################
class DataLineAngle (DataLine):

    def __init__(self, infile=None, x=None, y=None, name=None, plot_args=None, **kwargs):

        self.x = x
        self.y = y


        self.x_label = kwargs['x_label'] if 'x_label' in kwargs else 'angle (degrees)'
        self.y_label = kwargs['y_label'] if 'y_label' in kwargs else 'y'

        self.x_rlabel = kwargs['x_rlabel'] if 'x_rlabel' in kwargs else '$\chi \, (^{\circ})$'
        self.y_rlabel = kwargs['y_rlabel'] if 'y_rlabel' in kwargs else '$I(\chi)$'

        self.x_err = kwargs['x_err'] if 'x_err' in kwargs else None
        self.y_err = kwargs['y_err'] if 'y_err' in kwargs else None

        if name is not None:
            self.name = name
        elif infile is not None:
            self.name = tools.Filename(infile).get_filebase()
        else:
            self.name = None

        self.plot_valid_keys = ['color', 'linestyle', 'linewidth', 'marker', 'markerfacecolor', 'markersize', 'alpha', 'markeredgewidth', 'markeredgecolor', 'capsize', 'ecolor', 'elinewidth']

        self.plot_args = { 'color' : 'k',
                        'marker' : 'o',
                        'linewidth' : 3.0,
                        'rcParams': {'axes.labelsize': 35,
                                        'xtick.labelsize': 30,
                                        'ytick.labelsize': 30,
                                        },
                            }
        if plot_args: self.plot_args.update(plot_args)




    # Plotting
    ########################################

    def plot_polar(self, save=None, show=False, size=5, plot_buffers=[0.1,0.1,0.1,0.1], **kwargs):
        '''Plots the scattering data.

        Parameters
        ----------
        save : str
            Set to 'None' to avoid saving to disk. Provide filename to save.
        show : bool
            Set to true to open an interactive window.
        plot_range : [float, float, float, float]
            Set the range of the plotting (None scales automatically instead).
        '''

        self._plot_polar(save=save, show=show, size=size, plot_buffers=plot_buffers, **kwargs)


    def _plot_polar(self, save=None, show=False, size=5, plot_buffers=[0.2,0.2,0.2,0.2], assumed_symmetry=2, **kwargs):

        # TODO: Recast as part of plot_args
        #plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['axes.labelsize'] = 20
        plt.rcParams['xtick.labelsize'] = 15
        plt.rcParams['ytick.labelsize'] = 15


        self.fig = plt.figure( figsize=(size,size), facecolor='white' )
        left_buf, right_buf, bottom_buf, top_buf = plot_buffers
        fig_width = 1.0-right_buf-left_buf
        fig_height = 1.0-top_buf-bottom_buf
        self.ax = self.fig.add_axes( [left_buf, bottom_buf, fig_width, fig_height], polar=True )
        self.ax.set_theta_direction(-1)
        self.ax.set_theta_zero_location('N')

        plot_args = self.plot_args.copy()
        plot_args.update(kwargs)



        p_args = dict([(i, plot_args[i]) for i in self.plot_valid_keys if i in plot_args])
        self.ax.plot(np.radians(self.x), self.y, **p_args)
        #self.ax.fill_between(np.radians(self.x), 0, self.y, color='0.8')


        # Histogram of colors
        yh, xh = np.histogram(np.radians(self.x), 60, [-np.pi,+np.pi], weights=self.y)
        spacing = xh[1]-xh[0]
        yh = (yh/np.max(yh))*np.max(self.y)

        bins = len(yh)/assumed_symmetry
        color_list = cmap_cyclic_spectrum( np.linspace(0, 1.0, bins, endpoint=True) )

        #SHIFT
        #color_list = np.concatenate( (color_list[bins/2:], color_list[0:bins/2]) )

        color_list = np.concatenate( [color_list for i in range(assumed_symmetry)] )


        self.ax.bar(xh[:-1], yh, width=spacing*1.05, color=color_list, linewidth=0.0)


        self.ax.yaxis.set_ticklabels([])
        self.ax.xaxis.set_ticks([np.radians(angle) for angle in range(-180+45, 180+45, +45)])


        self._plot_extra_polar()

        if save:
            if 'dpi' in plot_args:
                plt.savefig(save, dpi=plot_args['dpi'], transparent=True)
            else:
                plt.savefig(save, transparent=True)

        if show:
            self._plot_interact()
            plt.show()

        plt.close(self.fig.number)


    def _plot_extra_polar(self, **plot_args):
        '''This internal function can be over-ridden in order to force additional
        plotting behavior.'''

        pass




    # End class DataLineAngle (DataLine)
    ########################################




# DataLines
################################################################################
class DataLines(DataLine):
    '''Holds multiple lines, so that they can be plotted together.'''

    def __init__(self, lines=[], plot_args=None, **kwargs):

        self.lines = lines

        self.x_label = kwargs['x_label'] if 'x_label' in kwargs else 'x'
        self.y_label = kwargs['y_label'] if 'y_label' in kwargs else 'y'

        self.x_rlabel = kwargs['x_rlabel'] if 'x_rlabel' in kwargs else self.x_label
        self.y_rlabel = kwargs['y_rlabel'] if 'y_rlabel' in kwargs else self.y_label

        self.plot_valid_keys = ['color', 'linestyle', 'linewidth', 'marker', 'markerfacecolor', 'markersize', 'alpha', 'markeredgewidth', 'markeredgecolor', 'capsize', 'ecolor', 'elinewidth']

        self.plot_args = { 'color' : 'k',
                        'marker' : 'o',
                        'linewidth' : 3.0,
                        'rcParams': {'axes.labelsize': 35,
                                        'xtick.labelsize': 30,
                                        'ytick.labelsize': 30,
                                        },
                            }
        if plot_args: self.plot_args.update(plot_args)



    def add_line(self, line):

        self.lines.append(line)


    def _plot_main(self, error=False, error_band=False, dashes=None, **plot_args):

        for line in self.lines:

            plot_args_current = {}
            plot_args_current.update(self.plot_args)
            plot_args_current.update(plot_args)
            plot_args_current.update(line.plot_args)

            p_args = dict([(i, plot_args_current[i]) for i in self.plot_valid_keys if i in plot_args_current])

            if error_band:
                l, = plt.plot(line.x, line.y, label=line.name, **p_args)
                self.ax.fill_between(line.x, line.y-line.y_err, line.y+line.y_err, facecolor='0.8', linewidth=0)

            elif error:
                l = plt.errorbar( line.x, line.y, xerr=line.x_err, yerr=line.y_err, label=line.name, **p_args)

            else:
                l, = plt.plot(line.x, line.y, label=line.name, **p_args)

            if dashes is not None:
                l.set_dashes(dashes)




    # End class DataLines(object)
    ########################################



# Data2D
################################################################################
class Data2D(object):


    def __init__(self, infile=None, format='auto', name=None, **kwargs):

        if name is not None:
            self.name = name
        elif infile is not None:
            self.name = tools.Filename(infile).get_filebase()

        if infile is not None:
            self.load(infile, format=format, **kwargs)

        self.x_label = kwargs['x_label'] if 'x_label' in kwargs else 'x'
        self.y_label = kwargs['y_label'] if 'y_label' in kwargs else 'y'

        self.x_rlabel = kwargs['x_rlabel'] if 'x_rlabel' in kwargs else self.x_label
        self.y_rlabel = kwargs['y_rlabel'] if 'y_rlabel' in kwargs else self.y_label


        self.x_scale = 1.0 # units/pixel
        self.y_scale = 1.0 # units/pixel
        if 'scale' in kwargs:
            self.x_scale = kwargs['scale'] # units/pixel
            self.y_scale = kwargs['scale'] # units/pixel


        self.set_z_display([None, None, 'linear', 1.0])
        self.plot_args = { 'rcParams': {'axes.labelsize': 40,
                                        'xtick.labelsize': 25,
                                        'ytick.labelsize': 25,
                                        },
                            }

        self.origin = [0, 0]

        self.regions = None # Optional overlay highlighting some region of interest

    # Data loading
    ########################################

    def load(self, infile, format='auto', **kwargs):
        '''Loads data from the specified file.'''

        f = tools.Filename(infile)
        ext = f.get_ext()[1:]

        if format=='image' or ext in ['png', 'tif', 'tiff', 'jpg', 'TIF']:
            self.load_image(infile)

        elif format=='npy' or ext=='npy':
            self.load_npy(infile)

        else:
            print("Couldn't identify data format for %s."%(infile))


        self.process_load_args(**kwargs)


    def load_image(self, infile):

        img = PIL.Image.open(infile).convert('I') # 'I' : 32-bit integer pixels
        self.data = np.asarray(img)
        del img


    def load_npy(self, infile, **kwargs):

        self.data = np.load(infile, **kwargs)


    def process_load_args(self, **kwargs):
        '''Follow the directives for the kwargs.'''

        if 'crop_left' in kwargs:
            self.data = self.data[:,kwargs['crop_left']:]
        if 'crop_right' in kwargs:
            self.data = self.data[:,:kwargs['crop_right']]
        if 'crop_top' in kwargs:
            self.data = self.data[kwargs['crop_top']:,:]
        if 'crop_bottom' in kwargs:
            self.data = self.data[:-kwargs['crop_bottom'],:]



    # Coordinate methods
    ########################################

    def get_origin(self):

        return self.origin


    def set_scale(self, scale):
        '''Conversion factor, in "units/pixel" for the image pixels into physical
        dimensions.'''

        # BUG: There is a conflict/inconsistency between the use of origin/scale vs. (x_axis, y_axis).

        self.x_scale = scale
        self.y_scale = scale


    def xy_axes(self):
        # BUG: There is a conflict/inconsistency between the use of origin/scale vs. (x_axis, y_axis).

        dim_y,dim_x = self.data.shape

        if self.origin[0] is None:
            x0 = dim_x/2.
        else:
            x0 = self.origin[0]
        if self.origin[1] is None:
            y0 = dim_y/2.
        else:
            y0 = self.origin[1]

        x_axis = (np.arange(dim_x) - x0)*self.x_scale
        y_axis = (np.arange(dim_y) - y0)*self.y_scale

        return x_axis, y_axis




    # Data analysis
    ########################################
    def stats(self, prepend='stats_'):

        results = {}

        results[prepend+'max'] = np.max(self.data)
        results[prepend+'min'] = np.min(self.data)
        results[prepend+'average'] = np.average(self.data)
        results[prepend+'std'] = np.std(self.data)
        results[prepend+'N'] = len(self.data.ravel())
        results[prepend+'total'] = np.sum(self.data.ravel())

        results[prepend+'skew'] = stats.skew(self.data.ravel())

        results[prepend+'spread'] = results[prepend+'max'] - results[prepend+'min']
        results[prepend+'std_rel'] = results[prepend+'std'] / results[prepend+'average']

        return results



    # Plotting
    ########################################

    def set_z_display(self, z_display):
        '''Controls how the z-values are converted into the false colormap.
        The provided array should have 4 elements. Example:
        [ 0, 10, 'gamma', 0.3]
         min max  mode    adjustment

        If min or max is set to 'None', then ztrim is used to pick values.
        mode can be:
          'linear'             adj ignored
          'log'                adj ignored
          'gamma'              adj is the log_gamma value
          'r'                  adj is the exponent

        'gamma' is a log-like gamma-correction function. 'adjustment' is the log_gamma value.
            log_gamma of 0.2 to 0.5 gives a nice 'logarithmic' response
            large values of log_gamma give a progressively more nearly response
            log_gamma = 2.0 gives a nearly linear response
            log_gamma < 0.2 give a very sharp response

        'r' multiplies the data by r**(adj), which can help to normalize data
        that decays away from a central origin.

        '''

        self.z_display = z_display





    def plot(self, save=None, show=False, ztrim=[0.01, 0.01], size=10.0, plot_buffers=[0.15,0.05,0.15,0.05], **kwargs):
        '''Plots the data.

        Parameters
        ----------
        save : str
            Set to 'None' to avoid saving to disk. Provide filename to save.
        show : bool
            Set to true to open an interactive window.
        ztrim : [float, float]
            Specify how to auto-set the z-scale. The floats indicate how much of
            the z-scale to 'trim' (relative units; i.e. 0.05 indicates 5%).
        '''

        self._plot(save=save, show=show, ztrim=ztrim, size=size, plot_buffers=plot_buffers, **kwargs)


    def _plot(self, save=None, show=False, ztrim=[0.01, 0.01], size=10.0, plot_buffers=[0.1,0.1,0.1,0.1], **kwargs):

        # Data2D._plot()

        plot_args = self.plot_args.copy()
        plot_args.update(kwargs)
        self.process_plot_args(**plot_args)


        self.fig = plt.figure( figsize=(size,size), facecolor='white' )
        left_buf, right_buf, bottom_buf, top_buf = plot_buffers
        fig_width = 1.0-right_buf-left_buf
        fig_height = 1.0-top_buf-bottom_buf
        self.ax = self.fig.add_axes( [left_buf, bottom_buf, fig_width, fig_height] )



        # Set zmin and zmax. Top priority is given to a kwarg to this plot function.
        # If that is not set, the value set for this object is used. If neither are
        # specified, a value is auto-selected using ztrim.

        values = np.sort( self.data.flatten() )
        if 'zmin' in plot_args and plot_args['zmin'] is not None:
            zmin = plot_args['zmin']
        elif self.z_display[0] is not None:
            zmin = self.z_display[0]
        else:
            zmin = values[ +int( len(values)*ztrim[0] ) ]

        if 'zmax' in plot_args and plot_args['zmax'] is not None:
            zmax = plot_args['zmax']
        elif self.z_display[1] is not None:
            zmax = self.z_display[1]
        else:
            idx = -int( len(values)*ztrim[1] )
            if idx>=0:
                idx = -1
            zmax = values[idx]

        if zmax==zmin:
            zmax = max(values)

        print( '        data: %.2f to %.2f\n        z-scaling: %.2f to %.2f\n' % (np.min(self.data), np.max(self.data), zmin, zmax) )

        self.z_display[0] = zmin
        self.z_display[1] = zmax
        self._plot_z_transform()


        shading = 'flat'
        #shading = 'gouraud'

        if 'cmap' in plot_args:
            cmap = plot_args['cmap']

        else:
            # http://matplotlib.org/examples/color/colormaps_reference.html
            #cmap = mpl.cm.RdBu
            #cmap = mpl.cm.RdBu_r
            #cmap = mpl.cm.hot
            #cmap = mpl.cm.gist_heat
            cmap = mpl.cm.jet

        x_axis, y_axis = self.xy_axes()
        extent = [x_axis[0], x_axis[-1], y_axis[0], y_axis[-1]]

        # TODO: Handle 'origin' correctly. (E.g. allow it to be set externally.)
        self.im = plt.imshow(self.Z, vmin=0, vmax=1, cmap=cmap, interpolation='nearest', extent=extent, origin='lower')
        #plt.pcolormesh( self.x_axis, self.y_axis, self.Z, cmap=cmap, vmin=zmin, vmax=zmax, shading=shading )

        if self.regions is not None:
            for region in self.regions:
                plt.imshow(region, cmap=mpl.cm.spring, interpolation='nearest', alpha=0.75)
                #plt.imshow(np.flipud(region), cmap=mpl.cm.spring, interpolation='nearest', alpha=0.75, origin='lower')

        x_label = self.x_rlabel if self.x_rlabel is not None else self.x_label
        y_label = self.y_rlabel if self.y_rlabel is not None else self.y_label
        plt.xlabel(x_label)
        plt.ylabel(y_label)

        if 'xticks' in kwargs and kwargs['xticks'] is not None:
            self.ax.set_xticks(kwargs['xticks'])
        if 'yticks' in kwargs and kwargs['yticks'] is not None:
            self.ax.set_yticks(kwargs['yticks'])


        if 'plot_range' in plot_args:
            plot_range = plot_args['plot_range']
            # Axis scaling
            xi, xf, yi, yf = self.ax.axis()
            if plot_range[0] != None: xi = plot_range[0]
            if plot_range[1] != None: xf = plot_range[1]
            if plot_range[2] != None: yi = plot_range[2]
            if plot_range[3] != None: yf = plot_range[3]
            self.ax.axis( [xi, xf, yi, yf] )

        if 'title' in plot_args:
            #size = plot_args['rcParams']['axes.labelsize']
            size = plot_args['rcParams']['xtick.labelsize']
            plt.figtext(0, 1, plot_args['title'], size=size, weight='bold', verticalalignment='top', horizontalalignment='left')

        self._plot_extra(**plot_args)

        if save:
            if 'transparent' not in plot_args:
                plot_args['transparent'] = True
            if 'dpi' in plot_args:
                plt.savefig(save, dpi=plot_args['dpi'], transparent=plot_args['transparent'])
            else:
                plt.savefig(save, transparent=plot_args['transparent'])

        if show:
            self._plot_interact()
            plt.show()

        plt.close(self.fig.number)


    def _plot_extra(self, **plot_args):
        '''This internal function can be over-ridden in order to force additional
        plotting behavior.'''

        pass


    def _plot_z_transform(self):
        '''Rescales the data according to the internal z_display setting.'''

        zmin, zmax, zmode, zadj = self.z_display

        if zmode=='log':
            #Z = np.log( (self.data-zmin)/(zmax-zmin) )

            #Z = np.log(self.data)/np.log(zmax)

            zmin = max(zmin,0.5)
            Z = (np.log(self.data)-np.log(zmin))/(np.log(zmax)-np.log(zmin))

        elif zmode=='gamma':
            log_gamma = zadj
            c = np.exp(1/log_gamma) - 1
            Z = (self.data-zmin)/(zmax-zmin)
            Z = log_gamma*np.log(Z*c + 1)

        elif zmode=='r':
            Z = self.data*np.power( self.r_map(), zadj )
            Z = (Z-zmin)/(zmax-zmin)

        elif zmode=='linear':
            Z = (self.data-zmin)/(zmax-zmin)

        else:
            print('Warning: z_display mode %s not recognized.'%(zmode))
            Z = (self.data-zmin)/(zmax-zmin)

        self.Z = np.nan_to_num(Z)


    def process_plot_args(self, **plot_args):

        if 'rcParams' in plot_args:
            for param, value in plot_args['rcParams'].items():
                plt.rcParams[param] = value



    # Plot interaction
    ########################################

    def _plot_interact(self):

        self.fig.canvas.set_window_title('SciAnalysis')
        #plt.get_current_fig_manager().toolbar.pan()
        #self.fig.canvas.toolbar.pan()
        self.fig.canvas.mpl_connect('scroll_event', self._scroll_event )
        #self.fig.canvas.mpl_connect('motion_notify_event', self._move_event )
        self.fig.canvas.mpl_connect('key_press_event', self._key_press_event)

        self.ax.format_coord = self._format_coord



    def _key_press_event(self, event):
        '''Gets called when a key is pressed when the plot is open.'''

        update = False

        if event.key == '[':
            self.z_display[3] *= 1.0/1.5
            update = True

        elif event.key == ']':
            self.z_display[3] *= 1.5
            update = True

        elif event.key == '-' or event.key=='_':
            self.z_display[1] *= 1.0/4.0
            update = True

        elif event.key == '+' or event.key=='=':
            self.z_display[1] *= 4.0
            update = True

        elif event.key == 'o':
            self.z_display[0] *= 1.0/4.0
            update = True

        elif event.key == 'p':
            if self.z_display[0]==0:
                self.z_display[0] = 1
            self.z_display[0] *= 4.0
            update = True

        elif event.key == 'm':
            if self.z_display[2]=='gamma':
                self.z_display[2] = 'linear'
            else:
                self.z_display[2] = 'gamma'
            update = True


        if update:
            #print( self.z_display)
            print('            zmin: %.1f, zmax: %.1f, %s (%.2f)'%(self.z_display[0], self.z_display[1], str(self.z_display[2]), self.z_display[3]))

            self._plot_z_transform()
            self.im.set_data(self.Z)
            self.fig.canvas.draw()


    def _scroll_event(self, event):
        '''Gets called when the mousewheel/scroll-wheel is used. This activates
        zooming.'''

        if event.inaxes!=self.ax:
            return


        current_plot_limits = self.ax.axis()
        x = event.xdata
        y = event.ydata


        # The following function converts from the wheel-mouse steps
        # into a zoom-percentage. Using a base of 4 and a divisor of 2
        # means that each wheel-click is a 50% zoom. However, the speed
        # of zooming can be altered by changing these numbers.

        # 50% zoom:
        step_percent = 4.0**( -event.step/2.0 )
        # Fast zoom:
        #step_percent = 100.0**( -event.step/2.0 )
        # Slow zoom:
        #step_percent = 2.0**( -event.step/4.0 )

        xi = x - step_percent*(x-current_plot_limits[0])
        xf = x + step_percent*(current_plot_limits[1]-x)
        yi = y - step_percent*(y-current_plot_limits[2])
        yf = y + step_percent*(current_plot_limits[3]-y)

        self.ax.axis( (xi, xf, yi, yf) )

        self.fig.canvas.draw()


    # Object
    ########################################
    def copy(self):
        import copy
        return copy.deepcopy(self)


    # End class Data2D(object)
    ########################################




# Custom colormaps
################################################################################
# ROYGBVR but with Cyan-Blue instead of Blue
color_list_cyclic_spectrum = [
    [ 1.0, 0.0, 0.0 ],
    [ 1.0, 165.0/255.0, 0.0 ],
    [ 1.0, 1.0, 0.0 ],
    [ 0.0, 1.0, 0.0 ],
    [ 0.0, 0.2, 1.0 ],
    [ 148.0/255.0, 0.0, 211.0/255.0 ],
    [ 1.0, 0.0, 0.0 ]
]
cmap_cyclic_spectrum = mpl.colors.LinearSegmentedColormap.from_list('cmap_cyclic_spectrum', color_list_cyclic_spectrum)

# classic jet, slightly tweaked
# (bears some similarity to mpl.cm.nipy_spectral)
color_list_jet_extended = [
    [0, 0, 0],
    [0.18, 0, 0.18],
    [0, 0, 0.5],
    [0, 0, 1],
    [ 0.        ,  0.38888889,  1.        ],
    [ 0.        ,  0.83333333,  1.        ],
    [ 0.3046595 ,  1.        ,  0.66308244],
    [ 0.66308244,  1.        ,  0.3046595 ],
    [ 1.        ,  0.90123457,  0.        ],
    [ 1.        ,  0.48971193,  0.        ],
    [ 1.        ,  0.0781893 ,  0.        ],
    [1, 0, 0],
    [ 0.5       ,  0.        ,  0.        ],
]
cmap_jet_extended = mpl.colors.LinearSegmentedColormap.from_list('cmap_jet_extended', color_list_jet_extended)

# Tweaked version of "view.gtk" default color scale
color_list_vge = [
    [ 0.0/255.0, 0.0/255.0, 0.0/255.0],
    [ 0.0/255.0, 0.0/255.0, 254.0/255.0],
    [ 188.0/255.0, 2.0/255.0, 107.0/255.0],
    [ 254.0/255.0, 55.0/255.0, 0.0/255.0],
    [ 254.0/255.0, 254.0/255.0, 0.0/255.0],
    [ 254.0/255.0, 254.0/255.0, 254.0/255.0]
]
cmap_vge = mpl.colors.LinearSegmentedColormap.from_list('cmap_vge', color_list_vge)

# High-dynamic-range (HDR) version of VGE
color_list_vge_hdr = [
    [ 255.0/255.0, 255.0/255.0, 255.0/255.0],
    [ 0.0/255.0, 0.0/255.0, 0.0/255.0],
    [ 0.0/255.0, 0.0/255.0, 255.0/255.0],
    [ 188.0/255.0, 0.0/255.0, 107.0/255.0],
    [ 254.0/255.0, 55.0/255.0, 0.0/255.0],
    [ 254.0/255.0, 254.0/255.0, 0.0/255.0],
    [ 254.0/255.0, 254.0/255.0, 254.0/255.0]
]
cmap_vge_hdr = mpl.colors.LinearSegmentedColormap.from_list('cmap_vge_hdr', color_list_vge_hdr)

# Simliar to Dectris ALBULA default color-scale
color_list_hdr_albula = [
    [ 255.0/255.0, 255.0/255.0, 255.0/255.0],
    [ 0.0/255.0, 0.0/255.0, 0.0/255.0],
    [ 255.0/255.0, 0.0/255.0, 0.0/255.0],
    [ 255.0/255.0, 255.0/255.0, 0.0/255.0],
    #[ 255.0/255.0, 255.0/255.0, 255.0/255.0],
]
cmap_hdr_albula = mpl.colors.LinearSegmentedColormap.from_list('cmap_hdr_albula', color_list_hdr_albula)

# Ugly color-scale, but good for highlighting many features in HDR data
color_list_cur_hdr_goldish = [
    [ 255.0/255.0, 255.0/255.0, 255.0/255.0], # white
    [ 0.0/255.0, 0.0/255.0, 0.0/255.0], # black
    [ 100.0/255.0, 127.0/255.0, 255.0/255.0], # light blue
    [ 0.0/255.0, 0.0/255.0, 127.0/255.0], # dark blue
    #[ 0.0/255.0, 127.0/255.0, 0.0/255.0], # dark green
    [ 127.0/255.0, 60.0/255.0, 0.0/255.0], # orange
    [ 255.0/255.0, 255.0/255.0, 0.0/255.0], # yellow
    [ 200.0/255.0, 0.0/255.0, 0.0/255.0], # red
    [ 255.0/255.0, 255.0/255.0, 255.0/255.0], # white
]
cmap_hdr_goldish = mpl.colors.LinearSegmentedColormap.from_list('cmap_hdr_goldish', color_list_cur_hdr_goldish)


# Ugly color-scale, but good for highlighting many features in HDR data
color_list_seismic_hdr = [
    [ 255.0/255.0, 255.0/255.0, 255.0/255.0],
    [ 0.0/255.0, 0.0/255.0, 0.0/255.0],
    [ 0.0/255.0, 0.0/255.0, 196.0/255.0],
    [ 255.0/255.0, 255.0/255.0, 255.0/255.0],
    #[ 255.0/255.0, 0.0/255.0, 0.0/255.0],
    [ 132.0/255.0, 0.0/255.0, 0.0/255.0],
]
cmap_hdr_seismic = mpl.colors.LinearSegmentedColormap.from_list('cmap_hdr_seismic', color_list_seismic_hdr)
