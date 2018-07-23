#!/usr/bin/python
# -*- coding: utf-8 -*-
# vi: ts=4 sw=4
'''
:mod:`SciAnalysis.XSAnalysis.Protocols` - Data analysis protocols
================================================
.. module:: SciAnalysis.XSAnalysis.Protocols
   :synopsis: Convenient protocols for data analysis.
.. moduleauthor:: Dr. Kevin G. Yager <kyager@bnl.gov>
                    Brookhaven National Laboratory
'''

################################################################################
#  Data analysis protocols.
################################################################################
# Known Bugs:
#  N/A
################################################################################
# TODO:
#  Search for "TODO" below.
################################################################################

from .Data import *
from ..tools import *


class ProcessorXS(Processor):


    def load(self, infile, **kwargs):

        data = Data2DScattering(infile, **kwargs)
        data.infile = infile

        data.threshold_pixels(4294967295-1) # Eiger inter-module gaps

        if 'dezing' in kwargs and kwargs['dezing']:
            data.dezinger()

        if 'flip' in kwargs and kwargs['flip']:
            #if flip: self.im = self.im.transpose(Image.ROTATE_90).transpose(Image.FLIP_LEFT_RIGHT)
            data.data = np.rot90(data.data) # rotate CCW
            data.data = np.fliplr(data.data) # Flip left/right

        if 'rotCCW' in kwargs and kwargs['rotCCW']:
            data.data = np.rot90(data.data) # rotate CCW

        if 'rot180' in kwargs and kwargs['rot180']:
            data.data = np.flipud(data.data) # Flip up/down
            data.data = np.fliplr(data.data) # Flip left/right

        if data.mask is not None:
            #data.data *= data.mask.data
            #np.multiply(data.data, data.mask.data)
            print("Note: No mask applied")


        return data



class circular_average(Protocol):

    def __init__(self, name=None, **kwargs):

        self.name = self.__class__.__name__ if name is None else name

        self.default_ext = '.png'
        self.run_args = {}
        self.run_args.update(kwargs)


    @run_default
    def run(self, data, output_dir, **run_args):

        results = {}

        if 'dezing' in run_args and run_args['dezing']:
            data.dezinger(sigma=3, tol=100, mode='median', mask=True, fill=False)


        line = data.circular_average_q_bin(error=True)
        #line.smooth(2.0, bins=10)

        outfile = self.get_outfile(data.name, output_dir)

        try:
            line.plot(save=outfile, show=False, **run_args)
        except ValueError:
            pass

        outfile = self.get_outfile(data.name, output_dir, ext='.dat')
        line.save_data(outfile)

        # TODO: Fit 1D data

        return results



class circular_average_q2I(Protocol):

    def __init__(self, name=None, **kwargs):

        self.name = self.__class__.__name__ if name is None else name

        self.default_ext = '.png'
        self.run_args = {}
        self.run_args.update(kwargs)


    @run_default
    def run(self, data, output_dir, **run_args):

        results = {}

        line = data.circular_average_q_bin(error=True)

        line.y *= np.square(line.x)
        line.y_label = 'q^2*I(q)'
        line.y_rlabel = '$q^2 I(q) \, (\AA^{-2} \mathrm{counts/pixel})$'


        outfile = self.get_outfile(data.name, output_dir, ext='_q2I{}'.format(self.default_ext))
        line.plot(save=outfile, show=False, **run_args)

        outfile = self.get_outfile(data.name, output_dir, ext='_q2I.dat')
        line.save_data(outfile)

        # TODO: Fit 1D data

        return results


    def output_exists(self, name, output_dir):

        if 'file_extension' in self.run_args:
            ext = '_q2I{}'.format(self.run_args['file_extension'])
        else:
            ext = '_q2I{}'.format(self.default_ext)

        outfile = self.get_outfile(name, output_dir, ext=ext)
        return os.path.isfile(outfile)



class linecut_qr(Protocol):

    def __init__(self, name='linecut_qr', **kwargs):

        self.name = self.__class__.__name__ if name is None else name

        self.default_ext = '.png'
        self.run_args = {'show_region' : False,
                         'plot_range' : [None, None, 0, None]
                         }
        self.run_args.update(kwargs)


    @run_default
    def run(self, data, output_dir, **run_args):

        results = {}

        line = data.linecut_qr(**run_args)

        if 'show_region' in run_args and run_args['show_region']:
            data.plot(show=True)


        #line.smooth(2.0, bins=10)

        outfile = self.get_outfile(data.name, output_dir)
        line.plot(save=outfile, **run_args)

        #outfile = self.get_outfile(data.name, output_dir, ext='_polar.png')
        #line.plot_polar(save=outfile, **run_args)

        outfile = self.get_outfile(data.name, output_dir, ext='.dat')
        line.save_data(outfile)

        return results







# Work in progress
################################################################################



class q_image(Protocol):

    def __init__(self, name='q_image', **kwargs):

        self.name = self.__class__.__name__ if name is None else name

        self.default_ext = '.png'
        self.run_args = {
                        'blur' : None,
                        'ztrim' : [0.05, 0.005],
                        'method' : 'nearest',
                        }
        self.run_args.update(kwargs)


    @run_default
    def run(self, data, output_dir, **run_args):

        results = {}

        if run_args['blur'] is not None:
            data.blur(run_args['blur'])

        q_data = data.remesh_q_bin(**run_args)

        if run_args['verbosity']>=10:
            # Diagnostic

            # WARNING: These outputs are not to be trusted.
            # The maps are oriented relative to data.data (not q_data.data)
            data_temp = Data2DReciprocal()

            data_temp.data = data.calibration.qx_map()
            outfile = self.get_outfile('qx-{}'.format(data.name), output_dir, ext='.png', ir=True)
            r = np.max( np.abs(data_temp.data) )
            data_temp.set_z_display([-r, +r, 'linear', 0.3])
            data_temp.plot(outfile, cmap='bwr', **run_args)

            data_temp.data = data.calibration.qy_map()
            outfile = self.get_outfile('qy-{}'.format(data.name), output_dir, ext='.png', ir=True)
            r = np.max( np.abs(data_temp.data) )
            data_temp.set_z_display([-r, +r, 'linear', 0.3])
            data_temp.plot(outfile, cmap='bwr', **run_args)

            data_temp.data = data.calibration.qz_map()
            outfile = self.get_outfile('qz-{}'.format(data.name), output_dir, ext='.png', ir=True)
            r = np.max( np.abs(data_temp.data) )
            data_temp.set_z_display([-r, +r, 'linear', 0.3])
            data_temp.plot(outfile, cmap='bwr', **run_args)



        if 'file_extension' in run_args and run_args['file_extension'] is not None:
            outfile = self.get_outfile(data.name, output_dir, ext=run_args['file_extension'])
        else:
            outfile = self.get_outfile(data.name, output_dir)

        if 'q_max' in run_args and run_args['q_max'] is not None:
            q_max = run_args['q_max']
            run_args['plot_range'] = [-q_max, +q_max, -q_max, +q_max]

        q_data.set_z_display([None, None, 'gamma', 0.3])
        q_data.plot_args = { 'rcParams': {'axes.labelsize': 55,
                                    'xtick.labelsize': 40,
                                    'ytick.labelsize': 40,
                                    'xtick.major.pad': 10,
                                    'ytick.major.pad': 10,
                                    },
                            }

        q_data.plot(outfile, plot_buffers=[0.30,0.05,0.25,0.05], **run_args)


        return results








