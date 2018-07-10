#!/usr/bin/python3
# -*- coding: utf-8 -*-

from SciAnalysis.XSAnalysis.Data import *
from SciAnalysis.XSAnalysis import Protocols


class main_peak(Protocols.circular_average_q2I):

    def __init__(self, name='main_peak', **kwargs):

        self.name = self.__class__.__name__ if name is None else name

        self.default_ext = '.png'
        self.run_args = {'Iqn_n': 0.25,
                         'show_region': False,
                         }
        self.run_args.update(kwargs)

    @Protocols.run_default
    def run(self, data, output_dir, **run_args):

        results = {}

        # 2D data
        data.dezinger(tol=1e5)
        #data.plot(show=True)

        # 1D curve
        start = time.clock()
        line = data.circular_average_q_bin_parallel(error=False)
        print('parallel time', time.clock() - start)
        #line.plot(show=True)
        start = time.clock()
        line = data.circular_average_q_bin(error=True)
        print('bin time', time.clock() - start)
        #line.plot(show=True)


        #line.plot(show=True)
        new_results = self.analyze_q0(data, line, output_dir, **run_args)
        results.update(new_results)

        # Data along arc of main ring
        run_args['q0'] = new_results['q0']['value'] * 2.0
        run_args['dq'] = new_results['sigma_q0']['value'] * 3.0

        line = data.linecut_angle(**run_args)

        if 'show_region' in run_args and run_args['show_region']:
            data.plot(show=True)

        new_results = self.orientation_q0(data, line, output_dir, result_prepend='peak0_', **run_args)
        results.update(new_results)
        #print(results)

        return results


    def analyze_q0(self, data, line, output_dir, **run_args):

        results = {}

        line.y = line.y - np.min(line.y)
        line.y *= np.power(line.x, run_args['Iqn_n'])

        line.y_label = 'q^{{ {:.2f} }}*I(q)'.format(run_args['Iqn_n'])
        line.y_rlabel = '$q^{{ {:.2f} }} I(q) \, (\AA^{{- {:.2f} }} \mathrm{{counts/pixel}})$'.format(run_args['Iqn_n'],
                                                                                                      run_args['Iqn_n'])

        if 'q0' not in run_args:
            # Try to find a peak, in order to guess q0
            q0, I_q0 = line.target_y(np.max(line.y))
            run_args['q0'] = q0

        if 'dq' not in run_args:
            run_args['dq'] = q0 * 0.6

        sub_line = line.sub_range(run_args['q0'] - run_args['dq'], run_args['q0'] + run_args['dq'])
        lm_result, fit_line, fit_line_extended = self.fit_peak(sub_line, **run_args)

        q0 = lm_result.params['x_center'].value
        q0_err = lm_result.params['x_center'].stderr
        d0 = 2 * np.pi / (q0 * 10)
        d0_err = (d0 / q0) * q0_err

        # Results
        results['q0'] = {'value': lm_result.params['x_center'].value,
                         'units': 'A^-1',
                         'units_latex': r'\mathrm{ \AA }^{-1}',
                         'unit_type': 'inverse distance',
                         'error': lm_result.params['x_center'].stderr,
                         'symbol': 'q0',
                         'symbol_latex': 'q_0',
                         }
        results['d0'] = {'value': d0,
                         'units': 'nm',
                         'units_latex': r'\mathrm{nm}',
                         'unit_type': 'distance',
                         'error': d0_err,
                         'symbol': 'd0',
                         'symbol_latex': 'd_0',
                         }
        results['sigma_q0'] = {'value': lm_result.params['sigma'].value,
                               'units': 'A^-1',
                               'units_latex': r'\mathrm{ \AA }^{-1}',
                               'unit_type': 'inverse distance',
                               'error': lm_result.params['sigma'].stderr,
                               'symbol': 'sigma_q',
                               'symbol_latex': '\sigma_q',
                               }
        xi = 0.1 * np.sqrt(2 * np.pi) / lm_result.params['sigma'].value
        xi_err = 0.1 * np.sqrt(2 * np.pi) * lm_result.params['sigma'].stderr / np.square(
            lm_result.params['sigma'].value)
        results['xi_q0'] = {'value': xi,
                            'units': 'nm',
                            'units_latex': r'\mathrm{ nm }',
                            'unit_type': 'distance',
                            'error': xi_err,
                            'symbol': 'xi',
                            'symbol_latex': r'\xi',
                            }

        results['peak0_prefactor'] = {'value': lm_result.params['prefactor'].value, 'units': 'a.u.',
                                      'units_latex': r'a.u.', 'unit_type': 'a.u.',
                                      'error': lm_result.params['prefactor'].stderr, 'symbol': 'c',
                                      'symbol_latex': 'c', }
        results['peak0_background_m'] = {'value': lm_result.params['m'].value, 'units': 'a.u./(A^-1)',
                                         'units_latex': r'a.u./\mathrm{ \AA }^{-1}', 'unit_type': 'slope',
                                         'error': lm_result.params['m'].stderr, 'symbol': 'm', 'symbol_latex': 'm', }
        results['peak0_background_b'] = {'value': lm_result.params['b'].value, 'units': 'a.u.', 'units_latex': r'a.u.',
                                         'unit_type': 'a.u.', 'error': lm_result.params['b'].stderr, 'symbol': 'b',
                                         'symbol_latex': 'b', }

        results['peak0_chi_squared'] = lm_result.chisqr / lm_result.nfree

        text = r'$q_0 = {:.3g} \pm {:.2g} \, \mathrm{{ \AA }}^{{-1}}$'.format(q0, q0_err)
        text += '\n'
        text += r'$\sigma_q = {:.3g} \pm {:.2g} \, \mathrm{{ \AA }}^{{-1}}$'.format(lm_result.params['sigma'].value,
                                                                                    lm_result.params['sigma'].stderr)
        text += '\n'
        text += r'$d_0 = {:.3g} \pm {:.2g} \, \mathrm{{nm}}$'.format(d0, d0_err)
        text += '\n'
        text += r'$\xi_0 = {:.3g} \pm {:.2g} \, \mathrm{{nm}}$'.format(xi, xi_err)

        class DataLines_current(DataLines):
            def _plot_extra(self, **plot_args):
                xi, xf, yi, yf = self.ax.axis()
                plt.text(xf, yf, text, size=20, verticalalignment='top', horizontalalignment='right')

        lines = DataLines_current([line, fit_line, fit_line_extended])
        lines.copy_labels(line)

        outfile = self.get_outfile(data.name, output_dir, ext='_qnI{}'.format(self.default_ext))
        lines.plot(save=outfile, show=False, **run_args) #third graph

        outfile = self.get_outfile(data.name, output_dir, ext='_qnI.dat')
        line.save_data(outfile)

        return results

    def fit_peak(self, line, **run_args):

        # Usage: lm_result, fit_line, fit_line_extended = self.fit_peak(line, **run_args)

        import lmfit

        def model(v, x):
            '''Gaussian with linear background.'''
            m = v['prefactor'] * np.exp(-np.square(x - v['x_center']) / (2 * (v['sigma'] ** 2))) + v['m'] * x + v['b']
            return m

        def func2minimize(params, x, data):
            v = params.valuesdict()
            m = model(v, x)

            return m - data

        peak_x, peak_y = line.target_y(np.max(line.y))
        peak_y -= np.min(line.y)
        span = np.max(line.x) - np.min(line.x)

        params = lmfit.Parameters()
        params.add('prefactor', value=peak_y, min=0)
        params.add('x_center', value=peak_x, min=np.min(line.x) * 0.95, max=np.max(line.x) * 1.05)
        params.add('sigma', value=span * 0.3, min=0)
        params.add('m', value=np.min(line.y), min=0, max=np.max(line.y))
        params.add('b', value=0)

        lm_result = lmfit.minimize(func2minimize, params, args=(line.x, line.y))

        if run_args['verbosity'] >= 5:
            print('Fit results (lmfit):')
            lmfit.report_fit(lm_result.params)

        # fix_x = line.x
        # fit_y = line.y + lm_result.residual
        fit_x = np.linspace(np.min(line.x), np.max(line.x), num=200)
        fit_y = model(lm_result.params.valuesdict(), fit_x)

        fit_line = DataLine(x=fit_x, y=fit_y,
                            plot_args={'linestyle': '-', 'color': 'r', 'marker': None, 'linewidth': 4.0})

        x_span = abs(np.max(line.x) - np.min(line.x))
        xe = 0.5
        fit_x = np.linspace(np.min(line.x) - xe * x_span, np.max(line.x) + xe * x_span, num=200)
        fit_y = model(lm_result.params.valuesdict(), fit_x)
        fit_line_extended = DataLine(x=fit_x, y=fit_y,
                                     plot_args={'linestyle': '-', 'color': 'r', 'marker': None, 'linewidth': 2.0})

        return lm_result, fit_line, fit_line_extended

    def orientation_q0(self, data, line, output_dir, result_prepend='peak0_', **run_args):

        results = {}

        # Clean up the curve
        # line.remove_spurious(bins=5, tol=1.5)
        # line.smooth(1.0)
        for angle in [-180, -90, 0, +90, +180]:
            line.kill_x(angle, 1)

        outfile = self.get_outfile('{}_ori'.format(data.name), output_dir, ext='.dat', ir=False)
        line.save_data(outfile)

        labels = []

        symmetry = 2
        lm_result, fit_line = self.angle_fit(line, symmetry_assumed=symmetry, color='b', **run_args)
        line_list = [line, fit_line]

        results['{}ori{}_{}'.format(result_prepend, symmetry, 'eta')] = {
            'value': lm_result.params['eta'].value,
            'units': '',
            'units_latex': r'',
            'unit_type': 'unitless',
            'error': lm_result.params['eta'].stderr,
            'symbol': 'eta',
            'symbol_latex': r'\eta',
        }
        results['{}ori{}_{}'.format(result_prepend, symmetry, 'angle')] = {
            'value': lm_result.params['x_center'].value,
            'units': 'degrees',
            'units_latex': r'^{\circ}',
            'unit_type': 'angle',
            'error': lm_result.params['x_center'].stderr,
            'symbol': 'chi0',
            'symbol_latex': r'\chi_0',
        }
        results['{}ori{}_{}'.format(result_prepend, symmetry, 'prefactor')] = {
            'value': lm_result.params['prefactor'].value, 'units': 'a.u.', 'units_latex': r'a.u.', 'unit_type': 'a.u.',
            'error': lm_result.params['prefactor'].stderr, 'symbol': 'c', 'symbol_latex': 'c', }

        xp, yp = fit_line.target_x(lm_result.params['x_center'].value)
        text = '$\eta_{{ {} }} = {:.3g}$'.format(symmetry, lm_result.params['eta'].value)
        labels.append([xp, yp * 1.2, text, 'b'])
        text = '${:.3g} ^{{ \circ }}$'.format(lm_result.params['x_center'].value)
        labels.append([xp, 0, text, 'b'])

        if 'symmetry' in run_args and run_args['symmetry'] != symmetry:
            # Run analysis again assuming a different symmetry

            symmetry = run_args['symmetry']
            lm_result, fit_line = self.angle_fit(line, symmetry_assumed=symmetry, color='purple', **run_args)
            line_list.append(fit_line)

            results['{}ori{}_{}'.format(result_prepend, symmetry, 'eta')] = {
                'value': lm_result.params['eta'].value,
                'units': '',
                'units_latex': r'',
                'unit_type': 'unitless',
                'error': lm_result.params['eta'].stderr,
                'symbol': 'eta',
                'symbol_latex': r'\eta',
            }
            results['{}ori{}_{}'.format(result_prepend, symmetry, 'angle')] = {
                'value': lm_result.params['x_center'].value,
                'units': 'degrees',
                'units_latex': r'^{\circ}',
                'unit_type': 'angle',
                'error': lm_result.params['x_center'].stderr,
                'symbol': 'chi0',
                'symbol_latex': r'\chi_0',
            }
            results['{}ori{}_{}'.format(result_prepend, symmetry, 'prefactor')] = {
                'value': lm_result.params['prefactor'].value, 'units': 'a.u.', 'units_latex': r'a.u.',
                'unit_type': 'a.u.', 'error': lm_result.params['prefactor'].stderr, 'symbol': 'c',
                'symbol_latex': 'c', }

            xp, yp = fit_line.target_x(lm_result.params['x_center'].value)
            text = '$\eta_{{ {} }} = {:.3g}$'.format(symmetry, lm_result.params['eta'].value)
            labels.append([xp, yp * 1.2, text, 'purple'])
            text = '${:.3g} ^{{ \circ }}$'.format(lm_result.params['x_center'].value)
            labels.append([xp, yp * 0.1, text, 'purple'])

        class DataLines_current(DataLines):
            def _plot_extra(self, **plot_args):
                xi, xf, yi, yf = self.ax.axis()
                for x, y, text, color in labels:
                    self.ax.text(x, y, text, size=20, color=color, verticalalignment='bottom',
                                 horizontalalignment='left')
                    self.ax.axvline(x, color=color)

        lines = DataLines_current(line_list)
        lines.x_label = 'angle'
        lines.x_rlabel = r'$\chi \, (^{\circ})$'
        lines.y_label = 'I'
        lines.y_rlabel = r'$I(\chi) \, (\mathrm{counts/pixel})$'

        if run_args['verbosity'] >= 2:
            outfile = self.get_outfile('{}_ori'.format(data.name), output_dir, ext='.png', ir=False)
            plot_range = [-180, +180, 0, np.max(line.y) * 1.2]
            lines.plot(save=outfile, plot_range=plot_range) #first graph

        if run_args['verbosity'] >= 4:
            outfile = self.get_outfile('{}_ori_polar'.format(data.name), output_dir, ext='.png', ir=False)
            line.plot_polar(save=outfile) #second graph

        # TODO: Obtain order parameter from line
        # TODO: Add line.stats() to results

        return results

    def angle_fit(self, line, symmetry_assumed=2, color='b', **run_args):

        import lmfit

        def model(v, x):
            '''Eta orientation function.'''
            m = v['prefactor'] * (1 - (v['eta'] ** 2)) / (((1 + v['eta']) ** 2) - 4 * v['eta'] * (
                np.square(np.cos((symmetry_assumed / 2.0) * np.radians(x - v['x_center'])))))
            return m

        def func2minimize(params, x, data):
            v = params.valuesdict()
            m = model(v, x)

            return m - data

        params = lmfit.Parameters()
        x_peak, y_peak = line.target_y(np.max(line.y))
        params.add('prefactor', value=y_peak, min=0)
        params.add('x_center', value=self.reduce_angle(x_peak, symmetry_assumed), min=-180 / symmetry_assumed,
                   max=+180 / symmetry_assumed)
        params.add('eta', value=0.8, min=0, max=1)

        lm_result = lmfit.minimize(func2minimize, params, args=(line.x, line.y))

        if run_args['verbosity'] >= 5:
            print('Fit results (lmfit):')
            lmfit.report_fit(lm_result.params)

        fit_x = np.linspace(-180, +180, num=360, endpoint=True)
        fit_y = model(lm_result.params.valuesdict(), fit_x)

        fit_line = DataLineAngle(x=fit_x, y=fit_y,
                                 plot_args={'linestyle': '-', 'color': color, 'marker': None, 'linewidth': 4.0})

        return lm_result, fit_line

    def reduce_angle(self, angle, symmetry):
        '''Reduce an angle to be minimal for the given symmetry.'''

        span = 180.0 / symmetry
        # sym=1, repeats every 360deg, so span is -180 to +180
        # sym=2, repeats every 180deg, so span is -90 to +90
        # sym=4, repeats every 90deg, so span is -45 to +45
        # sym=6, repeats every 60deg, so span is -30 to +30

        while angle < span:
            angle += 2 * span
        while angle > span:
            angle -= 2 * span

        return angle


if True:
    root_dir = './'

    source_dir = '/Users/renuka_diwan/PycharmProjects/XSAnalysis/'
    output_dir = source_dir + 'analysis'


    import glob

    infiles = glob.glob(source_dir + 'Ag*53_saxs.npy')
    #infiles += glob.glob(source_dir + 'YT*.npy')
    #infiles = glob.glob(source_dir + '*.npy')

    print(infiles)

    infiles.sort()

    # Experimental setup
    calibration = Calibration()
    calibration.set_energy(8.8984)  # CHX
    #calibration.set_image_size(512, 512)
    calibration.set_image_size(619, 487)
    calibration.set_pixel_size(pixel_size_um=75.0)
    calibration.set_distance(0.5944)
    calibration.set_beam_position(150.18, 390.59)

    mask_dir = './'
    mask = Mask()
    mask.load(mask_dir + 'mask.png')

    load_args = {'calibration': calibration, 'mask': mask}
    run_args = {'verbosity': 4}

    process = Protocols.ProcessorXS(load_args=load_args, run_args=run_args)

    protocols = [main_peak()]

    process.run(infiles, protocols, output_dir=output_dir, force=True)

    print("Note: Reached the end of runXS.py")