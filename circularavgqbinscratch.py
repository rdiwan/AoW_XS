import multiprocessing
import numpy as np
from joblib import Parallel, delayed

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

        # start = time.clock()

        if self.mask is None:
            mask = np.ones(self.data.shape)
        else:
            mask = self.mask.data

        # .ravel() is used to convert the 2D grids into 1D arrays.
        # This is not strictly necessary, but improves speed somewhat.

        data = self.data.ravel()
        # print('data raveled', data)
        pixel_list = np.where(mask.ravel() == 1)  # Non-masked pixels
        # print ('pixel list:', pixel_list)

        # 2D map of the q-value associated with each pixel position in the detector image
        Q = self.calibration.q_map().ravel()
        # print("printing Q:")
        # print(Q)

        # delta-q associated with a single pixel
        dq = self.calibration.get_q_per_pixel()

        # lowest intensity and highest intensity
        x_range = [np.min(Q[pixel_list]), np.max(Q[pixel_list])]
        # print(x_range)

        bins = int(bins_relative * abs(x_range[1] - x_range[0]) / dq)  # averaging part
        # print(bins)

        num_per_bin, rbins = np.histogram(Q[pixel_list], bins=bins, range=x_range)
        # print(num_per_bin)
        # print(rbins)

        idx = np.where(num_per_bin != 0)  # Bins that actually have data

        if error:
            # TODO: Include error calculations

            x_vals, rbins = np.histogram(Q[pixel_list], bins=bins, range=x_range, weights=Q[pixel_list])

            # Create array of the average values (mu), in the original array layout
            locations = np.digitize(Q[pixel_list], bins=rbins, right=True)  # Mark the bin IDs in the original array layout
            mu = (x_vals / num_per_bin)[locations - 1]

            weights = np.square(Q[pixel_list] - mu)

            x_err, rbins = np.histogram(Q[pixel_list], bins=bins, range=x_range, weights=weights)
            x_err = np.sqrt(x_err[idx] / num_per_bin[idx])
            x_err[0] = dq / 2  # np.digitize includes all the values less than the minimum bin into the first element

            x_vals = x_vals[idx] / num_per_bin[idx]

            I_vals, rbins = np.histogram(Q[pixel_list], bins=bins, range=x_range, weights=data[pixel_list])
            I_err_shot = np.sqrt(I_vals)[idx] / num_per_bin[idx]

            mu = (I_vals / num_per_bin)[locations - 1]
            weights = np.square(data[pixel_list] - mu)
            I_err_std, rbins = np.histogram(Q[pixel_list], bins=bins, range=x_range, weights=weights)
            I_err_std = np.sqrt(I_err_std[idx] / num_per_bin[idx])

            y_err = np.sqrt(np.square(I_err_shot) + np.square(I_err_std))
            I_vals = I_vals[idx] / num_per_bin[idx]

            line = DataLine(x=x_vals, y=I_vals, x_err=x_err, y_err=y_err, x_label='q', y_label='I(q)',
                            x_rlabel='$q \, (\AA^{-1})$', y_rlabel=r'$I(q) \, (\mathrm{counts/pixel})$')


        else:
            x_vals, rbins = np.histogram(Q[pixel_list], bins=bins, range=x_range, weights=Q[pixel_list])
            x_vals = x_vals[idx] / num_per_bin[idx]
            I_vals, rbins = np.histogram(Q[pixel_list], bins=bins, range=x_range, weights=data[pixel_list])
            I_vals = I_vals[idx] / num_per_bin[idx]

            line = DataLine(x=x_vals, y=I_vals, x_label='q', y_label='I(q)', x_rlabel='$q \, (\AA^{-1})$',
                            y_rlabel=r'$I(q) \, (\mathrm{counts/pixel})$')

        # print(time.clock() - start)
        # line.plot(show=True)
        return line

    def circular_average_q_bin_parallel(self, bins_relative=1.0, error=False, **kwargs):
        num_cores = multiprocessing.cpu_count()

        data_split = np.array_split(self.data, num_cores)
        mask_split = np.array_split(self.mask, num_cores)
        qmap_split = np.array_split(self.calibration.q_map(), num_cores)
        dq = self.calibration.get_q_per_pixel()

        Parallel(n_jobs=num_cores)(delayed(analyze)(data_chunk = i, mask_chunk = j, q_chunk = q, dq = dq) for i in data_split for j in mask_split for q in qmap_split)


    def analyze(self, data_chunk, mask_chunk, q_chunk, dq):

        if mask_chunk is None:
            mask_chunk = np.ones(data_chunk.shape)

        data = data_chunk.ravel()
        pixel_list = np.where(mask_chunk.ravel() == 1)
        Q = q_chunk.ravel()
        x_range = [np.min(Q[pixel_list]), np.max(Q[pixel_list])]
        bins = int(bins_relative * abs(x_range[1] - x_range[0]) / dq)
        num_per_bin, rbins = np.histogram(Q[pixel_list], bins=bins, range=x_range)










