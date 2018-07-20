def circular_average_q_bin_parallel_trial1(self, bins_relative=1.0, error=False, **kwargs):
    start = time.clock()

    num_cores = multiprocessing.cpu_count()
    # num_cores = 4

    if self.mask is None:
        mask = np.ones(self.data.shape)
    else:
        mask = self.mask.data

    data = self.data.ravel()
    Q = self.calibration.q_map().ravel()

    start2 = time.clock()

    self.mask_split = np.array_split(mask, num_cores)
    self.q_split = np.array_split(Q, num_cores)

    # count_pixels = Parallel(n_jobs=num_cores, backend = 'threading')(delayed(self.analyze)(mask_chunk = np.array_split(mask, num_cores)[i],
    #                                                                q_chunk = np.array_split
    # (Q, num_cores)[i],
    #                                                                core = i) for i in range(num_cores))

    count_pixels = Parallel(n_jobs=num_cores, backend='threading')(
        delayed(self.analyze)(core=i) for i in range(num_cores))

    print('parallel count pixels time', time.clock() - start2)

    pixel_list = np.concatenate(count_pixels, axis=1)

    print('parallel count pixels and concat time', time.clock() - start2)

    x_range = [np.min(self.maxmin_array), np.max(self.maxmin_array)]

    dq = self.calibration.get_q_per_pixel()
    bins = int(bins_relative * abs(x_range[1] - x_range[0]) / dq)

    num_per_bin, rbins = np.histogram(Q[pixel_list], bins=bins, range=x_range)
    idx = np.where(num_per_bin != 0)

    print('parallel part 1 time', time.clock() - start)

    if error:
        # TODO: Include error calculations

        x_vals, rbins = np.histogram(Q[pixel_list], bins=bins, range=x_range, weights=Q[pixel_list])

        # Create array of the average values (mu), in the original array layout
        locations = np.digitize(Q[pixel_list], bins=rbins,
                                right=True)  # Mark the bin IDs in the original array layout
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

    return line


def circular_average_q_bin_parallel_trial2(self, bins_relative=1.0, error=False, **kwargs):
    start = time.clock()

    if self.mask is None:
        mask = np.ones(self.data.shape)
    else:
        mask = self.mask.data

    data = self.data.ravel()
    Q = self.calibration.q_map().ravel()

    num_cores = multiprocessing.cpu_count()

    self.mask_split = np.array_split(mask, num_cores)
    self.q_split = np.array_split(Q, num_cores)

    pool = Pool(processes=num_cores)

    count_pixels = np.empty(num_cores, dtype=[])

    for core in range(num_cores):
        count_pixels[core] = pool.apply_async(self.analyze, core)

    pool.close()
    pool.join()

    print(self.maxmin_array)

    x_range = [np.min(self.maxmin_array), np.max(self.maxmin_array)]

    pixel_list = np.concatenate(count_pixels, axis=1)

    dq = self.calibration.get_q_per_pixel()
    bins = int(bins_relative * abs(x_range[1] - x_range[0]) / dq)

    num_per_bin, rbins = np.histogram(Q[pixel_list], bins=bins, range=x_range)
    idx = np.where(num_per_bin != 0)

    print('parallel part 1 time', time.clock() - start)

    if error:
        # TODO: Include error calculations

        x_vals, rbins = np.histogram(Q[pixel_list], bins=bins, range=x_range, weights=Q[pixel_list])

        # Create array of the average values (mu), in the original array layout
        locations = np.digitize(Q[pixel_list], bins=rbins,
                                right=True)  # Mark the bin IDs in the original array layout
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

    return line


def circular_average_q_bin_parallel_trial3(self, bins_relative=1.0, error=False, **kwargs):
    start = time.clock()

    if self.mask is None:
        mask = np.ones(self.data.shape)
    else:
        mask = self.mask.data

    data = self.data.ravel()
    Q = self.calibration.q_map().ravel()

    num_cores = multiprocessing.cpu_count()

    self.mask_split = np.array_split(mask, num_cores)
    self.q_split = np.array_split(Q, num_cores)

    pool = Pool(processes=num_cores)

    count_pixels = np.empty(num_cores, dtype=[])

    for core in range(num_cores):
        count_pixels[core] = pool.apply_async(self.analyze, core)

    pool.close()
    pool.join()

    print(self.maxmin_array)

    x_range = [np.min(self.maxmin_array), np.max(self.maxmin_array)]

    pixel_list = np.concatenate(count_pixels, axis=1)

    dq = self.calibration.get_q_per_pixel()
    bins = int(bins_relative * abs(x_range[1] - x_range[0]) / dq)

    num_per_bin, rbins = np.histogram(Q[pixel_list], bins=bins, range=x_range)
    idx = np.where(num_per_bin != 0)

    print('parallel part 1 time', time.clock() - start)

    if error:

        # execute error1 and error2 processes at the same time and get x_vals, and I_vals from each

        line = DataLine(x=x_vals, y=I_vals, x_err=x_err, y_err=y_err, x_label='q', y_label='I(q)',
                        x_rlabel='$q \, (\AA^{-1})$', y_rlabel=r'$I(q) \, (\mathrm{counts/pixel})$')

    else:

        # execute else1 and else2 simultaneously and get x_vals and I_vals from each

        line = DataLine(x=x_vals, y=I_vals, x_label='q', y_label='I(q)', x_rlabel='$q \, (\AA^{-1})$',
                        y_rlabel=r'$I(q) \, (\mathrm{counts/pixel})$')

    return line


def error1(self):
    x_vals, rbins = np.histogram(Q[pixel_list], bins=bins, range=x_range, weights=Q[pixel_list])

    # Create array of the average values (mu), in the original array layout
    locations = np.digitize(Q[pixel_list], bins=rbins,
                            right=True)  # Mark the bin IDs in the original array layout
    mu = (x_vals / num_per_bin)[locations - 1]

    weights = np.square(Q[pixel_list] - mu)

    x_err, rbins = np.histogram(Q[pixel_list], bins=bins, range=x_range, weights=weights)
    x_err = np.sqrt(x_err[idx] / num_per_bin[idx])
    x_err[0] = dq / 2  # np.digitize includes all the values less than the minimum bin into the first element

    x_vals = x_vals[idx] / num_per_bin[idx]


def error2(self):
    I_vals, rbins = np.histogram(Q[pixel_list], bins=bins, range=x_range, weights=data[pixel_list])
    I_err_shot = np.sqrt(I_vals)[idx] / num_per_bin[idx]

    mu = (I_vals / num_per_bin)[locations - 1]
    weights = np.square(data[pixel_list] - mu)
    I_err_std, rbins = np.histogram(Q[pixel_list], bins=bins, range=x_range, weights=weights)
    I_err_std = np.sqrt(I_err_std[idx] / num_per_bin[idx])

    y_err = np.sqrt(np.square(I_err_shot) + np.square(I_err_std))
    I_vals = I_vals[idx] / num_per_bin[idx]


def else1(self):
    x_vals, rbins = np.histogram(Q[pixel_list], bins=bins, range=x_range, weights=Q[pixel_list])
    x_vals = x_vals[idx] / num_per_bin[idx]


def else2(self):
    I_vals, rbins = np.histogram(Q[pixel_list], bins=bins, range=x_range, weights=data[pixel_list])
    I_vals = I_vals[idx] / num_per_bin[idx]

# count_pixels = Parallel(n_jobs=num_cores, backend = 'threading')(delayed(self.analyze)(mask_chunk = np.array_split(mask, num_cores)[i],
#                                                                q_chunk = np.array_split
# (Q, num_cores)[i],
#                                                                core = i) for i in range(num_cores))