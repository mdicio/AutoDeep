import os
import pickle as cp
import shutil
import time
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from scipy.stats import rankdata, spearmanr
from sklearn.preprocessing import MinMaxScaler


def drop_numerical_outliers(df, z_thresh=3):
    """drop_numerical_outliers

    Args:
    df : type
        Description
    z_thresh : type
        Description

    Returns:
        type: Description
    """
    constrains = (
        df.select_dtypes(include=[np.number])
        .apply(lambda x: np.abs(stats.zscore(x)) < z_thresh)
        .all(axis=1)
    )
    df.drop(df.index[~constrains])
    return df


def min_max_transform(data: pd.DataFrame, exclude_cols: List, feature_range=(0, 255)):
    """min_max_transform

    Args:
    data : type
        Description
    exclude_cols : type
        Description
    feature_range : type
        Description

    Returns:
        type: Description
    """
    numcols = list(set.difference(set(data.columns), set(exclude_cols)))
    scaler = MinMaxScaler(feature_range=feature_range)
    data[numcols] = scaler.fit_transform(data[numcols])
    return data


def generate_feature_distance_ranking(data, method="Pearson"):
    """generate_feature_distance_ranking

    Args:
    data : type
        Description
    method : type
        Description

    Returns:
        type: Description
    """
    num = data.shape[1]
    if method == "Pearson":
        corr = np.corrcoef(np.transpose(data))
    elif method == "Spearman":
        corr = spearmanr(data).correlation
    elif method == "Euclidean":
        corr = squareform(pdist(np.transpose(data), metric="euclidean"))
        corr = np.max(corr) - corr
        corr = corr / np.max(corr)
    elif method == "set":
        corr1 = np.dot(np.transpose(data), data)
        corr2 = data.shape[0] - np.dot(np.transpose(1 - data), 1 - data)
        corr = corr1 / corr2
    corr = 1 - corr
    corr = np.around(a=corr, decimals=10)
    tril_id = np.tril_indices(num, k=-1)
    rank = rankdata(corr[tril_id])
    ranking = np.zeros((num, num))
    ranking[tril_id] = rank
    ranking = ranking + np.transpose(ranking)
    return ranking, corr


def generate_matrix_distance_ranking(num_r, num_c, method="Euclidean"):
    """generate_matrix_distance_ranking

    Args:
    num_r : type
        Description
    num_c : type
        Description
    method : type
        Description

    Returns:
        type: Description
    """
    for r in range(num_r):
        if r == 0:
            coordinate = np.transpose(np.vstack((np.zeros(num_c), range(num_c))))
        else:
            coordinate = np.vstack(
                (
                    coordinate,
                    np.transpose(np.vstack((np.ones(num_c) * r, range(num_c)))),
                )
            )
    num = num_r * num_c
    cord_dist = np.zeros((num, num))
    if method == "Euclidean":
        for i in range(num):
            cord_dist[i, :] = np.sqrt(
                np.square(coordinate[i, 0] * np.ones(num) - coordinate[:, 0])
                + np.square(coordinate[i, 1] * np.ones(num) - coordinate[:, 1])
            )
    elif method == "Manhattan":
        for i in range(num):
            cord_dist[i, :] = np.abs(
                coordinate[i, 0] * np.ones(num) - coordinate[:, 0]
            ) + np.abs(coordinate[i, 1] * np.ones(num) - coordinate[:, 1])
    tril_id = np.tril_indices(num, k=-1)
    rank = rankdata(cord_dist[tril_id])
    ranking = np.zeros((num, num))
    ranking[tril_id] = rank
    ranking = ranking + np.transpose(ranking)
    coordinate = np.int64(coordinate)
    return (coordinate[:, 0], coordinate[:, 1]), ranking


def IGTD_absolute_error(
    source,
    target,
    max_step=1000,
    switch_t=0,
    val_step=50,
    min_gain=1e-05,
    random_state=1,
    save_folder=None,
    file_name="",
    print_every_nsteps=100,
):
    """IGTD_absolute_error

    Args:
    source : type
        Description
    target : type
        Description
    max_step : type
        Description
    switch_t : type
        Description
    val_step : type
        Description
    min_gain : type
        Description
    random_state : type
        Description
    save_folder : type
        Description
    file_name : type
        Description
    print_every_nsteps : type
        Description

    Returns:
        type: Description
    """
    np.random.RandomState(seed=random_state)
    if os.path.exists(save_folder):
        shutil.rmtree(save_folder)
    os.mkdir(save_folder)
    source = source.copy()
    num = source.shape[0]
    tril_id = np.tril_indices(num, k=-1)
    index = np.array(range(num))
    index_record = np.empty((max_step + 1, num))
    index_record.fill(np.nan)
    index_record[0, :] = index.copy()
    err_v = np.empty(num)
    err_v.fill(np.nan)
    for i in range(num):
        err_v[i] = np.sum(np.abs(source[i, 0:i] - target[i, 0:i])) + np.sum(
            np.abs(source[i + 1 :, i] - target[i + 1 :, i])
        )
    step_record = -np.ones(num)
    err_record = [np.sum(abs(source[tril_id] - target[tril_id]))]
    pre_err = err_record[0]
    t1 = time.time()
    run_time = [0]
    for s in range(max_step):
        delta = np.ones(num) * np.inf
        idr = np.where(step_record == np.min(step_record))[0]
        ii = idr[np.random.permutation(len(idr))[0]]
        for jj in range(num):
            if jj == ii:
                continue
            if ii < jj:
                i = ii
                j = jj
            else:
                i = jj
                j = ii
            err_ori = err_v[i] + err_v[j] - np.abs(source[j, i] - target[j, i])
            err_i = (
                np.sum(np.abs(source[j, :i] - target[i, :i]))
                + np.sum(np.abs(source[i + 1 : j, j] - target[i + 1 : j, i]))
                + np.sum(np.abs(source[j + 1 :, j] - target[j + 1 :, i]))
                + np.abs(source[i, j] - target[j, i])
            )
            err_j = (
                np.sum(np.abs(source[i, :i] - target[j, :i]))
                + np.sum(np.abs(source[i, i + 1 : j] - target[j, i + 1 : j]))
                + np.sum(np.abs(source[j + 1 :, i] - target[j + 1 :, j]))
                + np.abs(source[i, j] - target[j, i])
            )
            err_test = err_i + err_j - np.abs(source[i, j] - target[j, i])
            delta[jj] = err_test - err_ori
        delta_norm = delta / pre_err
        id = np.where(delta_norm <= switch_t)[0]
        if len(id) > 0:
            jj = np.argmin(delta)
            if ii < jj:
                i = ii
                j = jj
            else:
                i = jj
                j = ii
            for k in range(num):
                if k < i:
                    err_v[k] = (
                        err_v[k]
                        - np.abs(source[i, k] - target[i, k])
                        - np.abs(source[j, k] - target[j, k])
                        + np.abs(source[j, k] - target[i, k])
                        + np.abs(source[i, k] - target[j, k])
                    )
                elif k == i:
                    err_v[k] = (
                        np.sum(np.abs(source[j, :i] - target[i, :i]))
                        + np.sum(np.abs(source[i + 1 : j, j] - target[i + 1 : j, i]))
                        + np.sum(np.abs(source[j + 1 :, j] - target[j + 1 :, i]))
                        + np.abs(source[i, j] - target[j, i])
                    )
                elif k < j:
                    err_v[k] = (
                        err_v[k]
                        - np.abs(source[k, i] - target[k, i])
                        - np.abs(source[j, k] - target[j, k])
                        + np.abs(source[k, j] - target[k, i])
                        + np.abs(source[i, k] - target[j, k])
                    )
                elif k == j:
                    err_v[k] = (
                        np.sum(np.abs(source[i, :i] - target[j, :i]))
                        + np.sum(np.abs(source[i, i + 1 : j] - target[j, i + 1 : j]))
                        + np.sum(np.abs(source[j + 1 :, i] - target[j + 1 :, j]))
                        + np.abs(source[i, j] - target[j, i])
                    )
                else:
                    err_v[k] = (
                        err_v[k]
                        - np.abs(source[k, i] - target[k, i])
                        - np.abs(source[k, j] - target[k, j])
                        + np.abs(source[k, j] - target[k, i])
                        + np.abs(source[k, i] - target[k, j])
                    )
            ii_v = source[ii, :].copy()
            jj_v = source[jj, :].copy()
            source[ii, :] = jj_v
            source[jj, :] = ii_v
            ii_v = source[:, ii].copy()
            jj_v = source[:, jj].copy()
            source[:, ii] = jj_v
            source[:, jj] = ii_v
            err = delta[jj] + pre_err
            t = index[ii]
            index[ii] = index[jj]
            index[jj] = t
            step_record[ii] = s
            step_record[jj] = s
        else:
            err = pre_err
            step_record[ii] = s
        err_record.append(err)
        if s % print_every_nsteps == 0:
            print("Step " + str(s) + " err: " + str(err))
        index_record[s + 1, :] = index.copy()
        run_time.append(time.time() - t1)
        if s > val_step:
            if (
                np.sum(
                    (err_record[-val_step - 1] - np.array(err_record[-val_step:]))
                    / err_record[-val_step - 1]
                    >= min_gain
                )
                == 0
            ):
                break
        pre_err = err
    index_record = index_record[: len(err_record), :].astype(np.int64)
    print(save_folder)
    if save_folder is not None:
        pd.DataFrame(index_record).to_csv(
            save_folder + "/" + file_name + "_index.txt",
            header=False,
            index=False,
            sep="\t",
        )
        pd.DataFrame(
            np.transpose(np.vstack((err_record, np.array(range(s + 2))))),
            columns=["error", "steps"],
        ).to_csv(
            save_folder + "/" + file_name + "_error_and_step.txt",
            header=True,
            index=False,
            sep="\t",
        )
        pd.DataFrame(
            np.transpose(np.vstack((err_record, run_time))),
            columns=["error", "run_time"],
        ).to_csv(
            save_folder + "/" + file_name + "_error_and_time.txt",
            header=True,
            index=False,
            sep="\t",
        )
    return index_record, err_record, run_time


def IGTD_square_error(
    source,
    target,
    max_step=1000,
    switch_t=0,
    val_step=50,
    min_gain=1e-05,
    random_state=1,
    save_folder=None,
    file_name="",
    print_every_nsteps=100,
):
    """IGTD_square_error

    Args:
    source : type
        Description
    target : type
        Description
    max_step : type
        Description
    switch_t : type
        Description
    val_step : type
        Description
    min_gain : type
        Description
    random_state : type
        Description
    save_folder : type
        Description
    file_name : type
        Description
    print_every_nsteps : type
        Description

    Returns:
        type: Description
    """
    np.random.RandomState(seed=random_state)
    if os.path.exists(save_folder):
        shutil.rmtree(save_folder)
    os.mkdir(save_folder)
    source = source.copy()
    num = source.shape[0]
    tril_id = np.tril_indices(num, k=-1)
    index = np.array(range(num))
    index_record = np.empty((max_step + 1, num))
    index_record.fill(np.nan)
    index_record[0, :] = index.copy()
    err_v = np.empty(num)
    err_v.fill(np.nan)
    for i in range(num):
        err_v[i] = np.sum(np.square(source[i, 0:i] - target[i, 0:i])) + np.sum(
            np.square(source[i + 1 :, i] - target[i + 1 :, i])
        )
    step_record = -np.ones(num)
    err_record = [np.sum(np.square(source[tril_id] - target[tril_id]))]
    pre_err = err_record[0]
    t1 = time.time()
    run_time = [0]
    for s in range(max_step):
        delta = np.ones(num) * np.inf
        idr = np.where(step_record == np.min(step_record))[0]
        ii = idr[np.random.permutation(len(idr))[0]]
        for jj in range(num):
            if jj == ii:
                continue
            if ii < jj:
                i = ii
                j = jj
            else:
                i = jj
                j = ii
            err_ori = err_v[i] + err_v[j] - np.square(source[j, i] - target[j, i])
            err_i = (
                np.sum(np.square(source[j, :i] - target[i, :i]))
                + np.sum(np.square(source[i + 1 : j, j] - target[i + 1 : j, i]))
                + np.sum(np.square(source[j + 1 :, j] - target[j + 1 :, i]))
                + np.square(source[i, j] - target[j, i])
            )
            err_j = (
                np.sum(np.square(source[i, :i] - target[j, :i]))
                + np.sum(np.square(source[i, i + 1 : j] - target[j, i + 1 : j]))
                + np.sum(np.square(source[j + 1 :, i] - target[j + 1 :, j]))
                + np.square(source[i, j] - target[j, i])
            )
            err_test = err_i + err_j - np.square(source[i, j] - target[j, i])
            delta[jj] = err_test - err_ori
        delta_norm = delta / pre_err
        id = np.where(delta_norm <= switch_t)[0]
        if len(id) > 0:
            jj = np.argmin(delta)
            if ii < jj:
                i = ii
                j = jj
            else:
                i = jj
                j = ii
            for k in range(num):
                if k < i:
                    err_v[k] = (
                        err_v[k]
                        - np.square(source[i, k] - target[i, k])
                        - np.square(source[j, k] - target[j, k])
                        + np.square(source[j, k] - target[i, k])
                        + np.square(source[i, k] - target[j, k])
                    )
                elif k == i:
                    err_v[k] = (
                        np.sum(np.square(source[j, :i] - target[i, :i]))
                        + np.sum(np.square(source[i + 1 : j, j] - target[i + 1 : j, i]))
                        + np.sum(np.square(source[j + 1 :, j] - target[j + 1 :, i]))
                        + np.square(source[i, j] - target[j, i])
                    )
                elif k < j:
                    err_v[k] = (
                        err_v[k]
                        - np.square(source[k, i] - target[k, i])
                        - np.square(source[j, k] - target[j, k])
                        + np.square(source[k, j] - target[k, i])
                        + np.square(source[i, k] - target[j, k])
                    )
                elif k == j:
                    err_v[k] = (
                        np.sum(np.square(source[i, :i] - target[j, :i]))
                        + np.sum(np.square(source[i, i + 1 : j] - target[j, i + 1 : j]))
                        + np.sum(np.square(source[j + 1 :, i] - target[j + 1 :, j]))
                        + np.square(source[i, j] - target[j, i])
                    )
                else:
                    err_v[k] = (
                        err_v[k]
                        - np.square(source[k, i] - target[k, i])
                        - np.square(source[k, j] - target[k, j])
                        + np.square(source[k, j] - target[k, i])
                        + np.square(source[k, i] - target[k, j])
                    )
            ii_v = source[ii, :].copy()
            jj_v = source[jj, :].copy()
            source[ii, :] = jj_v
            source[jj, :] = ii_v
            ii_v = source[:, ii].copy()
            jj_v = source[:, jj].copy()
            source[:, ii] = jj_v
            source[:, jj] = ii_v
            err = delta[jj] + pre_err
            t = index[ii]
            index[ii] = index[jj]
            index[jj] = t
            step_record[ii] = s
            step_record[jj] = s
        else:
            err = pre_err
            step_record[ii] = s
        err_record.append(err)
        if s % print_every_nsteps == 0:
            print("Step " + str(s) + " err: " + str(err))
        index_record[s + 1, :] = index.copy()
        run_time.append(time.time() - t1)
        if s > val_step:
            if (
                np.sum(
                    (err_record[-val_step - 1] - np.array(err_record[-val_step:]))
                    / err_record[-val_step - 1]
                    >= min_gain
                )
                == 0
            ):
                break
        pre_err = err
    index_record = index_record[: len(err_record), :].astype(np.int64)
    print(f"SAVE FOLDER {save_folder}")
    if save_folder is not None:
        pd.DataFrame(index_record).to_csv(
            save_folder + "/" + file_name + "_index.txt",
            header=False,
            index=False,
            sep="\t",
        )
        pd.DataFrame(
            np.transpose(np.vstack((err_record, np.array(range(s + 2))))),
            columns=["error", "steps"],
        ).to_csv(
            save_folder + "/" + file_name + "_error_and_step.txt",
            header=True,
            index=False,
            sep="\t",
        )
        pd.DataFrame(
            np.transpose(np.vstack((err_record, run_time))),
            columns=["error", "run_time"],
        ).to_csv(
            save_folder + "/" + file_name + "_error_and_time.txt",
            header=True,
            index=False,
            sep="\t",
        )
    return index_record, err_record, run_time


def IGTD(
    source,
    target,
    err_measure="abs",
    max_step=1000,
    switch_t=0,
    val_step=50,
    min_gain=1e-05,
    random_state=1,
    save_folder=None,
    file_name="",
):
    """IGTD

    Args:
    source : type
        Description
    target : type
        Description
    err_measure : type
        Description
    max_step : type
        Description
    switch_t : type
        Description
    val_step : type
        Description
    min_gain : type
        Description
    random_state : type
        Description
    save_folder : type
        Description
    file_name : type
        Description

    Returns:
        type: Description
    """
    if err_measure == "abs":
        index_record, err_record, run_time = IGTD_absolute_error(
            source=source,
            target=target,
            max_step=max_step,
            switch_t=switch_t,
            val_step=val_step,
            min_gain=min_gain,
            random_state=random_state,
            save_folder=save_folder,
            file_name=file_name,
        )
    if err_measure == "squared":
        index_record, err_record, run_time = IGTD_square_error(
            source=source,
            target=target,
            max_step=max_step,
            switch_t=switch_t,
            val_step=val_step,
            min_gain=min_gain,
            random_state=random_state,
            save_folder=save_folder,
            file_name=file_name,
        )
    return index_record, err_record, run_time


def generate_image_data(
    data,
    index,
    img_rows,
    img_columns,
    coord,
    save_mode="normal",
    save_pngs=False,
    image_folder=None,
    file_name="",
    exclude_cols=[],
):
    """generate_image_data

    Args:
    data : type
        Description
    index : type
        Description
    img_rows : type
        Description
    img_columns : type
        Description
    coord : type
        Description
    save_mode : type
        Description
    save_pngs : type
        Description
    image_folder : type
        Description
    file_name : type
        Description
    exclude_cols : type
        Description

    Returns:
        type: Description
    """
    t0 = time.time()
    if os.path.exists(image_folder):
        shutil.rmtree(image_folder)
    os.mkdir(image_folder)
    if save_mode == "bulk":
        image_data = None
        samples = None
    else:
        if isinstance(data, pd.DataFrame):
            samples = data.index.map(np.str)
            data = data.values
        else:
            samples = [str(i) for i in range(data.shape[0])]
        data_2 = data.copy()
        data_2 = data_2[:, index]
        max_v = np.max(data_2)
        min_v = np.min(data_2)
        data_2 = 255 - (data_2 - min_v) / (max_v - min_v) * 255
        image_data = np.empty((img_rows, img_columns, data_2.shape[0]))
        image_data.fill(np.nan)
        for i in range(data_2.shape[0]):
            data_i = np.empty((img_rows, img_columns))
            data_i.fill(np.nan)
            data_i[coord] = data_2[i, :]
            image_data[:, :, i] = data_i
            if image_folder is not None:
                if save_pngs:
                    fig = plt.figure()
                    plt.imshow(data_i, cmap="gray", vmin=0, vmax=255)
                    plt.axis("scaled")
                    plt.savefig(
                        fname=image_folder
                        + "/"
                        + file_name
                        + "_"
                        + samples[i]
                        + "_image.png",
                        bbox_inches="tight",
                        pad_inches=0,
                    )
                    plt.close(fig)
                pd.DataFrame(image_data[:, :, i], index=None, columns=None).to_csv(
                    image_folder + "/" + file_name + "_" + samples[i] + "_data.txt",
                    header=None,
                    index=None,
                    sep=",",
                )
    print(f"RUNTIME {time.time() - t0}")
    return image_data, samples


def table_to_image(
    norm_d,
    scale,
    fea_dist_method,
    image_dist_method,
    save_image_size,
    max_step,
    val_step,
    normDir,
    error,
    switch_t=0,
    min_gain=1e-05,
    save_mode="bulk",
    save_pngs=False,
    exclude_cols=[],
):
    """table_to_image

    Args:
    norm_d : type
        Description
    scale : type
        Description
    fea_dist_method : type
        Description
    image_dist_method : type
        Description
    save_image_size : type
        Description
    max_step : type
        Description
    val_step : type
        Description
    normDir : type
        Description
    error : type
        Description
    switch_t : type
        Description
    min_gain : type
        Description
    save_mode : type
        Description
    save_pngs : type
        Description
    exclude_cols : type
        Description

    Returns:
        type: Description
    """
    feature_cols = list(set.difference(set(norm_d.columns), set(exclude_cols)))
    norm_d = norm_d[exclude_cols + feature_cols]
    if os.path.exists(normDir):
        shutil.rmtree(normDir)
    os.mkdir(normDir)
    ranking_feature, corr = generate_feature_distance_ranking(
        data=norm_d[feature_cols], method=fea_dist_method
    )
    fig = plt.figure(figsize=(save_image_size, save_image_size))
    plt.imshow(
        np.max(ranking_feature) - ranking_feature, cmap="gray", interpolation="nearest"
    )
    plt.savefig(
        fname=normDir + "/original_feature_ranking.png",
        bbox_inches="tight",
        pad_inches=0,
    )
    plt.close(fig)
    coordinate, ranking_image = generate_matrix_distance_ranking(
        num_r=scale[0], num_c=scale[1], method=image_dist_method
    )
    fig = plt.figure(figsize=(save_image_size, save_image_size))
    plt.imshow(
        np.max(ranking_image) - ranking_image, cmap="gray", interpolation="nearest"
    )
    plt.savefig(fname=normDir + "/image_ranking.png", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    index, err, time = IGTD(
        source=ranking_feature,
        target=ranking_image,
        err_measure=error,
        max_step=max_step,
        switch_t=switch_t,
        val_step=val_step,
        min_gain=min_gain,
        random_state=1,
        save_folder=normDir + "/" + error,
        file_name="",
    )
    fig = plt.figure()
    plt.plot(time, err)
    plt.savefig(
        fname=normDir + "/error_and_runtime.png", bbox_inches="tight", pad_inches=0
    )
    plt.close(fig)
    fig = plt.figure()
    plt.plot(range(len(err)), err)
    plt.savefig(
        fname=normDir + "/error_and_iteration.png", bbox_inches="tight", pad_inches=0
    )
    plt.close(fig)
    min_id = np.argmin(err)
    ranking_feature_random = ranking_feature[index[min_id, :], :]
    ranking_feature_random = ranking_feature_random[:, index[min_id, :]]
    fig = plt.figure(figsize=(save_image_size, save_image_size))
    plt.imshow(
        np.max(ranking_feature_random) - ranking_feature_random,
        cmap="gray",
        interpolation="nearest",
    )
    plt.savefig(
        fname=normDir + "/optimized_feature_ranking.png",
        bbox_inches="tight",
        pad_inches=0,
    )
    plt.close(fig)
    data, samples = generate_image_data(
        data=norm_d,
        index=index[min_id, :],
        exclude_cols=exclude_cols,
        img_rows=scale[0],
        img_columns=scale[1],
        coord=coordinate,
        image_folder=normDir + "/data",
        file_name="",
        save_mode=save_mode,
        save_pngs=save_pngs,
    )
    if save_mode == "bulk":
        print(
            "Skipping single image txt and png generation, returning dataframe sorted by IGTD..."
        )
    else:
        output = open(normDir + "/Results.pkl", "wb")
        cp.dump(norm_d, output)
        cp.dump(data, output)
        cp.dump(samples, output)
        output.close()
        output = open(normDir + "/Results_Auxiliary.pkl", "wb")
        cp.dump(ranking_feature, output)
        cp.dump(ranking_image, output)
        cp.dump(coordinate, output)
        cp.dump(err, output)
        cp.dump(time, output)
        output.close()
        print(
            "IGTD Algorithm Finished run, transforming and generating png and txt images"
        )
