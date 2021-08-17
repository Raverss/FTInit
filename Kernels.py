from __future__ import division

import itertools
from typing import List, Tuple

import cv2
import numpy as np
from PIL import Image
from scipy import signal


def get_gauss(size, angle=0) -> np.array:
    gauss = signal.gaussian(size, size // 3).reshape(1, size)
    gauss = gauss * gauss.transpose()
    M = cv2.getRotationMatrix2D((gauss.shape[0] // 2, gauss.shape[1] // 2), angle, 1)
    return cv2.warpAffine(gauss, M, (gauss.shape[0], gauss.shape[1]))


def get_log(size, angle=0):
    if size == 3:
        log = np.array(
            [
                [0.0, 1.0, 0.0],
                [1.0, -4.0, 1.0],
                [0.0, 1.0, 0.0]
            ], dtype=np.float32)
    elif size == 5:
        log = np.array(
            [
                [0.0, 0.0, -1.0, 0.0, 0.0],
                [0.0, -1.0, -2.0, -1.0, 0.0],
                [-1.0, -2.0, 16.0, -2.0, -1.0],
                [0.0, -1.0, -2.0, -1.0, 0.0],
                [0.0, 0.0, -1.0, 0.0, 0.0]
            ], dtype=np.float32)

    elif size == 7:
        # https://hcimage.com/help/Content/Quantitation/Measurements/Processing%20and%20Analysis/Enhance/Enhance%20Operations.htm
        log = np.array(
            [
                [0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0],
                [0.0, 1.0, 3.0, 3.0, 3.0, 1.0, 0.0],
                [1.0, 3.0, 0.0, -7.0, 0.0, 3.0, 1.0],
                [1.0, 3.0, -7.0, -24.0, -7.0, 3.0, 1.0],
                [1.0, 3.0, 0.0, -7.0, 0.0, 3.0, 1.0],
                [0.0, 1.0, 3.0, 3.0, 3.0, 1.0, 0.0],
                [0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0]
            ], dtype=np.float32)
    else:
        raise NotImplementedError("Only available sizes for Laplacian of Gaussian are 3, 5 and 7.")
    log /= np.max(np.abs(log))
    M = cv2.getRotationMatrix2D((log.shape[0] // 2, log.shape[1] // 2), angle, 1)
    return cv2.warpAffine(log, M, (log.shape[0], log.shape[1]))


def get_sobel(size, angle=0):
    if size == 3:
        sobel = np.array(
            [
                [-1.0, 0.0, 1.0],
                [-2.0, 0.0, 2.0],
                [-1.0, 0.0, 1.0]
            ], dtype=np.float32)
    elif size == 5:
        # https://www.xilinx.com/html_docs/xilinx2018_2/sdsoc_doc/sobel-filter-dyd1504034280488.html
        sobel = np.array(
            [
                [-1.0, -2.0, 0.0, 2.0, 1.0],
                [-4.0, -8.0, 0.0, 8.0, 4.0],
                [-6.0, -12.0, 0.0, 12.0, 6.0],
                [-4.0, -8.0, 0.0, 8.0, 4.0],
                [-1.0, -2.0, 0.0, 2.0, 1.0]
            ], dtype=np.float32)

    elif size == 7:
        # https://www.xilinx.com/html_docs/xilinx2018_2/sdsoc_doc/sobel-filter-dyd1504034280488.html
        sobel = np.array(
            [
                [-1.0, -4.0, -5.0, 0.0, 5.0, 4.0, 1.0],
                [-6.0, -24.0, -30.0, 0.0, 30.0, 24.0, 6.0],
                [-15.0, -60.0, -75.0, 0.0, 75.0, 60.0, 15.0],
                [-20.0, -80.0, -100.0, 0.0, 100.0, 80.0, 20.0],
                [-15.0, -60.0, -75.0, 0.0, 75.0, 60.0, 15.0],
                [-6.0, -24.0, -30.0, 0.0, 30.0, 24.0, 6.0],
                [-1.0, -4.0, -5.0, 0.0, 5.0, 4.0, 1.0]
            ], dtype=np.float32)
    else:
        raise NotImplementedError("Only available sizes for Sobel are 3, 5 and 7.")
    sobel /= np.max(np.abs(sobel))
    M = cv2.getRotationMatrix2D((sobel.shape[0] // 2, sobel.shape[1] // 2), angle, 1)
    return cv2.warpAffine(sobel, M, (sobel.shape[0], sobel.shape[1]))


def gabor_fn(sigma, theta, Lambda, psi, gamma):
    sigma_x = sigma
    sigma_y = float(sigma) / gamma

    # Bounding box
    nstds = 2
    xmax = max(abs(nstds * sigma_x * np.cos(theta)), abs(nstds * sigma_y * np.sin(theta)))
    xmax = np.ceil(max(1, xmax))
    ymax = max(abs(nstds * sigma_x * np.sin(theta)), abs(nstds * sigma_y * np.cos(theta)))
    ymax = np.ceil(max(1, ymax))
    xmin = -xmax
    ymin = -ymax
    (y, x) = np.meshgrid(np.arange(ymin, ymax + 1), np.arange(xmin, xmax + 1))

    # Rotation
    x_theta = x * np.cos(theta) + y * np.sin(theta)
    y_theta = -x * np.sin(theta) + y * np.cos(theta)

    gb = np.exp(-.5 * (x_theta ** 2 / sigma_x ** 2 + y_theta ** 2 / sigma_y ** 2)) * np.cos(
        2 * np.pi / Lambda * x_theta + psi)
    return gb


def get_ft0(size, angle=0):
    ft0 = signal.triang(size).reshape(1, size)
    ft0 = ft0 * ft0.transpose()
    M = cv2.getRotationMatrix2D((ft0.shape[0] // 2, ft0.shape[1] // 2), angle, 1)
    return cv2.warpAffine(ft0, M, (ft0.shape[0], ft0.shape[1]))


def get_ft1(size, angle=0):
    a = -1
    b = +1
    x = np.linspace(a, b, size + 2).reshape([1, size + 2])
    vec = 1 - np.abs(x)
    ft1 = np.dot(np.transpose(vec), x * vec)[1: size + 2 - 1, 1: size + 2 - 1]
    M = cv2.getRotationMatrix2D((ft1.shape[0] // 2, ft1.shape[1] // 2), angle, 1)
    return cv2.warpAffine(ft1, M, (ft1.shape[0], ft1.shape[1]))


def get_ft2_0(size):
    a = -1
    b = +1
    I2 = 1 / 6
    x = np.linspace(a, b, size + 2).reshape([1, size + 2])
    A = (1 - np.abs(x)).reshape([1, size + 2])
    P2 = np.square(x) - I2
    P2A = P2 * A
    return np.dot(np.transpose(A), P2A)[1: size + 2 - 1, 1: size + 2 - 1]


def get_ft2_90(size):
    a = -1
    b = +1
    I2 = 1 / 6
    x = np.linspace(a, b, size + 2).reshape([1, size + 2])
    A = (1 - np.abs(x)).reshape([1, size + 2])
    P2 = np.square(x) - I2
    P2A = P2 * A
    return np.dot(np.transpose(P2A), A)[1: size + 2 - 1, 1: size + 2 - 1]


def get_ft2c(size, angle=0):
    ft2c = get_ft2_0(size) + get_ft2_90(size)
    M = cv2.getRotationMatrix2D((ft2c.shape[0] // 2, ft2c.shape[1] // 2), angle, 1)
    return cv2.warpAffine(ft2c, M, (ft2c.shape[0], ft2c.shape[1]))


def rotate_matrix(m: np.ndarray, a: int):
    if m.ndim > 3:
        print('matrix has to be 2D, received array with dimensions equals to', m.ndim)
        return m
    if m.shape[0] != m.shape[1]:
        print('matrix has to have square size, received array with shape', m.shape)
        return m
    if m.shape[0] % 2 != 1:
        print('matrix has to have odd number of rows and columns')
        return m

    s = (m.shape[0] - 1) // 2
    o = np.zeros(m.shape)
    for c in range(1, s + 1):
        indicies = []
        for i in range(0, 2 * c):
            indicies.append((s + c, s + c - i))
        for i in range(0, 2 * c):
            indicies.append((s + c - i, s - c))
        for i in range(0, 2 * c):
            indicies.append((s - c, s - c + i))
        for i in range(0, 2 * c):
            indicies.append((s - c + i, s + c))

        skip = int(a // (360 / len(indicies)))
        for i in range(0, len(indicies)):
            o[indicies[(i + skip) % len(indicies)][0], indicies[(i + skip) % len(indicies)][1]] = m[
                indicies[i][0], indicies[i][1]]
    # center of the kernel is not part of the rotation and need to be copied in this extra step
    o[o.shape[0] // 2, o.shape[1] // 2] = m[m.shape[0] // 2, m.shape[1] // 2]
    return o


def get_kernels(params: List[Tuple[str, int, int, int, int]]) -> List[np.ndarray]:
    """
    Create list of kernels
    :param params: list of tuples with following format ("kernel name", angle, multiplier, rotation angle)
    :return: list of kernels
    """
    kernels = []  # type: List[np.ndarray]
    for param in params:
        if len(param) < 5:
            print('Number of parameters given must be 4, got', param, 'len(', len(param), ') instead')
            return None
        if param[0] == 'gauss':
            kernels.append(rotate_matrix(get_gauss(param[1], param[2]) * param[3], param[4]))
        elif param[0] == 'log':
            kernels.append(rotate_matrix(get_log(param[1], param[2]) * param[3], param[4]))
        elif param[0] == 'sobel':
            kernels.append(rotate_matrix(get_sobel(param[1], param[2]) * param[3], param[4]))
        elif param[0] == 'ft0':
            kernels.append(rotate_matrix(get_ft0(param[1], param[2]) * param[3], param[4]))
        elif param[0] == 'ft1':
            kernels.append(rotate_matrix(get_ft1(param[1], param[2]) * param[3], param[4]))
        elif param[0] == 'ft2c':
            kernels.append(rotate_matrix(get_ft2c(param[1], param[2]) * param[3], param[4]))
    if len(kernels) == 1:
        return kernels[0]
    else:
        return kernels


if __name__ == '__main__':
    print('Kernels.py main is not meant to be called')
