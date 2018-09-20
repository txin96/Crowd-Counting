import numpy as np
import math


def matlab_style_gauss2D(shape=(3, 3), sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2. * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h


def get_density_map_gaussian(im, points):
    im_density = np.zeros(im.shape)
    [h, w] = im_density.shape

    for j in range(0, len(points)):
        f_sz = 15
        sigma = 4.0
        # H = matlab.fspecial('Gaussian', [f_sz, f_sz], sigma)
        H = matlab_style_gauss2D([f_sz, f_sz], sigma)
        x = min(w, max(1, abs(int(math.floor(points[j, 0])))))
        y = min(h, max(1, abs(int(math.floor(points[j, 1])))))

        if x > w or y > h:
            continue
        x1 = x - int(np.floor(f_sz / 2))
        y1 = y - int(np.floor(f_sz / 2))
        x2 = x + int(np.floor(f_sz / 2))
        y2 = y + int(np.floor(f_sz / 2))
        dfx1 = 0
        dfy1 = 0
        dfx2 = 0
        dfy2 = 0
        change_H = False
        if x1 < 1:
            dfx1 = abs(x1) + 1
            x1 = 1
            change_H = True
        if y1 < 1:
            dfy1 = abs(y1) + 1
            y1 = 1
            change_H = True
        if x2 > w:
            dfx2 = x2 - w
            x2 = w
            change_H = True
        if y2 > h:
            dfy2 = y2 - h
            y2 = h
            change_H = True
        x1h = 1 + dfx1
        y1h = 1 + dfy1
        x2h = f_sz - dfx2
        y2h = f_sz - dfy2
        if change_H:
            # H = matlab.fspecial('Gaussian', [double(y2h - y1h + 1), double(x2h - x1h + 1)], sigma)
            H = matlab_style_gauss2D([float(y2h - y1h + 1), float(x2h - x1h + 1)], sigma)
        im_density[y1-1: y2, x1-1: x2] = im_density[y1-1: y2, x1-1: x2] + H

    return im_density

