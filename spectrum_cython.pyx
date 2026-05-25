# cython: boundscheck=False
import numpy as np
cimport numpy as np
from libc.math cimport log10, sqrt, fabs
from pyfftw.interfaces.numpy_fft import rfft


cpdef np.ndarray log_band_volumes(
    np.ndarray data,
    np.ndarray freqs,
    int num_bands,
    np.ndarray band_edges,
    float max_ref,
    int window,
    bint discrete,
):
    """Get logarythmic volume in dB for specified number of bands, from sound sample, with interpolation between bands"""
    cdef int i, b, left_bin, right_bin
    cdef float center, band_center, d, wsum
    cdef np.ndarray raw, raw_magnitude, magnitude, db, idx, bins, weights
    cdef list py_bins
    cdef np.float32_t left, right

    if window == 0:
        pass
    elif window == 1:
        np.multiply(data, np.hamming(len(data)), out=data)
    elif window == 2:
        np.multiply(data, np.hanning(len(data)), out=data)
    elif window == 3:
        np.multiply(data, np.blackman(len(data)), out=data)
    raw = rfft(data, threads=1)
    raw_magnitude = np.abs(raw)
    magnitude = np.zeros(num_bands)

    for i in range(num_bands):
        left = <np.float32_t>band_edges[i]
        right = <np.float32_t>band_edges[i + 1]
        idx = (np.where((freqs >= left) & (freqs < right))[0])
        if idx.shape[0] == 0:
            left_bin = np.searchsorted(freqs, left)
            right_bin = np.searchsorted(freqs, right)
            py_bins = []
            if left_bin > 0:
                py_bins.append(left_bin - 1)
            if left_bin < freqs.shape[0]:
                py_bins.append(left_bin)
            if right_bin > 0 and right_bin != left_bin:
                py_bins.append(right_bin - 1)
            if right_bin < freqs.shape[0] and right_bin != left_bin:
                py_bins.append(right_bin)
            bins = np.array(list(set(py_bins)), dtype=np.int32)
            band_center = (left + right) / 2
            weights = np.zeros(bins.shape[0], dtype=np.float32)
            wsum = 0.0
            for b in range(bins.shape[0]):
                center = freqs[bins[b]]
                d = fabs(center - band_center) + 1e-6
                weights[b] = 1.0 / d
                wsum += weights[b]
            weights /= wsum
            magnitude[i] = sqrt(np.sum((raw_magnitude[bins] ** 2) * weights))
        elif discrete:
            band_center = (left + right) / 2
            magnitude[i] = raw_magnitude[np.argmin(np.abs(freqs - band_center))]
        else:
            magnitude[i] = sqrt(np.mean(raw_magnitude[idx] ** 2))

    db = 20.0 * np.log10(magnitude / max_ref + 1e-12)
    return np.maximum(db, -90.0)


def generate_spectrum(
    list left_lines,
    list right_lines,
    np.ndarray[np.int32_t, ndim=1] bar_heights,
    np.ndarray[np.int32_t, ndim=1] peak_heights,
    int bar_height,
    str bar_char,
    str half_bar_char,
    str peak_char,
    bint peaks,
    bint box,
    bint axes,
    colors
):
    """Draw spectrum bars with peaks"""
    cdef list lines = []
    cdef int width = bar_heights.shape[0]
    cdef int y_raw, y, i
    cdef double relative
    cdef int color
    cdef list line

    if box:
        lines.append(left_lines[0])
    for y_raw in range(bar_height - box):
        y = y_raw + box
        line = [" "] * width

        for i in range(width):
            if y_raw >= bar_height - bar_heights[i]:
                line[i] = bar_char
            if peaks and y_raw == bar_height - peak_heights[i]:
                line[i] = peak_char

        if colors is not None:
            relative = (bar_height - y_raw) / bar_height
            if relative < 0.5:
                color = colors[0]
            elif relative < 0.8:
                color = colors[1]
            else:
                color = colors[2]
            lines.append(left_lines[y] + f"\x1b[38;5;{color}m" + "".join(line) + "\x1b[0m" + right_lines[y_raw])
        else:
            lines.append(left_lines[y] + "".join(line) + right_lines[y_raw])

    if axes:
        lines.append(left_lines[-2] + right_lines[-1])
    if box:
        lines.append(left_lines[-1])

    return lines


def generate_spectrum_half(
    list left_lines,
    list right_lines,
    np.ndarray[np.int32_t, ndim=1] bar_heights,
    np.ndarray[np.int32_t, ndim=1] peak_heights,
    int bar_height,
    str bar_char,
    str half_bar_char,
    str peak_char,
    bint peaks,
    bint box,
    bint axes,
    colors
):
    """Draw spectrum bars with peaks"""
    cdef list lines = []
    cdef int width = bar_heights.shape[0]
    cdef int y_raw, y, i, height, top_full_start
    cdef double relative
    cdef int color
    cdef list line

    bar_height = int(bar_height/2)

    if box:
        lines.append(left_lines[0])
    for y_raw in range(bar_height - box):
        y = y_raw + box
        line = [" "] * width

        for i in range(width):
            height = bar_heights[i]
            top_full_start = bar_height - height // 2
            if y_raw >= top_full_start:
                line[i] = bar_char
            elif height & 1 and y_raw == top_full_start - 1:
                line[i] = half_bar_char
            if peaks and y_raw == bar_height - (peak_heights[i] // 2) - 1:
                line[i] = peak_char

        if colors is not None:
            relative = (bar_height - y_raw) / bar_height
            if relative < 0.5:
                color = colors[0]
            elif relative < 0.8:
                color = colors[1]
            else:
                color = colors[2]
            lines.append(left_lines[y] + f"\x1b[38;5;{color}m" + "".join(line) + "\x1b[0m" + right_lines[y_raw])
        else:
            lines.append(left_lines[y] + "".join(line) + right_lines[y_raw])

    if axes:
        lines.append(left_lines[-2] + right_lines[-1])
    if box:
        lines.append(left_lines[-1])

    return lines
