# cython: boundscheck=False, wraparound=False
import curses
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
):
    """Get logarythmic volume in dB for specified number of bands, from sound sample, with interpolation between bands"""
    cdef int i, b, left_bin, right_bin
    cdef float center, band_center, d, wsum
    cdef np.ndarray raw, raw_magnitude, magnitude, db, idx, bins, weights
    cdef list py_bins
    cdef np.float32_t left, right

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
        else:
            magnitude[i] = sqrt(np.mean(raw_magnitude[idx] ** 2))

    db = 20.0 * np.log10(magnitude / max_ref + 1e-12)
    return np.maximum(db, -90.0)


cpdef int get_color(int y, int bar_height, bint use_color):
    """Get color id by bar height"""
    if not use_color:
        return curses.color_pair(0)
    cdef float relative = (bar_height - y) / bar_height
    if relative < 0.5:
        return curses.color_pair(1)
    if relative < 0.8:
        return curses.color_pair(2)
    return curses.color_pair(3)


cpdef void draw_spectrum(
    object spectrum_win,
    np.ndarray[np.int32_t, ndim=1] bar_heights,
    np.ndarray[np.int32_t, ndim=1] peak_heights,
    int bar_height,
    str bar_character,
    str peak_character,
    bint peaks,
    bint color,
    bint box
):
    """Draw spectrum bars with peaks"""
    cdef int y, x, bar, peak, i
    cdef int width = bar_heights.shape[0]
    cdef list line

    for y in range(bar_height - box):
        line = [" "] * width
        for i in range(width):
            bar = bar_heights[i]
            if y >= bar_height - bar:
                line[i] = bar_character
        if peaks:
            for x in range(width):
                peak = peak_heights[x]
                if y == bar_height - peak:
                    line[x] = peak_character
        spectrum_win.insstr(y, 0, "".join(line), get_color(y, bar_height, color))
        spectrum_win.refresh()
