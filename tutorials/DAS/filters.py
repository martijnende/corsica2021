import numpy as np
from scipy.signal import tukey, butter, filtfilt, sosfiltfilt
from scipy.ndimage import gaussian_filter
from skimage.draw import polygon
import scipy.fft  # Scipy's FFT package is faster than Numpy's


""" Bandpass filtering """


def _butter_bandpass(lowcut, highcut, fs, order, mode):
    """Compute the numerator and denominator of a butterworth filter
    
    Parameters
    ----------
    
    lowcut : float
        Lower passband frequency (sames units as `fs`). Set to `-1` for a lowpass filter
    highcut : float
        Upper passband frequency (same units as `fs`). Set to `-1` for a highpass filter
    fs : float
        Sampling frequency
    order : int
        Filter order. Note that this order is doubled due to the forward and backward pass
    mode : str, default "ba"
        Type of filter design. Using b/a coefficients ("ba") is faster, but less stable than second-order sections ("sos") for higher orders
        
    Returns
    -------
    
    if mode is "ba":
    
    b : `numpy.array`
        Array of denominator coefficients
    a : `numpy.array`
        Array of numerator coefficients    
        
    if mode is "sos":
    
    sos : `numpy.array`
        Second-order sections representation
    
    """
    
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    
    if low < 0:
        Wn = high
        btype = "lowpass"
    elif high < 0:
        Wn = low
        btype = "highpass"
    else:
        Wn = [low, high]
        btype = "bandpass"
        
    if mode == "ba":
        b, a = butter(order, Wn, btype=btype, output="ba")
        return b, a
    
    if mode == "sos":
        sos = butter(order, Wn, btype=btype, output="sos")
        return sos
    
    return False


def taper_filter(arr, fmin, fmax, samp_DAS, order=2, mode="ba"):
    """Apply a taper and a butterworth filter to the input data
    
    Filter the data in a given frequency band using a 4th-order butterworth filter. The filter is applied forward and backward in time to prevent phase shifts.
    
    To avoid boundary effects, a Tukey taper is first applied to the data.
    
    Parameters
    ----------
    
    arr : `numpy.array`
        Input data. The filter will be applied to the last axis
    fmin : float
        Lower passband frequency (sames units as `samp_DAS`). Set to `-1` for a lowpass filter
    fmax : float
        Upper passband frequency (same units as `samp_DAS`). Set to `-1` for a highpass filter
    samp_DAS : float
        Sampling frequency
    order : int, default 2
        Filter order. Note that this order is doubled due to the forward and backward pass
    mode : str, default "ba"
        Type of filter design. Using b/a coefficients ("ba") is faster, but less stable than second-order sections ("sos") for higher orders
    
    Returns
    -------
    
    arr_filt : `numpy.array`
        Filtered data
    
    """
    
    if mode not in ("ba", "sos"):
        print(f"Filter type {mode} not recognised")
        print("Valid options: {'ba', 'sos'}")
        return False
    
    window_time = tukey(arr.shape[-1], 0.1)
    arr_wind = arr * window_time
    
    if mode == "ba":
        b, a = _butter_bandpass(fmin, fmax, samp_DAS, order, mode)
        arr_filt = filtfilt(b, a, arr_wind, axis=-1)
    
    if mode == "sos":
        sos = _butter_bandpass(fmin, fmax, samp_DAS, order, mode)
        arr_filt = sosfiltfilt(sos, arr_wind, axis=-1)
    
    return arr_filt


""" fk-filtering """


def make_fk(arr, gauge, samp, merge=False, return_PSD=True):
    """Construct an `fk`-diagram
    
    Parameters
    ----------
    
    arr : `numpy.array`
        Input data of shape (channels, time)
    gauge : float
        Gauge length (in metres)
    samp : float
        Time sampling frequency (in Hertz)
    merge : bool, default False
        Whether or not to merge the positive sampling frequencies with the negative ones
    return_PSD : bool, default True
        Whether or not to return the PSD. Otherwise the `fk`-transformed data is returned
        
    Returns
    -------
    
    f : `numpy.array`
        Vector of time frequencies
    k : `numpy.array`
        Vector of wavenumbers
    PSD : `numpy.array`
        Power spectral density if `return_PSD` is `True`, `fk`-transformed data otherwise
    
    """
    
    # Data dimensions
    Nch, Nt = arr.shape
    
    # Sampling wavenumber and time frequency
    Fk = 1./gauge
    Fs = samp
    
    # Wavenumbers [1/m] and time frequencies [1/s] 
    k = np.linspace(-Fk / 2.0, Fk / 2.0, Nch)
    f = np.linspace(-Fs / 2.0, Fs / 2.0, Nt)
    
    # Y will have shape (Nch, Nt)
    Y = scipy.fft.fft2(arr)
    Y = scipy.fft.fftshift(Y)
    
    if merge:
        Y = Y + np.flip(Y, axis=(0,1))
        Y = Y[:, Nt//2:]
        f = f[Nt//2:]
        
    if return_PSD:
        # Compute the PSD and transpose
        PSD = 2 * np.log10(np.abs(Y)).T
    else:
        PSD = Y
    
    return f, k, PSD


def cut_fk(PSD, f, f_range, k, k_range):
    """Cut the `fk`-diagram within a given range of wavenumbers and time frequencies
    
    Parameters
    ----------
    
    PSD : `numpy.array`
        PSD of the `fk`-diagram
    f : `numpy.array`
        Vector of time frequencies
    f_range : tuple
        Lower and upper bounds of the time frequencies at which the `fk`-diagram is cut (same units as `f`)
    k : `numpy.array`
        Vector of wavenumbers
    k_range : tuple
        Lower and upper bounds of the wavenumbers at which the `fk`-diagram is cut (same units as `k`)
        
    Returns
    -------
    
    f_cut : `numpy.array`
        Vector of clipped time frequencies
    k_cut : `numpy.array`
        Vector of clipped wavenumbers
    PSD_cut : `numpy.array`
        Cut `fk`-diagram
    
    """
    
    fmin, fmax = f_range
    kmin, kmax = k_range
    f_inds = (f >= fmin) & (f <= fmax)
    k_inds = (k >= kmin) & (k <= kmax)
    
    f_cut = f[f_inds]
    k_cut = k[k_inds]
    
    PSD_cut = PSD[f_inds][:, k_inds]
    
    return f_cut, k_cut, PSD_cut


def filter_fk(data, gauge, samp, vmin, vmax):
    """Applies an `fk`-filter using two velocities
    
    Assuming a constant velocity, the phase/group velocity of a signal can be written as `v = f/k`. For a given range of `k`, velocity-based filtering is applied by setting `f > vmax * k` and `f < vmin * k` to zero.
    
    The sign of `vmin` and `vmax` indicates the direction of travel (positive towards the interrogator, negative away from the interrogator). To filter only a specific direction, set `vmin = 0` and `vmax = np.inf` (or `vmax = -np.inf` for the opposite direction).
    
    Parameters
    ----------
    
    data : `numpy.array`
        Data to be filtered
    gauge : float
        Gauge length (in m)
    samp : float
        Time sampling frequency (in Hz)
    vmin : float
        Lower velocity bound (in m/s)
    vmax : float
        Upper velocity bound (in m/s)
        
    Returns
    -------
    
    data_fk : `numpy.array`
        `fk`-filtered data
    win : `numpy.array`
        Mask applied to the `fk`-transformed data
    
    """
    
    f, k, Y = make_fk(data, gauge, samp, merge=False, return_PSD=False)
    
    win = _draw_polygon_win(f, k, vmin, vmax)

    Y_win = scipy.fft.ifftshift(win.T * Y)
    data_fk = np.real(scipy.fft.ifft2(Y_win))
    
    return data_fk, win


def _draw_polygon_win(f, k, vmin, vmax):
    """Creates a polygonal fk-mask
    
    The polygon is drawn within fk-space based on a lower and upper velocity (positive for signals travelling towards the interrogator, negative for the opposite direction).
    A Gaussian smoothing is applied to let the mask go to zero smoothly.
    
    Parameters
    ----------
    
    f : `numpy.array`
        Vector of time frequencies (in Hz)
    k : `numpy.array`
        Vector of wavenumbers (in 1/m)
    vmin : float
        Lower velocity bound
    vmax : float
        Upper velocity bound
    
    Returns
    -------
    
    win : `numpy.array`
        Smoothed fk-mask to be applied to fk-transformed data
        
    """
    
    Nk, Nf = len(k), len(f)
    
    # In the trivial case that vmin = -inf and vmax = inf, all the coefficients are kept
    if (vmax == np.inf) and (vmin == -np.inf):
        return np.ones((Nf, Nk))
    
    k_range = np.linspace(k.min(), k.max(), 1001)
    f_range_min = vmin * k_range
    f_range_max = vmax * k_range
    
    dir_vmin = np.sign(vmin)
    dir_vmax = np.sign(vmax)

    win = np.zeros((Nf, Nk))
    
    k_centre = np.argmin(np.abs(k))
    left = 0
    right = Nk - 1
    down = 0
    up = Nf - 1

    f_centre = np.argmin(np.abs(f))
    
    """ Centre point """
    r = [f_centre]
    c = [k_centre]
    
    """ Upper bound """
    
    # If vmax is zero
    if vmax == 0:
        r.append(f_centre)
        c.append(right)
    # If vmax is infinite
    elif not np.isfinite(vmax):
        r.append(up)
        r.append(up)
        c.append(k_centre)
        c.append(right)
    # If vmax is very large but finite
    elif f_range_max.max() > f.max():
        if dir_vmax > 0:
            r.append(up)
            r.append(up)
        else:
            r.append(down)
            r.append(down)
        c.append(np.argmin(np.abs(k - dir_vmax * f.max() / vmax)))
        c.append(right)
    else:
        r.append(np.argmin(np.abs(dir_vmax * f - f_range_max.max())))
        c.append(right)
        
    """ Lower bound """
    
    # If min is zero
    if vmin == 0:
        r.append(f_centre)
        c.append(right)
    # If vmin is (negative) infinite
    elif not np.isfinite(vmin):
        r.append(down)
        r.append(down)
        c.append(right)
        c.append(k_centre)
    # If vmin is very small but finite
    elif f_range_min.max() > f.max():
        if dir_vmin > 0:
            r.append(up)
            r.append(up)
        else:
            r.append(down)
            r.append(down)
        c.append(right)
        c.append(np.argmin(np.abs(k - dir_vmin * f.max() / vmin)))
    else:
        r.append(np.argmin(np.abs(dir_vmin * f - f_range_min.max())))
        c.append(right)
    
    """ Centre point """
    r.append(f_centre)
    c.append(k_centre)

    rr, cc = polygon(r, c)
    win[rr, cc] = 1
    win = win + np.flip(win, axis=(0, 1))

    win = gaussian_filter(win, sigma=1.0, mode="constant")
    
    return win
