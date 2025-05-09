import sys
import os
import numpy as np
from astropy.io import fits

from scipy import ndimage, signal
from scipy.interpolate import interp1d

from itertools import chain

import matplotlib.pyplot as plt
import matplotlib.collections as mcoll

from multiprocessing import Pool
import pickle as pkl

from collections import deque



# =======================================================================================
# ================================= Ellipse / Fitting ===================================
# =======================================================================================

def fit_ellipse(x, y):
    """
    Fit the coefficients a,b,c,d,e,f, representing an ellipse described by
    the formula F(x,y) = ax^2 + bxy + cy^2 + dx + ey + f = 0 to the provided
    arrays of data points x=[x1, x2, ..., xn] and y=[y1, y2, ..., yn].

    Based on the algorithm of Halir and Flusser, "Numerically stable direct
    least squares fitting of ellipses'.

    Parameters
    ----------

    x, y    :   numpy.ndarray
                ellipse points
    
    Returns
    -------

    list
                List of coefficients (a, b, c, d, e, f)
    """
    D1 = np.vstack([x**2, x*y, y**2]).T
    D2 = np.vstack([x, y, np.ones(len(x))]).T
    S1 = D1.T @ D1
    S2 = D1.T @ D2
    S3 = D2.T @ D2
    T = -np.linalg.inv(S3) @ S2.T
    M = S1 + S2 @ T
    C = np.array(((0, 0, 2), (0, -1, 0), (2, 0, 0)), dtype=float)
    M = np.linalg.inv(C) @ M
    eigval, eigvec = np.linalg.eig(M)
    con = 4 * eigvec[0]* eigvec[2] - eigvec[1]**2
    ak = eigvec[:, np.nonzero(con > 0)[0]]
    return np.concatenate((ak, T @ ak)).ravel()

def cart_to_pol(coeffs):
    """
    Convert the cartesian conic coefficients, (a, b, c, d, e, f), to the
    ellipse parameters, where F(x, y) = ax^2 + bxy + cy^2 + dx + ey + f = 0.
    The returned parameters are x0, y0, ap, bp, e, phi, where (x0, y0) is the
    ellipse centre; (ap, bp) are the semi-major and semi-minor axes,
    respectively; e is the eccentricity; and phi is the rotation of the semi-
    major axis from the x-axis.

    Parameters
    ----------

    coeffs      :   list
                    List of coefficients (a, b, c, d, e, f)

    Returns
    -------

    x0, y0      :   float
                    ellipse center
    ap, bp      :   float
                    semi-major and semi-minor axes
    e           :   float
                    eccentricity
    phi         :   rotation of the semi-major axis from the x-axis
    """

    # We use the formulas from https://mathworld.wolfram.com/Ellipse.html
    # which assumes a cartesian form ax^2 + 2bxy + cy^2 + 2dx + 2fy + g = 0.
    # Therefore, rename and scale b, d and f appropriately.
    a = coeffs[0]
    b = coeffs[1] / 2
    c = coeffs[2]
    d = coeffs[3] / 2
    f = coeffs[4] / 2
    g = coeffs[5]

    den = b**2 - a*c
    if den > 0:
        raise ValueError('coeffs do not represent an ellipse: b^2 - 4ac must'
                         ' be negative!')

    # The location of the ellipse centre.
    x0, y0 = (c*d - b*f) / den, (a*f - b*d) / den

    num = 2 * (a*f**2 + c*d**2 + g*b**2 - 2*b*d*f - a*c*g)
    fac = np.sqrt((a - c)**2 + 4*b**2)
    # The semi-major and semi-minor axis lengths (these are not sorted).
    ap = np.sqrt(num / den / (fac - a - c))
    bp = np.sqrt(num / den / (-fac - a - c))

    # Sort the semi-major and semi-minor axis lengths but keep track of
    # the original relative magnitudes of width and height.
    width_gt_height = True
    if ap < bp:
        width_gt_height = False
        ap, bp = bp, ap

    # The eccentricity.
    r = (bp/ap)**2
    if r > 1:
        r = 1/r
    e = np.sqrt(1 - r)

    # The angle of anticlockwise rotation of the major-axis from x-axis.
    if b == 0:
        phi = 0 if a < c else np.pi/2
    else:
        phi = np.arctan((2.*b) / (a - c)) / 2
        if a > c:
            phi += np.pi/2
    if not width_gt_height:
        # Ensure that phi is the angle to rotate to the semi-major axis.
        phi += np.pi/2
    phi = phi % (2*np.pi)
    return x0, y0, ap, bp, e, phi

def get_ellipse_pts(params, npts=100, tmin=0, tmax=2*np.pi, liste=None):
    """
    Return npts points on the ellipse described by the params = x0, y0, ap,
    bp, e, phi for values of the parametric variable t between tmin and tmax.

    Parameters
    ----------

    params      :   tuple
                    (x0, y0, ap, bp, e, phi) respectively ellipse center position, semi-major, semi-minor axis, eccentricity, rotation of the semi-major axis from the x-axis
    npts        :   int (optional)
                    number of ellipse points
    tmin, tmax  :   float
                    limits of parametric variable (tipically 0, 2*pi)
    liste       :   list
                    if defined, the points of the ellipse will be defined by this list of parametric variables, and no longer as a function of tmin and tmax

    Returns
    -------

    numpy.ndarray
                    ellipse points (with ntps (or liste size) number of points)
    """
    x0, y0, ap, bp, e, phi = params
    # A grid of the parametric variable, t.
    if liste == None:
        t = np.linspace(tmin, tmax, npts)
    else :
        t = np.array(liste)
    x = x0 - ap * np.cos(t) * np.cos(phi) - bp * np.sin(t) * np.sin(phi)
    y = y0 + ap * np.cos(t) * np.sin(phi) - bp * np.sin(t) * np.cos(phi)


    return x, y

def Max_pixel(image, R_max=None, gaussian_filter=3, smooth_filter=10, prominence=0.1, distance=1, width=1, threshold=None, HighPass=1, Mincut_Radius=49, Maxcut_Radius=50, nb_phi=360, method='Azimuthal'):
    """
    Detection of all brightness maxima on fits image using differents methods, and for many differents filter values

    Parameters
    ----------

    image               :   ndarray
                            fits image

    R_max               :   float (optional)
                            set a maximum radius for maxima detection

    gaussian_filter     :   float (optional)
                            set sigma value for gaussian filter

    smooth_filter       :   float (optional)
                            set sigma value for smoothing profiles

    prominence          :   float or ndarray or sequence (optional)
                            Required prominence of peaks. Either a number, None, an array matching x or a 2-element sequence of the former. The first element is always interpreted as the minimal and the second, if supplied, as the maximal required prominence. (see scipy.signal.find_peaks documentation)

    distance            :   float (optional)
                            Required minimal horizontal distance (>= 1) in samples between neighbouring peaks. Smaller peaks are removed first until the condition is fulfilled for all remaining peaks. (see scipy.signal.find_peaks documentation)

    width               :   float or ndarray or sequence (optional)
                            Required width of peaks in samples. Either a number, None, an array matching x or a 2-element sequence of the former. The first element is always interpreted as the minimal and the second, if supplied, as the maximal required width. (see scipy.signal.find_peaks documentation)

    threshold           :   float or ndarray or sequence (optional)
                            Required threshold of peaks, the vertical distance to its neighboring samples. Either a number, None, an array matching x or a 2-element sequence of the former. The first element is always interpreted as the minimal and the second, if supplied, as the maximal required threshold. (see scipy.signal.find_peaks documentation)

    Mincut_Radius       :   float (optional)
                            set minimum radius for maxima detection
                            
    Maxcut_Radius       :   float (optional)
                            set maximum radius for maxima detection

    nb_phi              :   float (optional)
                            set number of azimuth angles used for radial profile extration and maxima detection

    method              :   str (optional)
                            chose maxima detection method, the main method is “Azimuthal” but others are possible

    Returns
    -------

    X, Y        :   list
                    position of brightness peak detected and filtered, usable for ellipse fitting
    """
    
    X, Y  = [], []
    image = ndimage.gaussian_filter(image , sigma = gaussian_filter)

    if method == 'Diagonal':
        w = 0
        i = np.arange(0, len(image))
        j = i[::-1]
        s = len(image)
        for dec in range(1, len(image)-2):
            i_up = i[:-dec]
            j_up = j[:-dec]

            i_down = i[:-dec]
            j_down = j[dec:]

            flux_up   = (image[i_up + dec, j_up])
            flux_down = (image[i_down, j_down])
            flux_smooth_up   = moving_average(np.array(flux_up), smooth_filter)
            flux_smooth_down = moving_average(np.array(flux_down), smooth_filter)

            peaks_find_up  , _ = signal.find_peaks(flux_smooth_up,   prominence=prominence, distance=distance, width=width, threshold=threshold) #, distance=d)
            peaks_find_down, _ = signal.find_peaks(flux_smooth_down, prominence=prominence, distance=distance, width=width, threshold=threshold) #, distance=d)
            
            for peak in (peaks_find_up):
                x = dec + peak + w 
                y = s   - peak + w 
                if Mincut_Radius <= np.sqrt((x-s/2)**2 + (y-s/2)**2) <= Maxcut_Radius:
                    X.append(x)
                    Y.append(y)

            for peak in (peaks_find_down):
                x = peak + w
                y = s - dec - peak - w
                if Mincut_Radius <= np.sqrt((x-s/2)**2 + (y-s/2)**2) <= Maxcut_Radius:
                    X.append(x)
                    Y.append(y)
    elif method == 'Antidiagonal':
        w = 0
        s = len(image)
        for dec in range(1, len(image)-2):
            i = np.arange(0, len(image) - dec)
            flux_up   = (image[i      , i + dec])
            flux_down = (image[i + dec, i])
            flux_smooth_up   = moving_average(np.array(flux_up), smooth_filter)
            flux_smooth_down = moving_average(np.array(flux_down), smooth_filter)

            peaks_find_up  , _ = signal.find_peaks(flux_smooth_up,   prominence=prominence, distance=distance, width=width, threshold=threshold) #, distance=d)
            peaks_find_down, _ = signal.find_peaks(flux_smooth_down, prominence=prominence, distance=distance, width=width, threshold=threshold) #, distance=d)
            
            for peak in (peaks_find_up):
                x = peak
                y = dec + peak
                if Mincut_Radius <= np.sqrt((x-s/2)**2 + (y-s/2)**2) <= Maxcut_Radius:
                    X.append(x)
                    Y.append(y)
            for peak in (peaks_find_down):
                x = dec + peak
                y = peak
                if Mincut_Radius <= np.sqrt((x-s/2)**2 + (y-s/2)**2) <= Maxcut_Radius:
                    X.append(x)
                    Y.append(y)

    elif method == 'Horizontal':
        s = len(image)
        for i in range(len(image)):
            flux = moving_average(np.array(image[:, i]), smooth_filter)
            peaks_find  , _ = signal.find_peaks(flux,   prominence=prominence, distance=distance, width=width, threshold=threshold) #, distance=d)
            for peak in (peaks_find):
                x = peak
                y = i
                if Mincut_Radius <= np.sqrt((x-s/2)**2 + (y-s/2)**2) <= Maxcut_Radius:
                    X.append(x)
                    Y.append(y)

    elif method == 'Vertical':
        s = len(image)
        for i in range(len(image)):
            flux = moving_average(np.array(image[i, :]), smooth_filter)
            peaks_find  , _ = signal.find_peaks(flux,   prominence=prominence, distance=distance, width=width, threshold=threshold) #, distance=d)
            for peak in (peaks_find):
                x = i
                y = peak
                if Mincut_Radius <= np.sqrt((x-s/2)**2 + (y-s/2)**2) <= Maxcut_Radius:
                    X.append(x)
                    Y.append(y)
    
    elif method == 'Azimuthal':
        Phi = np.radians(np.linspace(0, 359, nb_phi))
        if R_max == None:
            R = np.arange(0, int(len(image)/2), 1)
        else :
            R = np.arange(Mincut_Radius, Maxcut_Radius, 1)
        for phi in Phi:
            x = list(map(int, len(image)/2 + R*np.sin(phi)))
            y = list(map(int, len(image)/2 + R*np.cos(phi)))
            # points = list(zip(x, y))
            # points_uniques = list(set(points))
            # x, y = zip(*points_uniques)
            # x = list(x)
            # y = list(y)
            Flux    = image[x, y]


            flux = np.log(Flux)
            flux_smooth = moving_average(np.array(flux), smooth_filter)
            peaks_find, _ = signal.find_peaks(flux_smooth, prominence=prominence, distance=distance, width=width, threshold=threshold) #, distance=d)
            for peak in peaks_find:                
                X.append(len(image)/2 + (Mincut_Radius + peak)*np.sin(phi))
                Y.append(len(image)/2 + (Mincut_Radius + peak)*np.cos(phi))
    return X, Y

def Ellipse(incl, R, R_ref, H, PA, chi, xc, yc, Phi=np.radians(np.linspace(0, 359, 360))):
    """
    Compute ellipse using protoplanetary disk geometric parameters

    Parameters
    ----------

    incl        :   float
                    ellipse incl in radians
    
    R           :   float
                    Position of ellipse radius
    
    R_ref       :   float
                    Fitted Reference Radius

    h           :   float
                    Fitted reference Disk Scattering Surface Height in pixels

    PA          :   float
                    ellipse Position Angle in radians

    chi         :   float
                    Disk Scattering Surface flaring Exponent

    xc, yc      :   float 
                    ellipse center position in pixels <=> star position if ring and star are centered
    
    phi         :   ndarray (optional)
                    array of parametric variable (typically np.radians(np.linspace(0, 359, 360)))

    Returns
    -------
    (Xe, Ye) coordinates of ellipse points
    """
    x     = R * np.sin(Phi)
    y     = H * (R/R_ref)**chi * np.sin(incl) - R * np.cos(Phi) * np.cos(incl)
    Xe = - x * np.sin(PA) - y * np.cos(PA) + xc
    Ye =   x * np.cos(PA) - y * np.sin(PA) + yc
    return Xe, Ye

def uniform_ellipse(incl, PA, Height, Radius, chi, R, space, x0=0, y0=0, init=0):
    """
    Compute uniformly space points along ellipse.

    Parameters
    ----------

    incl        :   float
                    ellipse incl in radians
    PA          :   float
                    ellipse Position Angle in radians

    Height      :   float
                    Disk Scattering Surface Height in pixels

    Radius      :   float
                    Disk Scattering Surface Midplane Radius in pixels

    chi         :   float
                    Disk Scattering Surface flaring Exponent

    R           :   float
                    Radius where we begin the extraction  in pixels

    space       :   float
                    space between one azimuth and the next one. Usually equal to 2 * PSF radius
    
    x0, y0      :   float (optional)
                    ellipse center position in pixels
    
    init        :   float (optional)
                    angle where to begin ellipse in radians
    
    Returns
    -------
    (x, y) coordinates of homogenized ellipse points
    """
    # Étape 1 : Discrétiser l'ellipse avec une haute résolution
    t = np.linspace(0, 2 * np.pi, 10000)  # Résolution élevée pour une approximation précise
    x, y = Ellipse(incl, R, Radius, Height, PA, chi, x0, y0)
    # Étape 2 : Calculer les longueurs des segments entre les points
    distances = np.sqrt(np.diff(x)**2 + np.diff(y)**2)  # Distance entre chaque point
    cumulative_distances = np.cumsum(distances)         # Longueurs cumulées
    total_perimeter = cumulative_distances[-1]          # Périmètre total de l'ellipse
    # Étape 3 : Positions uniformes le long du périmètre
    uniform_distances = np.arange(init, total_perimeter + init, space)
    # uniform_distances = np.linspace(0, total_perimeter, num_points)
    # Étape 4 : Interpoler les positions (x, y) sur l'ellipse
    uniform_x = np.interp(uniform_distances, np.hstack(([0], cumulative_distances)), x)
    uniform_y = np.interp(uniform_distances, np.hstack(([0], cumulative_distances)), y)
    return uniform_x, uniform_y

def Random_Ellipse(Radius, Phi, xs, ys, incl, R, H, Chi, PA):
    x     = Radius * np.sin(Phi)
    y     = H * (Radius/R)**Chi * np.sin(incl) - Radius * np.cos(Phi) * np.cos(incl)
    x_rot = xs + (x * np.cos(np.pi - PA) - y * np.sin(np.pi - PA))
    y_rot = ys + (x * np.sin(np.pi - PA) + y * np.cos(np.pi - PA))
    return x_rot, y_rot

def My_PositionAngle(xcenter, ycenter, x0, y0, a, b, e, PA_LSFE):
    """
    Compute position angle counterclockwise with respect to y-axis

    Parameters
    ----------

    xcenter, ycenter    :   float
                            image center position
    
    x0, y0              :   float
                            ellipse center position
    
    a, b                :   float
                            semi-major and semi-minor axis

    e                   :   float
                            eccentricity

    PA_LSFE             :   float
                            rotation of the semi-major axis from the x-axis (computed during least square fitting of ellipse)
    
    Returns
    -------

    float
            Ellipse Position Angle
    """
    Xa, Ya = get_ellipse_pts((x0, y0, a, b, e, PA_LSFE+np.pi/2), liste=[np.pi/2, 3*np.pi/2])

    D1 = np.sqrt((xcenter-Xa[0])**2 + (ycenter-Ya[0])**2)
    D2 = np.sqrt((xcenter-Xa[1])**2 + (ycenter-Ya[1])**2)
    if D1 < D2 :
        X_PA = Xa[1] - Xa[0]
        Y_PA = Ya[1] - Ya[0]
    else:
        X_PA = Xa[0] - Xa[1]
        Y_PA = Ya[0] - Ya[1]
    PA = ((-np.arctan2(X_PA, Y_PA)) % (2*np.pi) - np.pi/2) % (2*np.pi)
    return PA

def Height_Compute(x, y, xs, ys):
    """
    Find the distance between the star and  orthogonal projection  of the equation of the line corresponding to the major axis of the ellipse
    defined by the points x, y.

    Parameters
    ----------
    x, y    :   ndarray
                list or array containing the coordinates of the points of the ellipse


    Returns
    -------

    tuple 
                (a, b, c) representing the coefficients of the line ax + by + c = 0
    """
    a, b, c = find_ellipse_axis_coeffs(x, y, axis='major')
    
    x_proj = (b * (b * xs - a * ys) - a * c) / (a**2 + b**2)
    y_proj = (a * (-b * xs + a * ys) - b * c) / (a**2 + b**2)

    distance = np.sqrt((xs - x_proj)**2 + (ys - y_proj)**2)
    return distance

def find_ellipse_axis_coeffs(x, y, axis='major'):
    """
    Find the coefficients of the equation of the line corresponding to the major axis of the ellipse
    defined by the points x, y.

    Parameters
    ----------

    x, y        :   ndarray
                    numpy list or array containing the coordinates of the points of the ellipse
    axis        :   str
                    choose "major" or "minor" axis

    Returns
    -------
    tuple
                    (a, b, c) representing the coefficients of the line ax + by + c = 0
    """
    # center datas
    x_mean, y_mean = np.mean(x), np.mean(y)
    centered_x, centered_y = x - x_mean, y - y_mean

    # compute the covariance matrix
    cov_matrix = np.cov(centered_x, centered_y)

    # Compute eigenvectors and eigenvalues
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

    # Find the eigenvector associated with the smallest eigenvalue
    if axis == 'major':
        axis_index = np.argmin(eigenvalues)
    else :
        axis_index = np.argmin(eigenvalues)

    axis_vector = eigenvectors[:, axis_index]

    # Compute the coefficients of the line
    a, b = axis_vector
    c = -a * x_mean - b * y_mean

    return a, b, c

def Load_Structure(filepath, Type='Struct'):
    """
    Load the fitted parameters from '.tfit' or '.pfit' files

    Parameters
    ----------

    filepath    :   str
                    full path to the '.tfit' or '.pfit' file

    Type        :   str (optional)
                    define what do you want : "Struct" for geometrical parameters and errors ; "Points" for x, y coordinates of Maxima points ; "Ellipse" for fitted ellipse coordinates
    
    Returns
    -------

    list
                List of selected parameters
    """
    with open(filepath, 'rb') as fichier:
        Loaded_Data = pkl.load(fichier)
    if Type == 'Struct':
        return Loaded_Data["params"], Loaded_Data["Err"]
    elif Type == 'Points':
        return Loaded_Data["Points"]
    elif Type == 'Ellipse':
        return Loaded_Data['Ellipse']

def filtrer_nuage_points(points, seuil, dec=0):
    # points = sorted(points, key=lambda p: (p[0], p[1]))  # Trier par X puis Y
    dq = deque(points)
    dq.rotate(-dec)
    points = list(dq)
    points_filtres = []
    for p in points:
        if all(np.linalg.norm(np.array(p) - np.array(q)) >= seuil for q in points_filtres):
            points_filtres.append(p)
    return np.array(points_filtres)

def Ellipse_Estimation(points, x0, y0):
    X = points[:, 0]
    Y = points[:, 1]
    coeffs = fit_ellipse(np.array(X), np.array(Y))
    Xc, Yc, a, b, e, PA_LSQE = cart_to_pol(coeffs)

    t = np.array([0, np.pi]) + np.pi/2
    X_edge = Xc + a*np.cos(t)*np.cos(PA_LSQE) - b*np.sin(t)*np.sin(PA_LSQE)
    Y_edge = Yc + a*np.cos(t)*np.sin(PA_LSQE) + b*np.sin(t)*np.cos(PA_LSQE)

    dist = np.sqrt((X_edge - x0)**2 + (Y_edge - y0)**2)
    idx  = np.argmax(dist)

    PA = (np.arctan2(X_edge[idx] - Xc, Y_edge[idx] - Yc) - np.pi) % (2*np.pi)

    incl = np.arccos(b/a)

    Xe, Ye         = get_ellipse_pts((Xc, Yc, a, b, e, PA))

    xc_p, yc_p  = orthogonal_projection((x0, y0), (Xc, Yc), PA + np.pi/2)

    D = np.sqrt((Xc - xc_p)**2 + (Yc - yc_p)**2)

    H              = D / np.sin(incl)
    R              = a
    Aspect         = H/R
    return incl, (X_edge[idx], Y_edge[idx]), R, H, Aspect, Xe, Ye, Xc, Yc

def Std_Ellipse(points, Xe, Ye, Xc, Yc):
    X = points[:, 0]
    Y = points[:, 1]
    Point_Offset = []
    for idx in range(len(X)):
        idx_dist     = np.argmin(np.sqrt((Xe - X[idx])**2 + (Ye - Y[idx])**2))
        ellipse_dist = np.sqrt((Xc - Xe[idx_dist])**2 + (Yc - Ye[idx_dist])**2)
        point_dist   = np.sqrt((Xc - X[idx])**2        + (Yc - Y[idx])**2)
        dist         = ellipse_dist - point_dist
        Point_Offset.append(dist)
    return np.std(Point_Offset)

def orthogonal_projection(A, B, theta):
    x_A, y_A = A
    x_B, y_B = B

    # Calcul des cosinus et sinus de l'angle
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    # Calcul du paramètre t
    t = (x_A - x_B) * cos_theta + (y_A - y_B) * sin_theta

    # Coordonnées du projeté orthogonal P
    x_P = x_B + t * cos_theta
    y_P = y_B + t * sin_theta

    return x_P, y_P
# =======================================================================================
# ============================ Scattering Phase Functions ===============================
# =======================================================================================

# Old Version of SPF Extraction, Using Extreme cases for eache geometric parameters -> compute SPF + 8 others for each parameters (+ and - error)
def Flux_Extraction_extreme_case(params):
    """
    Compute raw SPF using geometrical parameters, extraction zone and limb brightening effet

    Parameters
    ----------

    params      :   list
                    list of parameters used for SPF extraction
                    img (ndarray)       -   fits image
                    distance (float)    -   target distance in parsec
                    pixelscale (float)  -   pixelscale in arcsec/pixel
                    (xs, ys) (float)    -   star position in pixel
                    (Xc, Yc) (float)    -   fitted ellispe position in pixel
                    incl (float)        -   inclination in radian
                    PA (float)          -   Position Angle in radian
                    R_ref (float)       -   Reference Radius of fitted disk geometry
                    H_ref (float)       -   Reference Height of fitted disk geometry
                    aspect (float)      -   H_ref/R_ref
                    alpha (float)       -   power law index for local scattering height computation
                    D_incl (float)      -   Error on inclination in radian
                    D_R_ref (float)     -   Error on Reference Radius of fitted disk geometry
                    D_H_ref (float)     -   Error on Reference Height of fitted disk geometry
                    D_aspect (float)    -   Error on H_ref/R_ref
                    R_in, R_out (float) -   Limits of extraction zone
                    n_bin (int)         -   number of bins for scattering angles
                    Phi (ndarray)       -   azimuth for scanning the disk image
                    Type (str)          -   SPF type, if its for values or for error estimations
    
    Returns
    -------

    Dictionnary
                    Data set of SPF for All the disk, and for each side : Dataset = {'All' : All_disk, "East" : East_side, "West" : West_side}
                    behind each keys []"All", "East", "West"] : 
                    {"Sca"          : Scatt, 
                     "SPF"          : SPF, 
                     "Err_Sca"      : D_Scatt, 
                     "Err_SPF"      : D_SPF, 
                     "LB"           : LB_effect_tot, 
                     "Err_LB"       : D_LB_tot}

    """
    [img, distance, pixelscale, r_beam, (xs, ys), (xc_p, yc_p), incl, PA, R_ref, H_ref, alpha, D_incl,   D_R_ref,    D_H_ref,   R_in, R_out, n_bin, Phi, Corono_mask, Type] = params    
    D_aspect = np.sqrt((D_R_ref/R_ref)**2 + (D_H_ref/H_ref)**2)
    pixelscale_au = 648000/np.pi * distance * np.tan(np.radians(pixelscale/3600))
    nb_radius = round((R_out - R_in)/pixelscale_au) + 5 
    Phi = np.radians(Phi)
    Phi0 = Phi.copy()
    if alpha == 1:
        chi = 1.00001
    else :
        chi = alpha

    x0 = y0 = len(img)/2
    slope = np.tan(PA)
    intercept = y0 - x0*slope
    mask_idx = np.indices((len(img), len(img)))
    mask = mask_idx[0] > slope * mask_idx[1] + intercept
    img_all  = img.copy()
    img_east = img.copy()
    img_west = img.copy()
    img_east[~mask] = np.nan
    img_west[mask] = np.nan

    x_rot = np.zeros(nb_radius * len(Phi))
    y_rot = np.zeros(nb_radius * len(Phi))
    sca   = np.zeros(nb_radius * len(Phi))

    idx = 0
    for R in np.linspace(R_in/pixelscale_au, R_out/pixelscale_au, nb_radius):
        x_circle = xc_p + R * np.cos(PA - Phi)
        y_circle = yc_p + R * np.sin(PA - Phi)
        Phi = (PA - np.arctan2(y_circle - xs, x_circle - ys)) % (2*np.pi)
        for phi in Phi :
            x     = R * np.sin(phi)
            y     = H_ref * (R/R_ref)**chi * np.sin(incl) - R * np.cos(phi) * np.cos(incl)
            x_rot[idx] = x * np.cos(np.pi - PA) - y * np.sin(np.pi - PA) + xc_p
            y_rot[idx] = x * np.sin(np.pi - PA) + y * np.cos(np.pi - PA) + yc_p
            sca[idx] = np.arccos(np.cos(H_ref/R_ref) * np.cos(phi) * np.sin(incl) + np.sin(H_ref/R_ref) *np.cos(incl))
            idx += 1

    Imgs    = [img_all, img_east, img_west]
    bin_Sca = np.linspace(0, 180, n_bin+1)
    
    Dataset = {}
    
    for idx, Side_type in enumerate(['All', 'East', 'West']):
        image = Imgs[idx]
        image_coro = image.copy() / Corono_mask
        image_EZ = image.copy()
        Angle, Flux, Flux_Coro = [], [], []
        already_used_pixels = []
        for jdx in range(len(x_rot)):
            if (round(x_rot[jdx]), round(y_rot[jdx])) not in already_used_pixels:
                Flux.append(image[round(x_rot[jdx]), round(y_rot[jdx])])
                Flux_Coro.append(image_coro[round(x_rot[jdx]), round(y_rot[jdx])])
                Angle.append(np.degrees(sca[jdx]))
                already_used_pixels.append((round(x_rot[jdx]), round(y_rot[jdx])))
                image_EZ[round(x_rot[jdx]), round(y_rot[jdx])] = 0
                
        Angle, Flux, Flux_Coro = np.array(Angle), np.array(Flux), np.array(Flux_Coro)

        Sca,      D_Sca      = [], []
        SPF,      D_SPF      = [], []
        SPF_coro, D_SPF_coro = [], []

        for idx in range(n_bin):
            mask      = np.where(np.logical_and(Angle >= bin_Sca[idx], Angle < bin_Sca[idx+1]))
            if len(mask[0])!=0:
                Sca.append(np.nanmean(Angle[mask]))
                D_Sca.append(2 * np.nanstd(Angle[mask])/np.sqrt(len(Angle[mask])))
                SPF.append(np.nanmean(Flux[mask]))
                SPF_coro.append(np.nanmean(Flux_Coro[mask]))
                D_SPF.append(2 * np.nanstd(Flux[mask])/np.sqrt(len(Flux[mask])))
                D_SPF_coro.append(2 * np.nanstd(Flux_Coro[mask])/np.sqrt(len(Flux_Coro[mask])))

        N = np.cos(H_ref/R_ref) * np.sin(chi*H_ref/R_ref - H_ref/R_ref)
        D = np.cos(H_ref/R_ref) * np.sin(chi*H_ref/R_ref - H_ref/R_ref) + np.cos(incl) * np.cos(chi*H_ref/R_ref - H_ref/R_ref) - np.sin(chi*H_ref/R_ref) * np.cos(np.radians(Sca))
        dN_g  = - np.sin(H_ref/R_ref) * np.sin(chi*H_ref/R_ref - H_ref/R_ref) + np.cos(H_ref/R_ref) * (np.cos(chi*H_ref/R_ref - H_ref/R_ref) * (chi - 1))
        dD_g  = - np.sin(H_ref/R_ref) * np.sin(chi*H_ref/R_ref - H_ref/R_ref) + np.cos(H_ref/R_ref) * (np.cos(chi*H_ref/R_ref - H_ref/R_ref) * (chi - 1)) + np.cos(incl) * (-np.sin(chi*H_ref/R_ref - H_ref/R_ref) * (chi-1)) - np.cos(chi*H_ref/R_ref) * chi * np.cos(np.radians(Sca))
        dLB_g = (D * dN_g * N * dD_g)/D**2
        dLB_i = (-N*np.sin(incl)*np.cos(chi*H_ref/R_ref-H_ref/R_ref))/D**2
        dLB_s = (-N*np.sin(chi*H_ref/R_ref)*np.sin(np.radians(Sca)))/D**2

        LB_effect_tot = N/D
        D_LB_tot = np.sqrt((dLB_g * D_aspect)**2 + (dLB_i * D_incl)**2 + (dLB_s * D_Sca)**2)

        Sca,           D_Sca      = np.array(Sca),           np.abs(np.array(D_Sca))
        SPF,           D_SPF      = np.array(SPF),           np.abs(np.array(D_SPF))
        SPF_coro,      D_SPF_coro = np.array(SPF_coro),      np.abs(np.array(D_SPF_coro))
        LB_effect_tot, D_LB_tot   = np.array(LB_effect_tot), np.abs(np.array(D_LB_tot)) 

        Dataset[Side_type] = {"Sca" : Sca, "SPF" : SPF, "Err_Sca" : D_Sca, "Err_SPF" : D_SPF, "LB" : LB_effect_tot, "Err_LB" : D_LB_tot, "SPF_coro" : SPF_coro, "Err_SPF_coro" : D_SPF_coro, "EZ" : image_EZ}
    return Dataset

def SPF_Extreme(params, folderpath, File_name, img_type, add=''):
    """
    Launch SPF computation for All side taking into account errors on geometric parameters by computing SPF for extreme parameters. Multiprocessing is applied for SPF computation for each parameter errors
    
    Parameters
    ----------
    
    params      :   list
                    list of parameters used for SPF extraction
                    img (ndarray)       -   fits image
                    distance (float)    -   target distance in parsec
                    pixelscale (float)  -   pixelscale in arcsec/pixel
                    (xs, ys) (float)    -   star position in pixel
                    (Xc, Yc) (float)    -   fitted ellispe position in pixel
                    incl (float)        -   inclination in radian
                    PA (float)          -   Position Angle in radian
                    R_ref (float)       -   Reference Radius of fitted disk geometry
                    H_ref (float)       -   Reference Height of fitted disk geometry
                    alpha (float)       -   power law index for local scattering height computation
                    D_incl (float)      -   Error on inclination in radian
                    D_R_ref (float)     -   Error on Reference Radius of fitted disk geometry
                    D_H_ref (float)     -   Error on Reference Height of fitted disk geometry
                    R_in, R_out (float) -   Limits of extraction zone
                    n_bin (int)         -   number of bins for scattering angles
                    Phi (ndarray)       -   azimuth for scanning the disk image
                    Type (str)          -   SPF type, if its for values or for error estimations
    
    folderpath  :   str
                    full path to the saving folder \DRAGyS_Results

    File_name   :   str
                    name of the fits file with extension
    
    img_type    :   str
                    "Polarized" or "Total" to well save data in a appropriate filename
    """
    with Pool(processes=len(params)) as pool_SPF:
        resultats = pool_SPF.map(Flux_Extraction_extreme_case, params)
    Data_tot        = resultats[0]
    Data_MinPA      = resultats[1]
    Data_MaxPA      = resultats[2]
    Data_MinIncl    = resultats[3]
    Data_MaxIncl    = resultats[4]
    Data_MinH       = resultats[5]
    Data_MaxH       = resultats[6]
    Data_MinR       = resultats[7]
    Data_MaxR       = resultats[8]

    Dataset = {}
    Types  = ['All', "East", "West"]
    for T in Types:
        Scatt_tot                  = Data_tot[T]["Sca"]
        SPF_tot, Err_SPF_tot       = Data_tot[T]['SPF'],      Data_tot[T]['Err_SPF']
        SPF_coro, Err_SPF_coro     = Data_tot[T]['SPF_coro'], Data_tot[T]['Err_SPF_coro']

        Scatt_MinPA     = Data_MinPA[T]["Sca"]
        Scatt_MaxPA     = Data_MaxPA[T]["Sca"]
        Scatt_MinIncl   = Data_MinIncl[T]["Sca"]
        Scatt_MaxIncl   = Data_MaxIncl[T]["Sca"]
        Scatt_MinH      = Data_MinH[T]["Sca"]
        Scatt_MaxH      = Data_MaxH[T]["Sca"]
        Scatt_MinR      = Data_MinR[T]["Sca"]
        Scatt_MaxR      = Data_MaxR[T]["Sca"]

        SPF_MinPA     = np.interp(Scatt_tot, Scatt_MinPA,   Data_MinPA[T]['SPF'])
        SPF_MaxPA     = np.interp(Scatt_tot, Scatt_MaxPA,   Data_MaxPA[T]['SPF'])
        SPF_MinIncl   = np.interp(Scatt_tot, Scatt_MinIncl, Data_MinIncl[T]['SPF'])
        SPF_MaxIncl   = np.interp(Scatt_tot, Scatt_MaxIncl, Data_MaxIncl[T]['SPF'])
        SPF_MinH      = np.interp(Scatt_tot, Scatt_MinH,    Data_MinH[T]['SPF'])
        SPF_MaxH      = np.interp(Scatt_tot, Scatt_MaxH,    Data_MaxH[T]['SPF'])
        SPF_MinR      = np.interp(Scatt_tot, Scatt_MinR,    Data_MinR[T]['SPF'])
        SPF_MaxR      = np.interp(Scatt_tot, Scatt_MaxR,    Data_MaxR[T]['SPF'])

        sigma_min = np.ones_like(Scatt_tot)
        sigma_max = np.ones_like(Scatt_tot)
        for idx in range(len(Scatt_tot)):
            supp = np.max([SPF_MinPA[idx], SPF_MaxPA[idx], SPF_MinIncl[idx], SPF_MaxIncl[idx], SPF_MinH[idx], SPF_MaxH[idx], SPF_MinR[idx], SPF_MaxR[idx]]) - SPF_tot[idx]
            inff = SPF_tot[idx] - np.min([SPF_MinPA[idx], SPF_MaxPA[idx], SPF_MinIncl[idx], SPF_MaxIncl[idx], SPF_MinH[idx], SPF_MaxH[idx], SPF_MinR[idx], SPF_MaxR[idx]])
            if supp > 0:
                sigma_max[idx] = supp
            else:
                sigma_max[idx] = 0
            if inff > 0:
                sigma_min[idx] = inff
            else:
                sigma_min[idx] = 0

        Err_SPF_max = np.sqrt((Err_SPF_tot)**2 + sigma_max**2)
        Err_SPF_min = np.sqrt((Err_SPF_tot)**2 + sigma_min**2)

        SPF_MinPA_coro     = np.interp(Scatt_tot, Scatt_MinPA,   Data_MinPA[T]['SPF_coro'])
        SPF_MaxPA_coro     = np.interp(Scatt_tot, Scatt_MaxPA,   Data_MaxPA[T]['SPF_coro'])
        SPF_MinIncl_coro   = np.interp(Scatt_tot, Scatt_MinIncl, Data_MinIncl[T]['SPF_coro'])
        SPF_MaxIncl_coro   = np.interp(Scatt_tot, Scatt_MaxIncl, Data_MaxIncl[T]['SPF_coro'])
        SPF_MinH_coro      = np.interp(Scatt_tot, Scatt_MinH,    Data_MinH[T]['SPF_coro'])
        SPF_MaxH_coro      = np.interp(Scatt_tot, Scatt_MaxH,    Data_MaxH[T]['SPF_coro'])
        SPF_MinR_coro      = np.interp(Scatt_tot, Scatt_MinR,    Data_MinR[T]['SPF_coro'])
        SPF_MaxR_coro      = np.interp(Scatt_tot, Scatt_MaxR,    Data_MaxR[T]['SPF_coro'])

        sigma_min_coro = np.ones_like(Scatt_tot)
        sigma_max_coro = np.ones_like(Scatt_tot)
        for idx in range(len(Scatt_tot)):
            supp_coro = np.max([SPF_MinPA_coro[idx], SPF_MaxPA_coro[idx], SPF_MinIncl_coro[idx], SPF_MaxIncl_coro[idx], SPF_MinH_coro[idx], SPF_MaxH_coro[idx], SPF_MinR_coro[idx], SPF_MaxR_coro[idx]]) - SPF_coro[idx]
            inff_coro = SPF_coro[idx] - np.min([SPF_MinPA_coro[idx], SPF_MaxPA_coro[idx], SPF_MinIncl_coro[idx], SPF_MaxIncl_coro[idx], SPF_MinH_coro[idx], SPF_MaxH_coro[idx], SPF_MinR_coro[idx], SPF_MaxR_coro[idx]])
            if supp_coro > 0:
                sigma_max_coro[idx] = supp_coro
            else:
                sigma_max_coro[idx] = 0
            if inff > 0:
                sigma_min_coro[idx] = inff_coro
            else:
                sigma_min_coro[idx] = 0

        Err_SPF_max_coro = np.sqrt((Err_SPF_coro)**2 + sigma_max_coro**2)
        Err_SPF_min_coro = np.sqrt((Err_SPF_coro)**2 + sigma_min_coro**2)
        
        Dataset[T] = {'Sca': Scatt_tot, 
                      'SPF' : SPF_tot, 'Err_Sca' : Data_tot[T]['Err_Sca'], 'Err_SPF'      : [Err_SPF_min,      Err_SPF_max     ], 
                      'SPF_coro' : SPF_coro                                   , 'Err_SPF_coro' : [Err_SPF_min_coro, Err_SPF_max_coro], 
                      'LB' : Data_tot[T]['LB'], 'Err_LB' : Data_tot[T]['Err_LB'], 
                      "Params" : params[0],
                      "EZ" : Data_tot[T]["EZ"],
                      }
    filename = f"{folderpath}/DRAGyS_Results/{File_name[:-5]}{add}.{(img_type[0]).lower()}spf"
    with open(filename, 'wb') as Data_PhF:
        pkl.dump(Dataset, Data_PhF)

# New Version of SPF Extraction, Using N=100 Random set of geometric parameters (normal law) to compute SPF and Errorbar as mean values and standard error for each scattering angle bins
def Flux_Extraction(params):
    """
    Compute raw SPF using geometrical parameters, extraction zone and limb brightening effet

    Parameters
    ----------

    params      :   list
                    list of parameters used for SPF extraction
                    img (ndarray)       -   fits image
                    distance (float)    -   target distance in parsec
                    pixelscale (float)  -   pixelscale in arcsec/pixel
                    (xs, ys) (float)    -   star position in pixel
                    (Xc, Yc) (float)    -   fitted ellispe position in pixel
                    incl (float)        -   inclination in radian
                    PA (float)          -   Position Angle in radian
                    R_ref (float)       -   Reference Radius of fitted disk geometry
                    H_ref (float)       -   Reference Height of fitted disk geometry
                    aspect (float)      -   H_ref/R_ref
                    alpha (float)       -   power law index for local scattering height computation
                    D_incl (float)      -   Error on inclination in radian
                    D_R_ref (float)     -   Error on Reference Radius of fitted disk geometry
                    D_H_ref (float)     -   Error on Reference Height of fitted disk geometry
                    D_aspect (float)    -   Error on H_ref/R_ref
                    R_in, R_out (float) -   Limits of extraction zone
                    n_bin (int)         -   number of bins for scattering angles
                    Phi (ndarray)       -   azimuth for scanning the disk image
                    Type (str)          -   SPF type, if its for values or for error estimations
    
    Returns
    -------

    Dictionnary
                    Data set of SPF for All the disk, and for each side : Dataset = {'All' : All_disk, "East" : East_side, "West" : West_side}
                    behind each keys []"All", "East", "West"] : 
                    {"Sca"          : Scatt, 
                     "SPF"          : SPF, 
                     "Err_Sca"      : D_Scatt, 
                     "Err_SPF"      : D_SPF, 
                     "LB"           : LB_effect_tot, 
                     "Err_LB"       : D_LB_tot}

    """
    [img, distance, pixelscale, r_beam, (xs, ys), (xc_p, yc_p), incl, PA, R_ref, H_ref, chi, R_in, R_out, n_bin, Phi, Corono_mask] = params    
    pixelscale_au = 648000/np.pi * distance * np.tan(np.radians(pixelscale/3600))
    nb_radius = round((R_out - R_in)/pixelscale_au) + 5 
    Phi = np.radians(Phi)

    x0 = y0 = len(img)/2
    slope = np.tan(PA)
    intercept = y0 - x0*slope
    mask_idx = np.indices((len(img), len(img)))
    mask = mask_idx[0] > slope * mask_idx[1] + intercept

    x_rot = np.zeros(nb_radius * len(Phi))
    y_rot = np.zeros(nb_radius * len(Phi))
    sca   = np.zeros(nb_radius * len(Phi))

    idx = 0
    for R in np.linspace(R_in/pixelscale_au, R_out/pixelscale_au, nb_radius):
        x_circle = xc_p + R * np.cos(PA - Phi)
        y_circle = yc_p + R * np.sin(PA - Phi)
        Phi = (PA - np.arctan2(y_circle - xs, x_circle - ys)) % (2*np.pi)
        for phi in Phi :
            x     = R * np.sin(phi)
            y     = H_ref * (R/R_ref)**chi * np.sin(incl) - R * np.cos(phi) * np.cos(incl)
            x_rot[idx] = x * np.cos(np.pi - PA) - y * np.sin(np.pi - PA) + xc_p
            y_rot[idx] = x * np.sin(np.pi - PA) + y * np.cos(np.pi - PA) + yc_p
            sca[idx] = np.arccos(np.cos(H_ref/R_ref) * np.cos(phi) * np.sin(incl) + np.sin(H_ref/R_ref) *np.cos(incl))
            idx += 1

    img_all  = img.copy()
    img_east = img.copy()
    img_west = img.copy()
    img_east[~mask] = np.nan
    img_west[mask] = np.nan

    Imgs    = [img_all, img_east, img_west]    
    Dataset = {}
    
    for idx, Side_type in enumerate(['All', 'East', 'West']):
        image = Imgs[idx]
        Angle, Flux = [], []
        already_used_pixels = []
        for jdx in range(len(x_rot)):
            if (round(x_rot[jdx]), round(y_rot[jdx])) not in already_used_pixels:
                Flux.append(image[round(x_rot[jdx]), round(y_rot[jdx])])
                Angle.append(np.degrees(sca[jdx]))
                already_used_pixels.append((round(x_rot[jdx]), round(y_rot[jdx])))
        Angle, Flux = np.array(Angle), np.array(Flux)
        Dataset[Side_type] = [Angle, Flux]
    return Dataset

def SPF_Random(params, folderpath, File_name, img_type, add=''):
    # [self.img_chose, distance, pixelscale, r_beam, (xs, ys), (xc_P, yc_P), self.incl, self.PA, self.r_ref, self.h_ref, self.alpha, (D_xc_P, D_yc_P), self.D_incl, self.D_PA, self.D_r_ref, self.D_h_ref, R_in, R_out, n_bin, self.AzimuthalAngle, Corono_mask]
    [img,            distance, pixelscale, r_beam, (xs, ys), (xc_p, yc_p), incl,      PA,      R_ref,      H_ref,      chi,        (D_xc_p, D_yc_p), D_incl,      D_PA,      D_R_ref,      D_H_ref,      R_in, R_out, n_bin, Phi, Correct_corona] = params    
    
    conf = 1
    n_steps = 100

    D_aspect = np.sqrt((D_H_ref/H_ref)**2 + (D_R_ref/R_ref)**2)
    bin_Sca = np.linspace(0, 180, n_bin+1)

    x_center      = np.random.normal(xc_p,  conf*D_xc_p,  n_steps)
    y_center      = np.random.normal(yc_p,  conf*D_yc_p,  n_steps)
    inclination   = np.random.normal(incl,  conf*D_incl,  n_steps)
    positionangle = np.random.normal(PA,    conf*D_PA,    n_steps)
    height        = np.random.normal(H_ref, conf*D_H_ref, n_steps)
    radius        = np.random.normal(R_ref, conf*D_R_ref, n_steps)

    Pool_params = []
    for step in range(n_steps):
        Pool_params.append([img, distance, pixelscale, r_beam, (xs, ys), (x_center[step], y_center[step]), inclination[step], positionangle[step], radius[step], height[step], chi, R_in, R_out, n_bin, Phi, Correct_corona])
    
    with Pool(processes=n_steps) as pool_SPF:
        results = pool_SPF.map(Flux_Extraction, Pool_params)

    Dataset = {}
    for idx, Side_type in enumerate(['All', 'East', 'West']):
        All_Sca = []
        All_SPF = []
        for step in range(n_steps):
            All_Sca.append(results[step][Side_type][0])
            All_SPF.append(results[step][Side_type][1])

        A_Sca   = []
        D_A_Sca   = []
        A_SPF   = []
        D_A_SPF = []
        for adx in range(len(All_Sca)):     

            b_Sca   = np.zeros(n_bin)
            D_b_Sca = np.zeros(n_bin)

            b_SPF   = np.zeros(n_bin)
            D_b_SPF = np.zeros(n_bin)

            Angle = All_Sca[adx]
            Flux  = All_SPF[adx]
            for idx in range(n_bin):
                mask = np.where(np.logical_and(Angle >= bin_Sca[idx], Angle < bin_Sca[idx+1]))
                if len(mask) == 0:
                    b_Sca[idx]   = np.nan()
                    b_SPF[idx]   = np.nan()
                    D_b_Sca[idx] = np.nan()
                    D_b_SPF[idx] = np.nan()
                else :
                    b_Sca[idx]   = (bin_Sca[idx] + bin_Sca[idx+1])/2
                    D_b_Sca[idx] = np.abs(bin_Sca[idx] - bin_Sca[idx+1])/2
                    b_SPF[idx]   = np.nanmean(Flux[mask])
                    D_b_SPF[idx] = np.nanstd(Flux[mask])/np.sqrt(len(Flux[mask]))
            A_Sca.append(b_Sca)
            D_A_Sca.append(D_b_Sca)
            A_SPF.append(b_SPF)
            D_A_SPF.append(D_b_SPF)

        Sca   = np.nanmean(A_Sca, axis=0)
        D_Sca = np.nanmean(D_A_Sca, axis=0)
        SPF   = np.nanmean(A_SPF, axis=0)
        # D_SPF = np.sqrt(np.nanstd(A_SPF, axis=0)**2 + np.nanmean(D_A_SPF, axis=0)**2)
        D_SPF = np.nanstd(A_SPF, axis=0)

        notnan = np.isnan(SPF)
        Sca    = Sca[~notnan]
        D_Sca  = D_Sca[~notnan]
        SPF    = SPF[~notnan]
        D_SPF  = D_SPF[~notnan]


        vars       = [  H_ref,   R_ref,   incl]
        sigma_vars = [D_H_ref, D_R_ref, D_incl]

        # --- LB Compute et Uncertainties ---
        LB       = compute_LB(*vars, chi, Sca)
        partials = [partial_derivative(compute_LB, i, vars, args=(chi, Sca)) for i in range(3)]
        sigma_LB = np.sqrt(sum((partials[i] * sigma_vars[i])**2 for i in range(3)))

        Sca,           D_Sca      = np.array(Sca),           np.abs(np.array(D_Sca))
        SPF,           D_SPF      = np.array(SPF),           np.abs(np.array(D_SPF))

        Dataset[Side_type] = {"Sca" : Sca, "SPF" : SPF, "Err_Sca" : D_Sca, "Err_SPF" : D_SPF, "LB" : LB, "Err_LB" : sigma_LB, "All_Sca" : All_Sca, "All_SPF" : All_SPF, "Params" : params}
    
    if Correct_corona:
        filename = f"{folderpath}/DRAGyS_Results/{File_name[:-5]}{add}_CoronaCorrected.{(img_type[0]).lower()}spf"
    else:
        filename = f"{folderpath}/DRAGyS_Results/{File_name[:-5]}{add}.{(img_type[0]).lower()}spf"

    with open(filename, 'wb') as Data_PhF:
        pkl.dump(Dataset, Data_PhF)

    # return Dataset

def New_Beam_pSPF(params):
    """
    Compute raw SPF using geometrical parameters, extraction zone and limb brightening effet

    Parameters
    ----------

    params      :   list
                    list of parameters used for SPF extraction
                    img (ndarray)       -   fits image
                    distance (float)    -   target distance in parsec
                    pixelscale (float)  -   pixelscale in arcsec/pixel
                    (xs, ys) (float)    -   star position in pixel
                    (Xc, Yc) (float)    -   fitted ellispe position in pixel
                    incl (float)        -   inclination in radian
                    PA (float)          -   Position Angle in radian
                    R_ref (float)       -   Reference Radius of fitted disk geometry
                    H_ref (float)       -   Reference Height of fitted disk geometry
                    aspect (float)      -   H_ref/R_ref
                    alpha (float)       -   power law index for local scattering height computation
                    D_incl (float)      -   Error on inclination in radian
                    D_R_ref (float)     -   Error on Reference Radius of fitted disk geometry
                    D_H_ref (float)     -   Error on Reference Height of fitted disk geometry
                    D_aspect (float)    -   Error on H_ref/R_ref
                    R_in, R_out (float) -   Limits of extraction zone
                    n_bin (int)         -   number of bins for scattering angles
                    Phi (ndarray)       -   azimuth for scanning the disk image
                    Type (str)          -   SPF type, if its for values or for error estimations
    
    Returns
    -------

    Dictionnary
                    Data set of SPF for All the disk, and for each side : Dataset = {'All' : All_disk, "East" : East_side, "West" : West_side}
                    behind each keys []"All", "East", "West"] : 
                    {"Scatt"        : Scatt, 
                     "I"            : SPF, 
                     "PI"           : SPF, 
                     "Err_Scatt"    : D_Scatt, 
                     "Err_I"        : D_SPF, 
                     "Err_PI"       : D_SPF, 
                     "LB"           : LB_effect_tot, 
                     "Err_LB"       : D_LB_tot}

    """
    [img_I, img_PI, distance, pixelscale, r_beam, incl, PA, aspect, alpha, D_incl, D_aspect, R_in, R_out, n_bin, Type] = params
    pixelscale_au = 648000/np.pi * distance * np.tan(np.radians(pixelscale/3600))

    range_R   = r_beam * 180/np.pi * 3600 /pixelscale
    R_in, R_out = R_in/pixelscale, R_out/pixelscale
    Radius = np.arange(R_in, R_out, range_R)

    Beam_X, Beam_Y, Beam_Phi = BeamSpace(Radius, range_R/2, PA, incl, aspect, alpha)
    h, w = np.shape(img_PI)
    Y, X = np.ogrid[:h, :w]
    Angle,      TotI,      PolarI      = [], [], []
    Angle_east, TotI_east, PolarI_east = [], [], []
    Angle_west, TotI_west, PolarI_west = [], [], []

    size = len(img_PI)/2
    Side = np.zeros_like(img_PI)
    x0 = y0 = len(img_PI)/2
    slope = np.tan(PA)
    intercept = y0 - x0*slope
    mask_idx = np.indices((len(img_PI), len(img_PI)))
    mask = mask_idx[0] > slope * mask_idx[1] + intercept
    Side[mask] = 1

    for idx, R in enumerate(Radius):
        Phi = Beam_Phi[idx]
        for jdx, phi in enumerate(Phi):
            y = Beam_X[idx][jdx] + size
            x = Beam_Y[idx][jdx] + size
            sca  = np.arccos(np.cos(aspect) * np.cos(phi) * np.sin(incl) + np.sin(aspect) *np.cos(incl))
            mask = create_circular_mask(X, Y, (y, x), range_R/2)
            if Side[round(x), round(y)] == 1:
                Angle_east.append(np.degrees(sca))
                TotI_east.append(np.nanmean(img_I[mask]))
                PolarI_east.append(np.nanmean(img_PI[mask]))
            else :
                Angle_west.append(np.degrees(sca))
                TotI_west.append(np.nanmean(img_I[mask]))
                PolarI_west.append(np.nanmean(img_PI[mask]))
            TotI.append(np.nanmean(img_I[mask]))
            PolarI.append(np.nanmean(img_PI[mask]))
            Angle.append(np.degrees(sca))
    Angle,      TotI,      PolarI      = np.array(Angle),      np.array(TotI),      np.array(PolarI)
    Angle_east, TotI_east, PolarI_east = np.array(Angle_east), np.array(TotI_east), np.array(PolarI_east)
    Angle_west, TotI_west, PolarI_west = np.array(Angle_west), np.array(TotI_west), np.array(PolarI_west)

    Scatt, D_Scatt   = [], []
    PI, D_PI         = [], []
    I, D_I           = [], []

    Scatt_east, D_Scatt_east   = [], []
    PI_east, D_PI_east         = [], []
    I_east, D_I_east           = [], []

    Scatt_west, D_Scatt_west   = [], []
    PI_west, D_PI_west         = [], []
    I_west, D_I_west           = [], []

    bin_Scatt = np.linspace(0, 180, n_bin+1)

    for idx in range(n_bin):
        mask      = np.where(np.logical_and(Angle      >= bin_Scatt[idx], Angle      < bin_Scatt[idx+1]))
        mask_east = np.where(np.logical_and(Angle_east >= bin_Scatt[idx], Angle_east < bin_Scatt[idx+1]))
        mask_west = np.where(np.logical_and(Angle_west >= bin_Scatt[idx], Angle_west < bin_Scatt[idx+1]))
        if len(mask[0])!=0:
            Scatt.append(np.nanmean(Angle[mask]))
            D_Scatt.append(np.nanstd(Angle[mask])/len(Angle[mask]))
            I.append(np.nanmean(TotI[mask]))
            PI.append(np.nanmean(PolarI[mask]))
            D_I.append(np.nanstd(TotI[mask])/len(TotI[mask]))
            D_PI.append(np.nanstd(PolarI[mask])/len(PolarI[mask]))
        if len(mask_east[0]) != 0:
            Scatt_east.append(np.nanmean(Angle_east[mask_east]))
            D_Scatt_east.append(np.nanstd(Angle_east[mask_east])/len(Angle_east[mask_east]))
            I_east.append(np.nanmean(TotI_east[mask_east]))
            D_I_east.append(np.nanstd(TotI_east[mask_east])/len(TotI_east[mask_east]))
            PI_east.append(np.nanmean(PolarI_east[mask_east]))
            D_PI_east.append(np.nanstd(PolarI_east[mask_east])/len(PolarI_east[mask_east]))
        if len(mask_west[0]) != 0:
            Scatt_west.append(np.nanmean(Angle_west[mask_west]))
            D_Scatt_west.append(np.nanstd(Angle_west[mask_west])/len(Angle_west[mask_west]))
            I_west.append(np.nanmean(TotI_west[mask_west]))
            D_I_west.append(np.nanstd(TotI_west[mask_west])/len(TotI_west[mask_west]))
            PI_west.append(np.nanmean(PolarI_west[mask_west]))
            D_PI_west.append(np.nanstd(PolarI_west[mask_west])/len(PolarI_west[mask_west]))

    if alpha == 1:
        chi = 1.00001
    else :
        chi = alpha
    N = np.cos(aspect) * np.sin(chi*aspect - aspect)
    D = np.cos(aspect) * np.sin(chi*aspect - aspect) + np.cos(incl) * np.cos(chi*aspect - aspect) - np.sin(chi*aspect) * np.cos(np.radians(Scatt))
    dN_g  = - np.sin(aspect) * np.sin(chi*aspect - aspect) + np.cos(aspect) * (np.cos(chi*aspect - aspect) * (chi - 1))
    dD_g  = - np.sin(aspect) * np.sin(chi*aspect - aspect) + np.cos(aspect) * (np.cos(chi*aspect - aspect) * (chi - 1)) + np.cos(incl) * (-np.sin(chi*aspect - aspect) * (chi-1)) - np.cos(chi*aspect) * chi * np.cos(np.radians(Scatt))
    dLB_g = (D * dN_g * N * dD_g)/D**2
    dLB_i = (-N*np.sin(incl)*np.cos(chi*aspect-aspect))/D**2
    dLB_s = (-N*np.sin(chi*aspect)*np.sin(np.radians(Scatt)))/D**2

    N_east = np.cos(aspect) * np.sin(chi*aspect - aspect)
    D_east = np.cos(aspect) * np.sin(chi*aspect - aspect) + np.cos(incl) * np.cos(chi*aspect - aspect) - np.sin(chi*aspect) * np.cos(np.radians(Scatt_east))
    dN_g_east  = - np.sin(aspect) * np.sin(chi*aspect - aspect) + np.cos(aspect) * (np.cos(chi*aspect - aspect) * (chi - 1))
    dD_g_east  = - np.sin(aspect) * np.sin(chi*aspect - aspect) + np.cos(aspect) * (np.cos(chi*aspect - aspect) * (chi - 1)) + np.cos(incl) * (-np.sin(chi*aspect - aspect) * (chi-1)) - np.cos(chi*aspect) * chi * np.cos(np.radians(Scatt_east))
    dLB_g_east = (D * dN_g * N * dD_g)/D**2
    dLB_i_east = (-N*np.sin(incl)*np.cos(chi*aspect-aspect))/D**2
    dLB_s_east = (-N*np.sin(chi*aspect)*np.sin(np.radians(Scatt_east)))/D**2

    N_west = np.cos(aspect) * np.sin(chi*aspect - aspect)
    D_west = np.cos(aspect) * np.sin(chi*aspect - aspect) + np.cos(incl) * np.cos(chi*aspect - aspect) - np.sin(chi*aspect) * np.cos(np.radians(Scatt_west))
    dN_g_west  = - np.sin(aspect) * np.sin(chi*aspect - aspect) + np.cos(aspect) * (np.cos(chi*aspect - aspect) * (chi - 1))
    dD_g_west  = - np.sin(aspect) * np.sin(chi*aspect - aspect) + np.cos(aspect) * (np.cos(chi*aspect - aspect) * (chi - 1)) + np.cos(incl) * (-np.sin(chi*aspect - aspect) * (chi-1)) - np.cos(chi*aspect) * chi * np.cos(np.radians(Scatt_west))
    dLB_g_west = (D * dN_g * N * dD_g)/D**2
    dLB_i_west = (-N*np.sin(incl)*np.cos(chi*aspect-aspect))/D**2
    dLB_s_west = (-N*np.sin(chi*aspect)*np.sin(np.radians(Scatt_west)))/D**2

    LB_effect_tot = N/D
    D_LB_tot = np.sqrt((dLB_g * D_aspect)**2 + (dLB_i * D_incl)**2 + (dLB_s * D_Scatt)**2)

    LB_effect_east = N_east/D_east
    D_LB_east = np.sqrt((dLB_g_east * D_aspect)**2 + (dLB_i_east * D_incl)**2 + (dLB_s_east * D_Scatt_east)**2)

    LB_effect_west = N_west/D_west
    D_LB_west = np.sqrt((dLB_g_west * D_aspect)**2 + (dLB_i_west * D_incl)**2 + (dLB_s_west * D_Scatt_west)**2)

    Scatt, Scatt_east, Scatt_west = np.array(Scatt),  np.array(Scatt_east), np.array(Scatt_west)
    I,         PI,         D_I,         D_PI         = np.array(I),         np.array(PI),         np.abs(np.array(D_I)),         np.abs(np.array(D_PI))
    I_east,    PI_east,    D_I_east,    D_PI_east    = np.array(I_east),    np.array(PI_east),    np.abs(np.array(D_I_east)),    np.abs(np.array(D_PI_east))
    I_west,    PI_west,    D_I_west,    D_PI_west    = np.array(I_west),    np.array(PI_west),    np.abs(np.array(D_I_west)),    np.abs(np.array(D_PI_west))
    All_disk          = {"Scatt" : Scatt,      "I" : I,              "PI" : PI,        "Err_Scatt" : D_Scatt,      "Err_I" : D_I,              "Err_PI" : D_PI,      "LB" : LB_effect_tot,  "Err_LB" : D_LB_tot}
    East_side         = {"Scatt" : Scatt_east, "I" : I_east,         "PI" : PI_east,   "Err_Scatt" : D_Scatt_east, "Err_I" : D_I_east,         "Err_PI" : D_PI_east, "LB" : LB_effect_east, "Err_LB" : D_LB_east}
    West_side         = {"Scatt" : Scatt_west, "I" : I_west,         "PI" : PI_west,   "Err_Scatt" : D_Scatt_west, "Err_I" : D_I_west,         "Err_PI" : D_PI_west, "LB" : LB_effect_west, "Err_LB" : D_LB_west}
    Dataset = {'All' : All_disk, "East" : East_side, "West" : West_side}
    return Dataset

def colorline_with_error(x, y, cmap='viridis', alpha=1.0, linestyle='-', linewidth=2, label=None):
    """Plot a colored curve"""
    # Création des segments colorés
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    norm = plt.Normalize(x.min(), x.max())  # Normalisation des couleurs
    lc = mcoll.LineCollection(segments, cmap=cmap, norm=norm, alpha=alpha, linestyle=linestyle, linewidth=linewidth)
    lc.set_array(x)  # Définit les couleurs selon x
    return lc

# for considering beam size when extract flux
def get_mean_value_in_circle(image, center, radius):
    """
    Computes the average value inside the circle

    Parameters
    ----------

    image   :   ndarray
                fits image

    center  :   float
                pixel position

    radius  :   float
                circle radius in pixel

    Returns
    -------

    float
            :   mean value in the circle
    """
    cx, cy = center
    y, x = np.ogrid[:image.shape[0], :image.shape[1]]
    mask = (x - cx)**2 + (y - cy)**2 <= radius**2
    # return image[mask].mean()
    return np.nanmean(image[mask])

def BeamSpace(Radius, r_beam, PA, incl, aspect, alpha):
    """
    Compute coordinates and azimuthal angle of ellipse points taking into account the size of PSF

    Parameters
    ----------

    Radius      :   float
                    Ellipse radius for disk geometry equations

    r_beam      :   float
                    size of the circle PSF

    PA          :   float
                    position angle in radian

    incl        :   float
                    inclination in radian

    aspect      :   float
                    h/r of estimated disk geometry

    alpha       :   float
                    power law index for local scattering height equation
    
    Returns
    -------
    list
                    Beam_X is x-coordinates of ellipse point spaced by a resolution element
    list
                    Beam_X is y-coordinates of ellipse point spaced by a resolution element
    list
                    Beam_Phi is the list of azimuthal angle to have ellipse point spaced by a resolution element
 
    """
    Beam_X, Beam_Y, Beam_Phi= [], [], []
    for R in Radius :
        Phi = np.linspace(0, 2*np.pi -np.pi/180, 10000)
        x_rot = (R * np.sin(Phi)) * np.cos(np.pi - PA) - (aspect * R**alpha * np.sin(incl) - R * np.cos(Phi) * np.cos(incl)) * np.sin(np.pi - PA)
        y_rot = (R * np.sin(Phi)) * np.sin(np.pi - PA) + (aspect * R**alpha * np.sin(incl) - R * np.cos(Phi) * np.cos(incl)) * np.cos(np.pi - PA)
        xc, yc = np.mean(x_rot), np.mean(y_rot)
        x0, y0, phi0 = x_rot[0], y_rot[0], Phi[0]
        new_x, new_y, new_Phi= [x0], [y0], [phi0]
        for i in range(1, len(x_rot)):
            x1, y1 = x_rot[i], y_rot[i]
            dist = np.sqrt((x0-x1)**2 + (y0-y1)**2)
            if dist >= 2*r_beam:
                x0, y0 = x1, y1
                new_x.append(x1)
                new_y.append(y1)
                new_Phi.append(Phi[i])
        Beam_X.append(new_x)
        Beam_Y.append(new_y)
        Beam_Phi.append(new_Phi)          
    return Beam_X, Beam_Y, Beam_Phi

def create_circular_mask(Y, X, center, radius):
    """
    Create a circle mask on image

    Parameters
    ----------

    Y, X        :   ndarray
                    coordinates of points
    center      :   tuple
                    (x0, y0) image center
    radius      :   float
                    circle radius
    
    Returns
    -------
    
    ndarray
                    circular mask
    """
    dist_from_center = np.sqrt((X - center[0])**2 + (Y - center[1])**2)
    mask = dist_from_center < radius
    return mask

def Normalize(Scatt, I, PI, Err_I, Err_PI, Type='Norm'):
    if Type=='Norm':
        Norm_I        = np.max(I)
        Norm_PI       = np.max(PI)
    elif Type=='90':
        Norm_I        = np.interp(90, Scatt, I)
        Norm_PI       = np.interp(90, Scatt, PI)
    I, PI         = I/Norm_I, PI/Norm_PI
    Err_I, Err_PI = Err_I/Norm_I, Err_PI/Norm_PI
    return I, PI, np.abs(Err_I), np.abs(Err_PI)

# Take into account non centered star position
def NewScatt(R, ellipse_center, Phi, star=(0, 0)):
    """
    Compute Scattering angle to take into account non centered stars

    Parameters
    ----------

    R               :   float
                        Radius
    
    ellipse_center  :   tuple
                        (x, y) coordinates of ellipse center

    Phi             :   ndarray
                        array of azimuthal angle to compute ellipse
    
    star            :   tuple
                        (xs, ys) coordinates of star position
    
    Returns
    -------

    ndarray
                        array of new scattering angle computed
    """
    xs, ys = star
    xc, yc = ellipse_center
    xp, yp = xc + R * np.cos(Phi), yc + R * np.sin(Phi)
    xl, yl = xc + R * np.cos(0),   yc + R * np.sin(0)
    x_SL, y_SL = xl - xs, yl - ys
    x_SP, y_SP = xp - xs, yp - ys

    SL = np.sqrt(x_SL**2 + y_SL**2)
    SP = np.sqrt(x_SP**2 + y_SP**2)

    SL_dot_SP = x_SP*x_SL + y_SP*y_SL
    det_SL_SP = x_SL*y_SP - y_SL*x_SP

    New_Phi = np.arccos(SL_dot_SP/(SL*SP))
    New_Phi = np.abs(np.where(det_SL_SP < 0, New_Phi, 2*np.pi - New_Phi) - 2*np.pi)
    return New_Phi

def angle_oriente_2d(v1, v2):
    """
    Compute oriented angle between two vectors
    
    Parameters
    ----------

    v1, v2  :   float
                vectors to compute oriented angle
    
    Returns
    -------
    
    float
                oriented angle

    """
    dot_product   = np.dot(v1, v2)
    norm_v1       = np.linalg.norm(v1)
    norm_v2       = np.linalg.norm(v2)
    no_oriented   = np.arccos(dot_product / (norm_v1 * norm_v2))
    cross_product = v1[0] * v2[1] - v1[1] * v2[0]
    if cross_product < 0:
        oriented = -no_oriented
    else:
        oriented = no_oriented
    return oriented

def Non_Centered_Star_AzimithalAngle(R, star_position, ellipse_center, PA, Phi):
    """
    Compute new scattering angle and orthogonal projection of non centered star system

    Parameters
    ----------
    R                   :   float
                            Radius

    star_position       :   tuple
                            (xs, ys) star position
    
    ellipse_center      :   tuple
                            (xc, yc) ellipse position
    
    PA                  :   float
                            Position Angle in radian

    Phi                 :   ndarray
                            array of azimuthal angle to compute ellipse
    
    
    Returns
    -------

    New_Phi             :   ndarray
                            array of new scattering angle computed

    xc_prime            :   float
                            x-coordinate of orthogonal projection

    yc_prime            :   float
                            y-coordinate of orthogonal projection
    """
    (xs, ys) = star_position
    (xc, yc) = ellipse_center
    a = np.cos(-PA)
    b = np.sin(-PA)
    xc_prime, yc_prime = orthogonal_projection((xs, ys), (xc, yc), PA)
    X_PA = a + xs
    Y_PA = b + ys

    X_SC = xc - xs
    Y_SC = yc - ys
    X_SPA = X_PA - xs
    Y_SPA = Y_PA - ys
    angle = angle_oriente_2d((X_SPA, Y_SPA), (X_SC, Y_SC))
    if 0 <= angle <= np.pi/2:
        D =   np.sqrt((xs - xc_prime)**2 + (ys - yc_prime)**2)
    elif np.pi/2 < angle :
        D = - np.sqrt((xs - xc_prime)**2 + (ys - yc_prime)**2)
    elif -np.pi/2 <= angle <= 0:
        D = - np.sqrt((xs - xc_prime)**2 + (ys - yc_prime)**2)
    elif -np.pi/2 > angle:
        D =   np.sqrt((xs - xc_prime)**2 + (ys - yc_prime)**2)

    New_Phi = NewScatt(R, (xs + D, ys), Phi, star=(xs, ys))
    return New_Phi, xc_prime, yc_prime

# Get Phase Function Data
def Get_PhF(filename, side='All', LBCorrected=False, norm='none'):
    """
    Get SPF from pickle files
    """
    with open(filename, 'rb') as fichier:
        Loaded_Data = pkl.load(fichier)
    Scatt     = np.array(Loaded_Data[side]["Scatt"])
    I         = np.array(Loaded_Data[side]["I"])
    PI        = np.array(Loaded_Data[side]["PI"])
    Err_Scatt = np.array(Loaded_Data[side]["Err_Scatt"])
    Err_I     = np.array(Loaded_Data[side]["Err_I"])
    Err_PI    = np.array(Loaded_Data[side]["Err_PI"])
    LB        = np.array(Loaded_Data[side]["LB"])
    Err_LB    = np.array(Loaded_Data[side]["Err_LB"])
    
    if LBCorrected:
        I  = I/LB
        PI = PI/LB
        Err_I  = I  * np.sqrt((Err_I/I)**2   + (Err_LB/LB)**2)
        Err_PI = PI * np.sqrt((Err_PI/PI)**2 + (Err_LB/LB)**2)

    if norm=='90':
        normI  = np.interp(90, Scatt, I)
        normPI = np.interp(90, Scatt, PI)
    else :
        normI = normPI = 1

    return Scatt, I/normI, PI/normPI, Err_Scatt, Err_I/normI, Err_PI/normPI, LB, Err_LB

def Get_SPF(filename, side='All', LBCorrected=False, norm='none', coronagraph_transmission=False):          # Not Used Yet
    with open(filename, 'rb') as fichier:
        Loaded_Data = pkl.load(fichier)

    Scatt     = np.array(Loaded_Data[side]["Sca"])
    Err_Scatt = np.array(Loaded_Data[side]["Err_Sca"])
    if coronagraph_transmission == True:
        SPF       = np.array(Loaded_Data[side]["SPF_coro"])
        Err_SPF   = np.array(Loaded_Data[side]["Err_SPF_coro"])
    else:
        SPF       = np.array(Loaded_Data[side]["SPF"])
        Err_SPF   = np.array(Loaded_Data[side]["Err_SPF"])
    LB        = np.array(Loaded_Data[side]["LB"])
    Err_LB    = np.array(Loaded_Data[side]["Err_LB"])

    if LBCorrected:
        Err_SPF = SPF/LB * np.sqrt((Err_SPF/SPF)**2 + (Err_LB/LB)**2)
        SPF  = SPF/LB

    if norm == 'max':
        Norm = np.max(SPF)
    elif norm == 'first':
        Norm = SPF[0]
    elif norm == '90':
        Norm = np.interp(90, Scatt, SPF)
    elif norm == 'mean90':
        Norm = np.mean(np.interp(np.linspace(80, 100, 20), Scatt, SPF))
    else :
        Norm = 1

    return Scatt, SPF/Norm, LB, Err_Scatt, Err_SPF/Norm, Err_LB

def MCFOST_PhaseFunction(file_path, Normalization):
    """
    Get MCFOST SPF from simulated files

    Parameters
    ----------
    file_path       :   str
                        path to the folder that contains "data_{lambda}" and "data_dust" folders
    
    Normalization   :   Bool
                        to normalized to 90° scattering angle
    
    Returns
    -------
    MCFOST_Scatt        :   ndarray
                            MCFOST Scattering Angle

    MCFOST_I            :   ndarray
                            MCFOST total intensity SPF

    MCFOST_PI           :   ndarray
                            MCFOST polarized intensity SPF

    MCFOST_DoP          :   ndarray
                            MCFOST Degree of polarization

    MCFOST_Err_I        :   ndarray
                            Error on MCFOST total intensity SPF

    MCFOST_Err_PI       :   ndarray
                            Error on MCFOST polarized intensity SPF

    MCFOST_Err_DoP      :   ndarray
                            Error on MCFOST Degree of polarization


    """
    if 'data_dust' not in os.listdir(file_path):
        print("No MCFOST data_dust folder here...")
        MCFOST_Scatt, MCFOST_I, MCFOST_PI, MCFOST_DoP, MCFOST_Err_I, MCFOST_Err_PI, MCFOST_Err_DoP = [0], [0], [0], [0], [0], [0], [0]
    else :
        MCFOST_folder = file_path + '/data_dust/'
        Lambda        = np.array(fits.getdata(file_path + '/data_dust/' + 'lambda.fits.gz'))
        Lambda_idx    = min(range(len(Lambda)), key=lambda i: abs(Lambda[i] - 1.6))
        polar         = fits.getdata(file_path + '/data_dust/' +'polarizability.fits.gz')[:, Lambda_idx]
        MCFOST_I      = fits.getdata(file_path + '/data_dust/' +'phase_function.fits.gz')[:, Lambda_idx]
        MCFOST_Scatt   = np.linspace(0, 180, len(MCFOST_I))
    
        MCFOST_PI      = MCFOST_I*polar
        # MCFOST_DoP     = MCFOST_PI / MCFOST_I
        MCFOST_DoP = polar
        MCFOST_I  = MCFOST_I/np.max(MCFOST_I)
        MCFOST_PI = MCFOST_PI/np.max(MCFOST_PI)

        MCFOST_Err_I, MCFOST_Err_PI = np.zeros_like(MCFOST_I), np.zeros_like(MCFOST_PI)
        MCFOST_Err_DoP = np.sqrt((MCFOST_Err_PI/MCFOST_I)**2 + (MCFOST_Err_I*MCFOST_PI/MCFOST_I**2)**2)
        if Normalization: 
            MCFOST_I, MCFOST_PI, MCFOST_Err_I, MCFOST_Err_PI = Normalize(MCFOST_Scatt, MCFOST_I, MCFOST_PI, MCFOST_Err_I, MCFOST_Err_PI, Type='90')

    return MCFOST_Scatt, MCFOST_I, MCFOST_PI, MCFOST_DoP, MCFOST_Err_I, MCFOST_Err_PI, MCFOST_Err_DoP

# Compute R2-scaled
def R2_scaled(img, incl, PA, R2_type='midplane'):
    x0 = y0 = len(img)/2
    meshy, meshx = np.meshgrid(np.linspace(0, len(img)-1, len(img)),
                                np.linspace(0, len(img)-1, len(img)))
    if R2_type == 'midplane':
        Phi = np.arctan(((meshx-x0)*np.cos(np.pi-PA) + (meshy-y0)*np.sin(np.pi-PA))/((meshx-x0)*np.sin(np.pi-PA) - (meshy-y0)*np.cos(np.pi-PA))*np.cos(incl))
        R2  = ((meshx-x0)**2 + (meshy-y0)**2)/(1 - (np.sin(incl)**2)*(np.cos(Phi)**2))
    else :
        R2  = (meshx - x0)**2 + (meshy- y0)**2
    return R2

# =======================================================================================
# =============================     Computation Tool      ===============================
# =======================================================================================

def LimbBrightening(Scatt, incl, aspect, chi):
    """
    Compute Limb Brightening

    Parameters
    ----------

    Scatt       :   ndarray
                    Scattering Angle
    
    incl        :   float
                    inclination in radian
            
    aspect      :   float
                    Height/Radius ratio
    
    chi         :   float
                    scattering height surface power-law index

    Returns
    -------

    ndarray     
                Limb Brightening effect with respect to the scattering angle
    """
    return (np.cos(aspect) * np.sin(aspect*chi-aspect)) / (np.cos(aspect)*np.sin(aspect*chi-aspect) + np.cos(incl)*np.cos(aspect*chi-aspect) - np.sin(aspect*chi)*np.cos(np.radians(Scatt)))

def compute_LB(H_ref, R_ref, incl, chi, Sca):
    x = H_ref / R_ref
    theta = (chi - 1) * x
    Sca_rad = np.radians(Sca)

    num   = np.cos(x) * np.sin(theta)
    denom = np.cos(x) * np.sin(theta) + np.cos(incl) * np.cos(theta) - np.sin(chi * x) * np.cos(Sca_rad)

    return num / denom

def partial_derivative(func, var_idx, vars, args=(), h=1e-5):
    vars_forward = vars[:]
    vars_backward = vars[:]
    vars_forward[var_idx] += h
    vars_backward[var_idx] -= h
    return (func(*vars_forward, *args) - func(*vars_backward, *args)) / (2 * h)

# =======================================================================================
# ==================================     Images      ====================================
# =======================================================================================

def Images_Opener(filepath):
    """
    Open data cube fits files for several types of data

    Parameters
    ----------

    filepath        :   str
                        path to the fits file
    
    Returns
    -------

    IMG         :   list
                    list of 6 images data (all equal to zeros arrays except if data are founded in fits file)
    Tresh       :   list
                    list of thresholds from images to avoid LogNorm error and use SymLogNorm (useful for MCFOST data !!)
    Data_Type   :   str
                    Check type of image, if its MCFOST image, it will use SymLogNorm instead of LogNorm
    """
    img    = fits.getdata(filepath)
    nb_dim = len(np.shape(img))
    img_0,    img_1,    img_2,    img_3,    img_4,    img_5    = None, None, None, None, None, None
    thresh_0, thresh_1, thresh_2, thresh_3, thresh_4, thresh_5 = None, None, None, None, None, None
    IMG    = [img_0,    img_1,    img_2,    img_3,    img_4,    img_5]
    Thresh = [thresh_0, thresh_1, thresh_2, thresh_3, thresh_4, thresh_5]

    if nb_dim == 2 :
        IMG[0]    = img
        mask = np.where(img > 0.)
        if len(img[mask]) == 0:
            Thresh[0] = 1e-40
        else :
            Thresh[0] = np.nanmin(img[mask])
        for nb in range(1, 6):
            IMG[nb] = np.zeros_like(img) + 1
            Thresh[nb] = 1e-40
        Data_Type = "Simple_Data_-_Total_or_Polarized"

    elif nb_dim == 5:  # if it is an MCFOST Simulated Image, dimension is [8, 1, 1, dim, dim]
        # threshold = 0.00000
        Image_0 = np.abs(img[0, 0, 0])

        Q = img[1, 0, 0]
        U = img[2, 0, 0]
        center = len(Q)/2
        X = np.arange(1, len(Q) + 1) - center
        Y = np.arange(1, len(Q) + 1) - center
        X, Y = np.meshgrid(X, Y)
        Phi = np.arctan2(Y, X)

        Image_1 = np.abs(- Q*np.cos(2*Phi) - U*np.sin(2*Phi))   # Q_phi
        Image_2 = np.abs(  Q*np.sin(2*Phi) - U*np.cos(2*Phi))   # U_phi
        Image_3 = np.abs(img[3, 0, 0])
        Image_4 = np.abs(np.sqrt(Image_1**2 + Image_2**2))
        Image_5 = np.zeros_like(Image_0) + 1
        for idx, image in enumerate([Image_0, Image_1, Image_2, Image_3, Image_4, Image_5]):
            IMG[idx] = image
            mask = np.where(image > 0.)
            if len(image[mask]) == 0:
                Thresh[idx] = 1e-40
            else :
                Thresh[idx] = np.nanmin(image[mask])
        Data_Type = "MCFOST_Data"
    else :
        for nb in range(len(IMG)):
            if nb < np.shape(img)[0]:
                IMG[nb] = img[nb, :, :]
                mask = np.where(img[nb, :, :] > 0.)
                if len((img[nb, :, :])[mask]) == 0:
                    Thresh[nb] = 1e-40
                else :
                    Thresh[nb] = np.nanmin((img[nb, :, :])[mask])

            else :
                IMG[nb]    = np.zeros_like(img[0, :, :]) + 1
                Thresh[nb] = 1e-40
        Data_Type = "Observational_Data_-_SPHERE"
    return IMG, Thresh, Data_Type

def moving_average(data, window_size):
    """
    Compute smooth profile

    Parameters
    ----------

    data            :   ndarray
                        radial profile

    window_size     :   float
                        smooth parameter
    
    Returns
    -------

    smoothed_data   :   ndarray
                        smoothed radial profile   
    """
    smoothed_data = np.convolve(data, np.ones(window_size)/window_size, mode='same')
    return smoothed_data

def DiskDistances():
    """
    Stock Distance of targets that I have
    """
    Distances = {}

    Distances["MCFOST"]   = 100.0

    Distances["HD34282"]  = 308.6
    Distances["HD163296"] = 101.0
    Distances["LkCa15"]   = 157.2
    Distances["V4046"]   = 71.5
    Distances["PDS 66"]   = 97.9
    Distances["RX J1615"] = 155.6
    Distances["RX J1852"] = 147.1
    Distances["hr4796"]   = 71.9

    Distances["CI_Tau"]    = 160.3
    Distances["DM_Tau"]    = 144
    Distances["GG_Tau"]    = 145.5
    Distances["GM_Aur"]    = 158.4
    Distances["HD_31648"]  = 156.3
    Distances["HD_97048"]  = 184.4
    Distances["Hen_3-545"] = 185.2
    Distances["IP_Tau"]    = 129.4
    Distances["LkCa_15"]   = 157.2
    Distances["SY_Cha"]    = 180.7
    Distances["SZCha"]     = 190.2
    Distances["UX_Tau"]    = 142.2
    Distances["UX_TAU"]    = 142.2
    Distances["V351_Ori"]  = 323.82
    Distances["V599_Ori"]  = 401.07

    Distances["J16090075"]  = 137.4
    Distances["J16120668"]  = 131.97
    Distances["BD-08_1115"] = 158.39
    Distances["DL_Tau"]     = 159.9
    Distances["DoAr_25"]    = 138.16
    Distances["Haro_1-1"]   = 151.1
    Distances["HD_34700"]   = 156.29
    Distances["HD_100453"]  = 350.5
    Distances["HD_142527"]  = 103.8
    Distances["TYC_6213"]   = 154.6

    return Distances

def Get_Distance(Path):
    Distances = DiskDistances()
    for key in Distances.keys():
        if key in Path:
            return Distances[key]        

def Get_Band(Path):
    if "_H." in Path or "_H_" in Path:
        return 1.65
    elif "_J." in Path or "_J_" in Path:
        return 1.25
    elif "_K." in Path or "_K_" in Path or "_Ks." in Path or "_Ks_" in Path:
        return 2.18
    else :
        return 1.65

def EZ_Radius(Path):
    """
    Stock Distance of targets that I have
    """
    Radius = {}

    Radius["MCFOST"]     = [70, 80]

    Radius["HD34282"]    = [190, 220]
    Radius["HD163296"]   = [55, 75]
    Radius["LkCa15"]     = [58, 94]
    Radius["V4046"]      = [23, 34]
    Radius["PDS 66"]     = [64, 104]
    Radius["RX J1615"]   = [146, 181]
    Radius["RX J1852"]   = [36, 68]
    Radius["hr4796"]     = [36, 68]

    Radius["CI_Tau"]     = [50, 60]
    Radius["DM_Tau"]     = [20, 30]
    Radius["GG_Tau"]     = [200, 240]
    Radius["GM_Aur"]     = [105, 130]
    Radius["HD_31648"]   = [25, 40]
    Radius["HD_97048"]   = [155, 195]
    Radius["Hen_3-545"]  = [25, 60]
    Radius["IP_Tau"]     = [20, 30]
    Radius["LkCa_15"]    = [60, 80]
    Radius["SY_Cha"]     = [60, 90]
    Radius["SZCha"]      = [50, 70]
    Radius["UX_Tau"]     = [34, 42]
    Radius["UX_TAU"]     = [34, 42]
    Radius["V351_Ori"]   = [110, 140]
    Radius["V599_Ori"]   = [150, 200]

    Radius["J16090075"]  = [13, 28]
    Radius["J16120668"]  = [70, 90]
    Radius["BD-08_1115"] = [60, 100]
    Radius["DL_Tau"]     = [50, 60]
    Radius["DoAr_25"]    = [60, 150]
    Radius["Haro_1-1"]   = [45, 65]
    Radius["HD_34700"]   = [60, 100]
    Radius["HD_100453"]  = [50, 90]
    Radius["HD_142527"]  = [100, 120]
    Radius["TYC_6213"]   = [30, 50]

    for key in Radius.keys():
        if key in Path:
            R_in, R_out = Radius[key]   
            break
        else:
            R_in, R_out = 50, 100
    return R_in, R_out

def Correct_Corono_Transmission(img, band, pixelscale, output='img'):
    c = len(img)/2
    Radius  = np.array([0.006125,   0.01225,    0.018375,   0.0245,     0.030625,   0.03675,    0.042875,   0.049,      0.055125,   0.06125,    0.067375,   0.0735,     0.079625,   0.091875,   0.104125,  0.116375,  0.128625,  0.153125,  0.177625,   0.2,       0.3,       0.5,        1.])
    if band == 1.25 :   # J-Band
        Trans = np.array([0.00780758, 0.00780758, 0.0117624,  0.01876871, 0.01876871, 0.02704523, 0.03394285, 0.03394285, 0.04112804, 0.05961476, 0.05961476, 0.10860413, 0.20551687, 0.35192513, 0.5265369, 0.8197032, 0.8940119, 0.9494303, 0.9875806,  0.9945728, 0.997758,  0.9984513,  1.])
    elif band == 1.65 : # H-Band
        Trans = np.array([0.00841281, 0.00841281, 0.01311053, 0.02022418, 0.02022418, 0.02746119, 0.03306766, 0.03306766, 0.04040532, 0.06114158, 0.06114158, 0.11278401, 0.20972891, 0.3521183,  0.5206344, 0.8097272, 0.8878255, 0.9476569, 0.98447585, 0.9940046, 0.9978705, 0.99844426, 1.])
    else :              # K-Band
        Trans = np.array([0.0058310116, 0.0058310116, 0.008097187, 0.008097187, 0.011497507, 0.011497507, 0.014604552, 0.014604552, 0.01684789, 0.01684789, 0.02330623, 0.02330623, 0.04786689, 0.10906341, 0.2190966, 0.37338492, 0.5486097, 0.83560383, 0.9430666, 0.9601188, 0.99295855, 0.99876374, 1.0])

    y, x = np.indices((len(img), len(img)))
    r    = np.sqrt((x - c)**2 + (y - c)**2)
    transmission_interp = interp1d(Radius/pixelscale, Trans, bounds_error=False, fill_value=1)
    mask = transmission_interp(r)
    if output == 'img':
        return img/mask
    else :
        return mask

def pix2au(value, distance, pix2arcsec):
    return value * 648000/np.pi * distance * np.tan(np.radians(pix2arcsec/3600)) 