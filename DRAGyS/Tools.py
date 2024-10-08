import sys
import os
sys.path.append(os.getcwd())
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
import math
from scipy.interpolate import splrep, splev, griddata
from scipy import ndimage, signal
from scipy.stats import beta
import random
import pickle
import cv2
import pathlib


def Init_Folders():
    current_folder = pathlib.Path(__file__).parent
    with open(f"{current_folder}/DRAGyS_path.txt", 'r', encoding='utf-8') as file:  # Ouvre le fichier en mode lecture
        content = file.read()
    Folders = content.split('___separation___')
    GUI_Folder     = Folders[0]
    Fitting_Folder = Folders[1]
    SPF_Folder     = Folders[2]
    return GUI_Folder, Fitting_Folder, SPF_Folder
    
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

def Max_pixel(image, xmin, xmax, Noise, R_max=None, gaussian_filter=3, smooth_filter=10, prominence=0.1, distance=1, width=1, threshold=None, HighPass=1, Mincut_Radius=49, Maxcut_Radius=50, method='Azimuthal'):
    X, Y  = [], []
    min_X = []
    max_X = []
    min_Y = []
    max_Y = []
    image = ndimage.gaussian_filter(image , sigma = gaussian_filter)
    blur_size = HighPass
    blurred = cv2.GaussianBlur(image, (blur_size, blur_size), 0)
    image = cv2.subtract(image, blurred)

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
                range_pixel = MaxPixel_range(np.arange(len(flux_smooth_up)), np.exp(flux_smooth_up), peak, 3 * Noise, max=10)
                x = dec + peak + w 
                y = s   - peak + w 
                if Mincut_Radius <= np.sqrt((x-s/2)**2 + (y-s/2)**2) <= Maxcut_Radius:
                    X.append(x)
                    Y.append(y)
                    min_X.append(dec + np.min(range_pixel) + w)
                    max_X.append(dec + np.max(range_pixel) + w)
                    min_Y.append(s   - np.min(range_pixel) + w)
                    max_Y.append(s   - np.max(range_pixel) + w)

            for peak in (peaks_find_down):
                range_pixel = MaxPixel_range(np.arange(len(flux_smooth_down)), np.exp(flux_smooth_down), peak, 3 * Noise, max=10)
                x = peak + w
                y = s - dec - peak - w
                if Mincut_Radius <= np.sqrt((x-s/2)**2 + (y-s/2)**2) <= Maxcut_Radius:
                    X.append(x)
                    Y.append(y)
                    min_X.append(np.min(range_pixel) + w)
                    max_X.append(np.max(range_pixel) + w)
                    min_Y.append(s - dec - np.min(range_pixel) - w)
                    max_Y.append(s - dec - np.max(range_pixel) - w)
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
                range_pixel = MaxPixel_range(np.arange(len(flux_smooth_up)), np.exp(flux_smooth_up), peak, 3 * Noise, max=10)
                x = peak
                y = dec + peak
                if Mincut_Radius <= np.sqrt((x-s/2)**2 + (y-s/2)**2) <= Maxcut_Radius:
                    X.append(x)
                    Y.append(y)
                    min_X.append(np.min(range_pixel))
                    max_X.append(np.max(range_pixel))
                    min_Y.append(dec + np.min(range_pixel))
                    max_Y.append(dec + np.max(range_pixel))
            for peak in (peaks_find_down):
                range_pixel = MaxPixel_range(np.arange(len(flux_smooth_down)), np.exp(flux_smooth_down), peak, 3 * Noise, max=10)
                x = dec + peak
                y = peak
                if Mincut_Radius <= np.sqrt((x-s/2)**2 + (y-s/2)**2) <= Maxcut_Radius:
                    X.append(x)
                    Y.append(y)
                    min_X.append(dec + np.min(range_pixel))
                    max_X.append(dec + np.min(range_pixel))
                    min_Y.append(np.min(range_pixel))
                    max_Y.append(np.max(range_pixel))

    elif method == 'Horizontal':
        s = len(image)
        for i in range(len(image)):
            flux = moving_average(np.array(image[:, i]), smooth_filter)
            peaks_find  , _ = signal.find_peaks(flux,   prominence=prominence, distance=distance, width=width, threshold=threshold) #, distance=d)
            for peak in (peaks_find):
                range_pixel = MaxPixel_range(np.arange(len(flux)), np.exp(flux), peak, 3 * Noise, max=10)
                x = peak
                y = i
                if Mincut_Radius <= np.sqrt((x-s/2)**2 + (y-s/2)**2) <= Maxcut_Radius:
                    X.append(x)
                    Y.append(y)
                    min_X.append(np.min(range_pixel))
                    max_X.append(np.max(range_pixel))
                    min_Y.append(y)
                    max_Y.append(y)

    elif method == 'Vertical':
        s = len(image)
        for i in range(len(image)):
            flux = moving_average(np.array(image[i, :]), smooth_filter)
            peaks_find  , _ = signal.find_peaks(flux,   prominence=prominence, distance=distance, width=width, threshold=threshold) #, distance=d)
            for peak in (peaks_find):
                range_pixel = MaxPixel_range(np.arange(len(flux)), np.exp(flux), peak, 3 * Noise, max=10)
                x = i
                y = peak
                if Mincut_Radius <= np.sqrt((x-s/2)**2 + (y-s/2)**2) <= Maxcut_Radius:
                    X.append(x)
                    Y.append(y)
                    min_X.append(x - 1)
                    max_X.append(x + 1)
                    min_Y.append(np.min(range_pixel))
                    max_Y.append(np.max(range_pixel))
    
    elif method == 'Azimuthal':
        Phi = np.radians(np.linspace(0, 359, 360))
        if R_max == None:
            R = np.arange(0, int(len(image)/2), 1)
        else :
            R = np.arange(Mincut_Radius, Maxcut_Radius, 1)
        for phi in Phi:
            x = list(map(int, len(image)/2 + R*np.sin(phi)))
            y = list(map(int, len(image)/2 + R*np.cos(phi)))
            Flux    = image[x, y]
            flux = np.log(Flux)
            flux_smooth = moving_average(np.array(flux), smooth_filter)
            peaks_find, _ = signal.find_peaks(flux_smooth, prominence=prominence, distance=distance, width=width, threshold=threshold) #, distance=d)
            for peak in peaks_find:
                range_pixel = MaxPixel_range(np.arange(len(flux_smooth)), np.exp(flux_smooth), peak, 3 * Noise, max=10)
                
                X.append(len(image)/2 + (Mincut_Radius + peak)*np.sin(phi))
                Y.append(len(image)/2 + (Mincut_Radius + peak)*np.cos(phi))
                min_X.append(len(image)/2 + (Mincut_Radius + (np.min(range_pixel)))*np.sin(phi))
                max_X.append(len(image)/2 + (Mincut_Radius + (np.max(range_pixel)))*np.sin(phi))
                min_Y.append(len(image)/2 + (Mincut_Radius + (np.min(range_pixel)))*np.cos(phi))
                max_Y.append(len(image)/2 + (Mincut_Radius + (np.max(range_pixel)))*np.cos(phi))
    return X, Y, min_X, max_X, min_Y, max_Y

def generate_pert_random(a, b, m, size=1):
    if a >= m or b <= m :
        return m    
    else:
        alpha = 1 + 4 * (m - a) / (b - a)
        beta_param = 1 + 4 * (b - m) / (b - a)
        res = a + (b - a) * beta.rvs(alpha, beta_param, size=size)
        return res[0]

def Triangul(a, b, m, size=1):
    values = []
    if a > b and a != m and b != m:
        b, a = a, b
    if a == m or b == m :
        return m    
    else:
        return np.random.triangular(a, m, b)

def Fitting_Ring(x0, y0, X, Y, x_min, x_max, y_min, y_max, nb, filename, Tsigma=1):
    ## Compute Values of Inclination, Position angle, and Scale Height
    coeffs = fit_ellipse(np.array(X), np.array(Y))
    Xc, Yc, a, b, e, PA_LSQE = cart_to_pol(coeffs)

    inclination   = np.arccos(b/a)
    PositionAngle = My_PositionAngle(x0, y0, Yc, Xc, a, b, e, PA_LSQE)
    X_e, Y_e      = get_ellipse_pts((Xc, Yc, a, b, e, PositionAngle))
    # Height        = Height_Compute(X_e, Y_e, x0, y0)/np.sin(inclination)
    Height        = np.sqrt((x0-Xc)**2 + (y0-Yc)**2)/np.sin(inclination)
    # m, c = 0.1708257, -0.24881799
    # Height        = Height * (m+1) + c
    Radius        = a
    Aspect        = Height/Radius

    ## Compute Errors of Inclination, Position Angle, and Scale Height
    incl, PosAngle, radius, h, h_r = [], [], [], [], []
    for step in range(nb):
        X_rand, Y_rand = [], []
        for az in range(len(X)):
            sigmaX = np.max([np.abs(X[az] - x_min[az]), np.abs(X[az] - x_max[az])])
            sigmaY = np.max([np.abs(Y[az] - y_min[az]), np.abs(Y[az] - y_max[az])])
            X_rand.append(np.random.normal(X[az], sigmaX, 1)[0])
            Y_rand.append(np.random.normal(Y[az], sigmaY, 1)[0])

        coeffs = fit_ellipse(np.array(X_rand), np.array(Y_rand))
        Xc, Yc, a, b, e, PA_LSQE = cart_to_pol(coeffs)
        PA = My_PositionAngle(x0, y0, Yc, Xc, a, b, e, PA_LSQE)
        PosAngle.append(PA)

        i = np.arccos(b/a)
        incl.append(i)

        X_e, Y_e = get_ellipse_pts((Xc, Yc, a, b, e, PA))
        h0 = Height_Compute(X_e, Y_e, x0, y0)/np.sin(i)
        # h0        = np.sqrt((x0-Xc)**2 + (y0-Yc)**2)/np.sin(i)

        h.append(h0)
        radius.append(a)
        h_r.append(h0/a)

    Error_Inclination       = np.nanstd(incl)
    Error_PositionAngle     = np.nanstd(PosAngle)
    Error_Height            = np.nanstd(h)
    Error_Radius            = np.nanstd(radius)
    Error_Aspect            = Aspect * np.sqrt((Error_Height/Height)**2 + (Error_Radius/Radius)**2)
    # Error_Aspect            = 3 * np.nanstd(h_r)

    ## Save Data estimated on a fitting file
    Data_to_Save    = {"params"     : [inclination, Radius, Height, Aspect, 1, PositionAngle], 
                       "Err"        : [Error_Inclination, Error_Radius, Error_Height, Error_Aspect, 0, Error_PositionAngle], 
                       'Points'     : [X, Y, x_min, x_max, y_min, y_max], 
                       "Ellipse"    : [Xc, Yc, X_e, Y_e, a, b, e]}
    GUI_Folder, Fitting_Folder, SPF_Folder = Init_Folders()
    with open(f"{Fitting_Folder}/{filename}", 'wb') as fichier:
        pickle.dump(Data_to_Save, fichier)

def ellipse_mask(x_center, y_center, width, height, angle, shape):
    y, x = np.ogrid[:shape[0], :shape[1]]
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)
    x_ = x - x_center
    y_ = y - y_center
    ellipse_eq = ((x_ * cos_angle + y_ * sin_angle) ** 2 / (width / 2) ** 2 +
                  (x_ * sin_angle - y_ * cos_angle) ** 2 / (height / 2) ** 2)
    mask = ellipse_eq <= 1
    return mask
    
def BeamSpace(Radius, r_beam, PA, incl, aspect, alpha):
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
    dist_from_center = np.sqrt((X - center[0])**2 + (Y - center[1])**2)
    mask = dist_from_center < radius
    return mask

def Elliptic_Mask(R_in, R_out, PA, incl, aspect, alpha, shape):
    
    Phi = np.radians(np.linspace(0, 359, 360))
    x_rot_in = R_in * np.sin(Phi) * np.cos(np.pi - PA) - (aspect * R_in**alpha * np.sin(incl) - R_in * np.cos(Phi) * np.cos(incl)) * np.sin(np.pi - PA) + shape[0]/2
    y_rot_in = R_in * np.sin(Phi) * np.sin(np.pi - PA) + (aspect * R_in**alpha * np.sin(incl) - R_in * np.cos(Phi) * np.cos(incl)) * np.cos(np.pi - PA) + shape[0]/2
    x_rot_out = (R_out * np.sin(Phi)) * np.cos(np.pi - PA) - (aspect * R_out**alpha * np.sin(incl) - R_out * np.cos(Phi) * np.cos(incl)) * np.sin(np.pi - PA) + shape[0]/2
    y_rot_out = (R_out * np.sin(Phi)) * np.sin(np.pi - PA) + (aspect * R_out**alpha * np.sin(incl) - R_out * np.cos(Phi) * np.cos(incl)) * np.cos(np.pi - PA) + shape[0]/2

    x_c_in,  y_c_in  = np.mean(x_rot_in),  np.mean(y_rot_in)
    x_c_out, y_c_out = np.mean(x_rot_out), np.mean(y_rot_out)
    a_in,  b_in  = R_in,  R_in * np.cos(incl)
    a_out, b_out = R_out, R_out * np.cos(incl)
    mask2 = ellipse_mask(y_c_in,  x_c_in,  2 * a_in,  2 * b_in,  PA - np.pi/2, shape)
    mask1 = ellipse_mask(y_c_out, x_c_out, 2 * a_out, 2 * b_out, PA - np.pi/2, shape)
    return mask1 & ~mask2

def My_PositionAngle(xcenter, ycenter, x0, y0, a, b, e, PA_LSFE):
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

    Arguments :
    - x: numpy list or array containing the x-coordinates of the points of the ellipse
    - y: numpy list or array containing the y coordinates of the ellipse's points

    Returns :
    - tuple (a, b, c) representing the coefficients of the line ax + by + c = 0
    """
    a, b, c = find_ellipse_axis_coeffs(x, y, axis='major')
    
    # Calcul du projeté orthogonal
    x_proj = (b * (b * xs - a * ys) - a * c) / (a**2 + b**2)
    y_proj = (a * (-b * xs + a * ys) - b * c) / (a**2 + b**2)

    distance = np.sqrt((xs - x_proj)**2 + (ys - y_proj)**2)
    return distance

def find_ellipse_axis_coeffs(x, y, axis='major'):
    """
    Find the coefficients of the equation of the line corresponding to the major axis of the ellipse
    defined by the points x, y.

    Arguments :
    - x: numpy list or array containing the x-coordinates of the points of the ellipse
    - y: numpy list or array containing the y coordinates of the ellipse's points
    - axis: string to choose "major" or "minor" axis

    Returns :
    - tuple (a, b, c) representing the coefficients of the line ax + by + c = 0
    """
    # Centrer les données
    x_mean, y_mean = np.mean(x), np.mean(y)
    centered_x, centered_y = x - x_mean, y - y_mean

    # Calculer la matrice de covariance
    cov_matrix = np.cov(centered_x, centered_y)

    # Calculer les vecteurs propres et les valeurs propres
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

    # Trouver le vecteur propre associé à la plus petite valeur propre
    if axis == 'major':
        axis_index = np.argmin(eigenvalues)
    else :
        axis_index = np.argmin(eigenvalues)

    axis_vector = eigenvectors[:, axis_index]

    # Calculer les coefficients de la droite
    a, b = axis_vector
    c = -a * x_mean - b * y_mean

    return a, b, c

def Load_Structure(filename, Type='Struct'):
    GUI_Folder, Fitting_Folder, SPF_Folder = Init_Folders()
    # with open(os.getcwd()+'/Results/Fitting/' + filename, 'rb') as fichier:
    with open(f'{Fitting_Folder}/{filename}', 'rb') as fichier:
        Loaded_Data = pickle.load(fichier)
    if Type == 'Struct':
        return Loaded_Data["params"], Loaded_Data["Err"]
    elif Type == 'Points':
        return Loaded_Data["Points"]
    elif Type == 'Ellipse':
        return Loaded_Data['Ellipse']

def moving_average(data, window_size):
    smoothed_data = np.convolve(data, np.ones(window_size)/window_size, mode='same')
    return smoothed_data

def MaxPixel_range(x, y, peak, Noise, max=10):
    seuil = y[peak] - Noise
    n = 0
    m = 0
    while peak - n > 0 and y[peak - n] >= seuil :
        if n >= max:
            break
        n += 1
    while peak + m < len(y) and y[peak + m] >= seuil :
        m += 1
        if m >= max:
            break
    if n==0 and m==0 :
        range = [peak]
    else :
        range = x[peak-n:peak+m]
    return range
# =======================================================================================
# =================================== Normalization =====================================
# =======================================================================================

def Same90(ref, Scatt1, phF1, Scatt2, phF2):
    phF1_90 = splev(ref, splrep(Scatt1, phF1))
    phF2_90 = splev(ref, splrep(Scatt2, phF2))

    Same_Amp_phF2 = phF2 * phF1_90/phF2_90
    return Same_Amp_phF2

def Normalize(Scatt, I, PI, Err_I, Err_PI, Type='Norm'):
    if Type=='Norm':
        Norm_I        = np.max(I)
        Norm_PI       = np.max(PI)
    elif Type=='90':
        Norm_I        = splev(90, splrep(Scatt, I))
        Norm_PI       = splev(90, splrep(Scatt, PI))
    I, PI         = I/Norm_I, PI/Norm_PI
    Err_I, Err_PI = Err_I/Norm_I, Err_PI/Norm_PI
    return I, PI, np.abs(Err_I), np.abs(Err_PI)

# =======================================================================================
# ================================== Phase Functions ====================================
# =======================================================================================
def get_mean_value_in_circle(image, center, radius):
    """Calcule la valeur moyenne à l'intérieur du cercle."""
    cx, cy = center
    y, x = np.ogrid[:image.shape[0], :image.shape[1]]
    mask = (x - cx)**2 + (y - cy)**2 <= radius**2
    # return image[mask].mean()
    return np.nanmean(image[mask])

def MCFOST_PhaseFunction(file_path, Name, Normalization):
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
        MCFOST_DoP     = MCFOST_PI / MCFOST_I
        MCFOST_I  = MCFOST_I/np.max(MCFOST_I)
        MCFOST_PI = MCFOST_PI/np.max(MCFOST_PI)

        MCFOST_Err_I, MCFOST_Err_PI = np.zeros_like(MCFOST_I), np.zeros_like(MCFOST_PI)
        MCFOST_Err_DoP = np.sqrt((MCFOST_Err_PI/MCFOST_I)**2 + (MCFOST_Err_I*MCFOST_PI/MCFOST_I**2)**2)
        if Normalization: 
            MCFOST_I, MCFOST_PI, MCFOST_Err_I, MCFOST_Err_PI = Normalize(MCFOST_Scatt, MCFOST_I, MCFOST_PI, MCFOST_Err_I, MCFOST_Err_PI, Type='90')

    return MCFOST_Scatt, MCFOST_I, MCFOST_PI, MCFOST_DoP, MCFOST_Err_I, MCFOST_Err_PI, MCFOST_Err_DoP

# def BHMIE_PhF(filepath, Grain_file, obs_wl=1.6, Norm=True):
#     from Bhmie import Fbhmie
#     print('Compute Phase Function with Bhmie code for "' +Grain_file+ '" Grain type...')
#     refrel = Interpolation_Bhmie(os.getcwd()+"/Bhmie/utils/Dust/" +Grain_file, obs_wl)
#     Scatt  = np.linspace(0, 180, 181)
#     a_gr   = (fits.getdata(filepath+'/data_disk/grain_sizes.fits.gz'))#[:50]
#     p      = 3.5
#     if 'HD163296' in filepath:
#         a_gr = a_gr
#     else : 
#         a_gr = a_gr[:50]
#     int_S11, int_S12 = 0, 0
#     for a in a_gr:
#         x = 2*np.pi/obs_wl*a
#         S1, S2, _, _, _, _ = Fbhmie.bhmie(x, refrel, 91)
#         int_S11 += ( 0.5 * ( np.abs(S2)**2 + np.abs(S1)**2 ))*a**(1-p)
#         int_S12 += ( 0.5 * ( np.abs(S2)**2 - np.abs(S1)**2 ))*a**(1-p)  
#     denom = (np.max(a_gr)**(1-p) - np.min(a_gr)**(1-p))/(1-p)
#     I  = np.array(int_S11/denom)
#     PI = np.array(int_S12/denom)
#     Err_I    = np.zeros(len(I))
#     Err_PI   = np.zeros(len(PI))
#     if Norm:
#         I, PI, Err_I, Err_PI = Normalize(Scatt, I, PI, Err_I, Err_PI, Type='90')
#     return Scatt, I, PI, Err_I, Err_PI
# 
# def Plot_RefractiveIndex(Refrels, Names):
#     Dust_Folder = "C://Users/mroum/OneDrive/Bureau/PhD/Bhmie/utils/Dust"
#     obs_wl      = 1.6
#     for dust in os.listdir(Dust_Folder):
#         Refract = Interpolation_Bhmie(Dust_Folder + '/' + dust, obs_wl, Plot_Interpolation=False, Log_distrib=False)
#         Re_ref = np.real(Refract)
#         Im_ref = np.imag(Refract)
#         if dust in ['Drain_Si_sUV.dat', 'nk2_amC_Zubko_BE.dat', 'Graphite_para.dat', 'Graphite_perp.dat', 'ice_opct.dat']:
#             name = ['Silicate', 'AmCarbon', 'Graph_para', 'Graphite_perp', 'ice']
#             plt.scatter(Re_ref, np.log(Im_ref), c='b')
#             plt.annotate(name[np.where(dust == os.listdir(Dust_Folder))], (Re_ref, np.log(Im_ref)))          
#         plt.scatter(Re_ref, np.log(Im_ref), c='k')
#     # plt.yscale("log")
#     for idx, refrel in enumerate(Refrels):
#         plt.scatter(np.real(refrel), np.log(np.imag(refrel)), c='r', marker='s', label=Names[idx])
#     plt.xlabel('n')
#     plt.ylabel('log(k)')
#     plt.show()
# 
# def Interpolation_Bhmie(dust_path, obs_wl, Plot_Interpolation=False, Log_distrib=False):
#     file = open(dust_path, 'r')
#     lignes = file.readlines()
#     wl, Re_idx, Im_idx = [], [], []
#     for lign in lignes:
#         if lign[0][0] != '#' and len(lign.split()) == 3 :
#             values = lign.split()
#             wl.append(float(values[0]))
#             Re_idx.append(float(values[1]))
#             Im_idx.append(float(values[2]))
#     if wl[0] > wl[1]:
#         Re_idx = Re_idx[::-1]
#         Im_idx = Im_idx[::-1]
#         wl     = wl[::-1]
#     Re_ref = splev(obs_wl, splrep(wl, Re_idx))
#     Im_ref = splev(obs_wl, splrep(wl, Im_idx))
#     if Log_distrib:
#         Re_min = np.interp(obs_wl-0.1, wl, Re_idx)
#         Re_max = np.interp(obs_wl+0.1, wl, Re_idx)
#         Im_min = np.interp(obs_wl-0.1, wl, Im_idx)
#         Im_max = np.interp(obs_wl+0.1, wl, Im_idx)
#         wl_l = np.linspace(obs_wl-1, obs_wl+1, 50)
#         Re = 10**np.linspace(np.log10(Re_min), np.log10(Re_max), 50)
#         Im = 10**np.linspace(np.log10(Im_min), np.log10(Im_max), 50)
#         Re_ref = np.interp(obs_wl, wl_l, Re)
#         Im_ref = np.interp(obs_wl, wl_l, Im)
#     if Plot_Interpolation:
#         plt.figure("Interpolation of Bhmie Refraction Index")
#         plt.suptitle("Real and Imaginary part of refraction index", fontsize=18, fontweight='bold')
#         plt.title('interpolate values : Re_{index} = ' + str(Re_ref) + ' ; Im_{index} = ' + str(Im_ref), fontsize=15)
#         plt.semilogx(wl, Re_idx, color='green', label='real part')
#         plt.semilogx(wl, Im_idx, color='blue', label='imaginary part')
#         plt.scatter([obs_wl, obs_wl], [Re_ref, Im_ref], color='red', marker='x', label='Iterpolation values')
#         plt.xlabel('wavelengths [$\mu m$]', fontsize=15)
#         plt.legend()
#     return Re_ref + Im_ref*1j

def NewScatt(R, ellipse_center, star=(0, 0)):
    Phi    = np.radians(np.linspace(0, 359, 360))
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

def Orthogonal_Prejection(star_position, ellipse_center, PA):
    (xs, ys) = star_position
    (xc, yc) = ellipse_center
    a = np.cos(-PA)
    b = np.sin(-PA)
    t = ((xc - xs) * a + (yc - ys) * b) / (a * a + b * b)
    xc_prime = xs + t * a
    yc_prime = ys + t * b
    return xc_prime, yc_prime

def Non_Centered_Star_AzimithalAngle(R, star_position, ellipse_center, PA):
    (xs, ys) = star_position
    (xc, yc) = ellipse_center
    a = np.cos(-PA)
    b = np.sin(-PA)
    # t = ((xc - xs) * a + (yc - ys) * b) / (a * a + b * b)
    # xc_prime = xs + t * a
    # yc_prime = ys + t * b
    xc_prime, yc_prime = Orthogonal_Prejection(star_position, ellipse_center, PA)
    X_PA = a + xs
    Y_PA = b + ys

    X_SC = xc - xs
    Y_SC = yc - ys
    X_SPA = X_PA - xs
    Y_SPA = Y_PA - ys
    angle = angle_oriente_2d((X_SPA, Y_SPA), (X_SC, Y_SC))
    # np.degrees(np.arctan2(Y_SC, X_SC))
    if 0 <= angle <= np.pi/2:
        D =   np.sqrt((xs - xc_prime)**2 + (ys - yc_prime)**2)
    elif np.pi/2 < angle :
        D = - np.sqrt((xs - xc_prime)**2 + (ys - yc_prime)**2)
    elif -np.pi/2 <= angle <= 0:
        D = - np.sqrt((xs - xc_prime)**2 + (ys - yc_prime)**2)
    elif -np.pi/2 > angle:
        D =   np.sqrt((xs - xc_prime)**2 + (ys - yc_prime)**2)
    # D = - np.sqrt((xs - xc_prime)**2 + (ys - yc_prime)**2)
    New_Phi = NewScatt(R, (xs + D, ys), star=(xs, ys))
    return New_Phi, xc_prime, yc_prime

def Get_PhF(filename, side='All'):
    with open(filename, 'rb') as fichier:
        Loaded_Data = pickle.load(fichier)
    Scatt     = np.array(Loaded_Data[side]["Scatt"])
    I         = np.array(Loaded_Data[side]["I"])
    PI        = np.array(Loaded_Data[side]["PI"])
    Err_Scatt = np.array(Loaded_Data[side]["Err_Scatt"])
    Err_I     = np.array(Loaded_Data[side]["Err_I"])
    Err_PI    = np.array(Loaded_Data[side]["Err_PI"])
    LB        = np.array(Loaded_Data[side]["LB"])
    Err_LB    = np.array(Loaded_Data[side]["Err_LB"])
    return Scatt, I, PI, Err_Scatt, Err_I, Err_PI, LB, Err_LB

def Remove_LB(Scatt, Flux, Err_Flux, incl, aspect, chi):
    LB_effect = (np.cos(aspect) * np.sin(aspect*chi-aspect)) / (np.cos(aspect)*np.sin(aspect*chi-aspect) + np.cos(incl)*np.cos(aspect*chi-aspect) - np.sin(aspect*chi)*np.cos(np.radians(Scatt)))
    flux = Flux/LB_effect
    Err_flux = Err_Flux/LB_effect
    return flux, Err_flux

# =======================================================================================
# ==================================     Images      ====================================
# =======================================================================================

def Images_Opener(filepath):
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

    elif nb_dim == 5:  # if it is an MCFOST Simulated Image, dimension is [8, 1, 1, dim, dim]
        # threshold = 0.00000
        Image_0 = img[0, 0, 0]
        Image_1 = img[1, 0, 0]
        Image_2 = img[2, 0, 0]
        Image_3 = img[3, 0, 0]
        Image_4 = np.sqrt(Image_1**2 + Image_2**2)
        Image_5 = np.zeros_like(Image_0) + 1
        for idx, image in enumerate([Image_0, Image_1, Image_2, Image_3, Image_4, Image_5]):
            IMG[idx] = image
            mask = np.where(image > 0.)
            if len(image[mask]) == 0:
                Thresh[idx] = 1e-40
            else :
                Thresh[idx] = np.nanmin(image[mask])
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
    return IMG, Thresh

def DiskDistances():
    D = [160.3, 144.0, 145.5, 158.4, 156.3, 184.4, 184.4, 185.2, 129.4, 157.2, 157.2, 180.7, 180.7, 190.2, 142.2, 142.2, 323.82, 401.07]
    Distances = {}
    Folder = 'C:\\Users\mroum\OneDrive\Bureau\PhD\Data\DESTINYS/'
    List_File = os.listdir(Folder)
    List_File.remove('additional_suggestions')
    for idx, file in enumerate(List_File):
        hdu = fits.open(Folder + file)
        header = hdu[0].header
        name = header['TARGET']
        Distances[name] = D[idx]

    D = [137.4, 131.9, 158.39, 159.9, 138.16, 151.1, 350.5, 103.8, 159.26, 154.6]
    Folder = 'C:\\Users\mroum\OneDrive\Bureau\PhD\Data\DESTINYS/additional_suggestions/'
    List_File = os.listdir(Folder)
    for idx, file in enumerate(List_File):
        hdu = fits.open(Folder + file)
        header = hdu[0].header
        name = header['TARGET']
        Distances[name] = D[idx]

    Distances["HD 34282"] = 308.6
    Distances["LkCa15"]   = 157.2
    Distances["V 4046"]   = 71.5
    Distances["PDS 66"]   = 97.9
    Distances["RX J1852"] = 147.1
    Distances["RX J1615"] = 155.6
    Distances["HD163296"] = 101.0
    Distances["MCFOST"]   = 100.0
    Distances["hr4796"]   = 71.9
    return Distances

def Get_Distance(Distances, Path):
    Folder = '/'.join(Path.split('/')[:-1]) + '/'
    File   = Path.split('/')[-1]
    FILE = File.replace(' ', '').replace('_', '').replace('-', '').upper()
    hdu = fits.open(Folder + File)
    header = hdu[0].header
    if 'TARGET' in header: 
        # print(header['TARGET'], Distances[header['TARGET']])
        return Distances[header['TARGET']]
    else :
        for keys in Distances.keys():
            KEYS = keys.replace(' ', '').replace('_', '').replace('-', '').upper()
            if KEYS in FILE :
                # print(keys, '-', Distances[keys])
                return Distances[keys]

