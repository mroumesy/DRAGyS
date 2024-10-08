import numpy as np
import sys
sys.path.append("C:\\Users\mroum\OneDrive\Bureau\PhD")
import Tools
import numpy as np
from multiprocessing import Pool
import pickle as pkl


def New_pSPF(params):
    [img_I, img_PI, distance, pixelscale, r_beam, (xs, ys), (Xc, Yc), incl, PA, aspect, alpha, D_incl, D_aspect, R_in, R_out, n_bin, Type] = params
    pixelscale_au = 648000/np.pi * distance * np.tan(np.radians(pixelscale/3600))

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
    # if Xc == None:
    #     Phi = np.radians(np.linspace(0, 359, 360))
    # else :
    #     # xs, ys = 203, 267
    #     # Phi = Tools.NewAzimuthalAngle(size, (xc, yc), star=(xs-size, ys-size))
    #     plt.figure(num="test")
    #     plt.plot(Phi)
    #     plt.plot(np.linspace(0, 2*np.pi, 361))
    #     plt.show()
    for R in np.linspace(R_in/pixelscale_au, R_out/pixelscale_au, int(R_out - R_in)):
        if Xc == None:
            Phi = np.radians(np.linspace(0, 359, 360))
            xc_p, yc_p = size, size
        else:
            Phi, yc_p, xc_p = Tools.Non_Centered_Star_AzimithalAngle(R, (xs, ys), (Yc, Xc), np.pi/2 -PA)
            # plt.plot(Phi)
            # plt.show()
        # Phi = np.radians(np.arange(0, 360, 180/np.pi * r_beam/R/2))
        # for phi in np.radians(np.linspace(0, 359, 360)) :
        for phi in Phi :
            x     = R * np.sin(phi)
            y     = aspect * R**alpha * np.sin(incl) - R * np.cos(phi) * np.cos(incl)
            x_rot = x * np.cos(np.pi - PA) - y * np.sin(np.pi - PA) + xc_p
            y_rot = x * np.sin(np.pi - PA) + y * np.cos(np.pi - PA) + yc_p
            sca = np.arccos(np.cos(aspect) * np.cos(phi) * np.sin(incl) + np.sin(aspect) *np.cos(incl))
            # sca = Tools.NewScatt(size, phi, PA, incl, aspect, alpha, (Xc, Yc), star=(0 + size, 0 + size))
            if Side[int(x_rot), int(y_rot)] == 1:
                Angle_east.append(np.degrees(sca))
                TotI_east.append(img_I[int(x_rot), int(y_rot)])
                PolarI_east.append(img_PI[int(x_rot), int(y_rot)])
            else :
                Angle_west.append(np.degrees(sca))
                TotI_west.append(img_I[int(x_rot), int(y_rot)])
                PolarI_west.append(img_PI[int(x_rot), int(y_rot)])
            # flux = Tools.get_mean_value_in_circle(img_PI, (y_rot, x_rot), r_beam)
            TotI.append(img_I[int(x_rot), int(y_rot)])
            # PolarI.append(flux)
            PolarI.append(img_PI[int(x_rot), int(y_rot)])
            Angle.append(np.degrees(sca))
    # print(Angle)
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
    dLB_g_east = (D_east * dN_g_east * N_east * dD_g_east)/D_east**2
    dLB_i_east = (-N_east*np.sin(incl)*np.cos(chi*aspect-aspect))/D_east**2
    dLB_s_east = (-N_east*np.sin(chi*aspect)*np.sin(np.radians(Scatt_east)))/D_east**2

    N_west = np.cos(aspect) * np.sin(chi*aspect - aspect)
    D_west = np.cos(aspect) * np.sin(chi*aspect - aspect) + np.cos(incl) * np.cos(chi*aspect - aspect) - np.sin(chi*aspect) * np.cos(np.radians(Scatt_west))
    dN_g_west  = - np.sin(aspect) * np.sin(chi*aspect - aspect) + np.cos(aspect) * (np.cos(chi*aspect - aspect) * (chi - 1))
    dD_g_west  = - np.sin(aspect) * np.sin(chi*aspect - aspect) + np.cos(aspect) * (np.cos(chi*aspect - aspect) * (chi - 1)) + np.cos(incl) * (-np.sin(chi*aspect - aspect) * (chi-1)) - np.cos(chi*aspect) * chi * np.cos(np.radians(Scatt_west))
    dLB_g_west = (D_west * dN_g_west * N_west * dD_g_west)/D_west**2
    dLB_i_west = (-N_west*np.sin(incl)*np.cos(chi*aspect-aspect))/D_west**2
    dLB_s_west = (-N_west*np.sin(chi*aspect)*np.sin(np.radians(Scatt_west)))/D_west**2

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


def New_pSPF_OneData(params):
    [img, distance, pixelscale, r_beam, (xs, ys), (Xc, Yc), incl, PA, R_ref, H_ref, aspect, alpha, D_incl, D_R_ref, D_H_ref, D_aspect, R_in, R_out, n_bin, Type] = params
    pixelscale_au = 648000/np.pi * distance * np.tan(np.radians(pixelscale/3600))

    R_ref = R_ref/pixelscale_au
    H_ref = H_ref/pixelscale_au
    aspect = H_ref/R_ref

    Angle,      Flux      = [], []
    Angle_east, Flux_east = [], []
    Angle_west, Flux_west = [], []
    already_used_pixels = []
    size = len(img)/2
    Side = np.zeros_like(img)
    x0 = y0 = len(img)/2
    slope = np.tan(PA)
    intercept = y0 - x0*slope
    mask_idx = np.indices((len(img), len(img)))
    mask = mask_idx[0] > slope * mask_idx[1] + intercept
    Side[mask] = 1

    for R in np.linspace(R_in/pixelscale_au, R_out/pixelscale_au, int((R_out - R_in)/pixelscale_au) + 5):
        if Xc == None:
            Phi = np.radians(np.linspace(0, 359, 360))
            xc_p, yc_p = size, size
        else:
            Phi, yc_p, xc_p = Tools.Non_Centered_Star_AzimithalAngle(R, (xs, ys), (Yc, Xc), np.pi/2 -PA)
        for phi in Phi :
            x     = R * np.sin(phi)
            y     = H_ref * (R/R_ref)**alpha * np.sin(incl) - R * np.cos(phi) * np.cos(incl)
            x_rot = x * np.cos(np.pi - PA) - y * np.sin(np.pi - PA) + xc_p
            y_rot = x * np.sin(np.pi - PA) + y * np.cos(np.pi - PA) + yc_p
            if (int(x_rot), int(y_rot)) not in already_used_pixels:
                sca = np.arccos(np.cos(aspect) * np.cos(phi) * np.sin(incl) + np.sin(aspect) *np.cos(incl))
                if Side[int(x_rot), int(y_rot)] == 1:
                    Angle_east.append(np.degrees(sca))
                    Flux_east.append(img[int(x_rot), int(y_rot)])
                else :
                    Angle_west.append(np.degrees(sca))
                    Flux_west.append(img[int(x_rot), int(y_rot)])
                Flux.append(img[int(x_rot), int(y_rot)])
                Angle.append(np.degrees(sca))
                already_used_pixels.append((int(x_rot), int(y_rot)))
    # print(Angle)
    Angle,      Flux      = np.array(Angle),      np.array(Flux)
    Angle_east, Flux_east = np.array(Angle_east), np.array(Flux_east)
    Angle_west, Flux_west = np.array(Angle_west), np.array(Flux_west)

    Scatt, D_Scatt   = [], []
    SPF, D_SPF       = [], []

    Scatt_east, D_Scatt_east   = [], []
    SPF_east, D_SPF_east       = [], []

    Scatt_west, D_Scatt_west   = [], []
    SPF_west, D_SPF_west       = [], []

    bin_Scatt = np.linspace(0, 180, n_bin+1)

    for idx in range(n_bin):
        mask      = np.where(np.logical_and(Angle      >= bin_Scatt[idx], Angle      < bin_Scatt[idx+1]))
        mask_east = np.where(np.logical_and(Angle_east >= bin_Scatt[idx], Angle_east < bin_Scatt[idx+1]))
        mask_west = np.where(np.logical_and(Angle_west >= bin_Scatt[idx], Angle_west < bin_Scatt[idx+1]))
        if len(mask[0])!=0:
            Scatt.append(np.nanmean(Angle[mask]))
            D_Scatt.append(np.nanstd(Angle[mask])/len(Angle[mask]))
            SPF.append(np.nanmean(Flux[mask]))
            D_SPF.append(np.nanstd(Flux[mask])/len(Flux[mask]))
        if len(mask_east[0]) != 0:
            Scatt_east.append(np.nanmean(Angle_east[mask_east]))
            D_Scatt_east.append(np.nanstd(Angle_east[mask_east])/len(Angle_east[mask_east]))
            SPF_east.append(np.nanmean(Flux_east[mask_east]))
            D_SPF_east.append(np.nanstd(Flux_east[mask_east])/len(Flux_east[mask_east]))
        if len(mask_west[0]) != 0:
            Scatt_west.append(np.nanmean(Angle_west[mask_west]))
            D_Scatt_west.append(np.nanstd(Angle_west[mask_west])/len(Angle_west[mask_west]))
            SPF_west.append(np.nanmean(Flux_west[mask_west]))
            D_SPF_west.append(np.nanstd(Flux_west[mask_west])/len(Flux_west[mask_west]))
    if alpha == 1:
        chi = 1.00001
        print('$\chi$ is modified to 1.00001')
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
    dLB_g_east = (D_east * dN_g_east * N_east * dD_g_east)/D_east**2
    dLB_i_east = (-N_east*np.sin(incl)*np.cos(chi*aspect-aspect))/D_east**2
    dLB_s_east = (-N_east*np.sin(chi*aspect)*np.sin(np.radians(Scatt_east)))/D_east**2

    N_west = np.cos(aspect) * np.sin(chi*aspect - aspect)
    D_west = np.cos(aspect) * np.sin(chi*aspect - aspect) + np.cos(incl) * np.cos(chi*aspect - aspect) - np.sin(chi*aspect) * np.cos(np.radians(Scatt_west))
    dN_g_west  = - np.sin(aspect) * np.sin(chi*aspect - aspect) + np.cos(aspect) * (np.cos(chi*aspect - aspect) * (chi - 1))
    dD_g_west  = - np.sin(aspect) * np.sin(chi*aspect - aspect) + np.cos(aspect) * (np.cos(chi*aspect - aspect) * (chi - 1)) + np.cos(incl) * (-np.sin(chi*aspect - aspect) * (chi-1)) - np.cos(chi*aspect) * chi * np.cos(np.radians(Scatt_west))
    dLB_g_west = (D_west * dN_g_west * N_west * dD_g_west)/D_west**2
    dLB_i_west = (-N_west*np.sin(incl)*np.cos(chi*aspect-aspect))/D_west**2
    dLB_s_west = (-N_west*np.sin(chi*aspect)*np.sin(np.radians(Scatt_west)))/D_west**2

    LB_effect_tot = N/D
    D_LB_tot = np.sqrt((dLB_g * D_aspect)**2 + (dLB_i * D_incl)**2 + (dLB_s * D_Scatt)**2)

    LB_effect_east = N_east/D_east
    D_LB_east = np.sqrt((dLB_g_east * D_aspect)**2 + (dLB_i_east * D_incl)**2 + (dLB_s_east * D_Scatt_east)**2)

    LB_effect_west = N_west/D_west
    D_LB_west = np.sqrt((dLB_g_west * D_aspect)**2 + (dLB_i_west * D_incl)**2 + (dLB_s_west * D_Scatt_west)**2)

    Scatt, Scatt_east, Scatt_west = np.array(Scatt),  np.array(Scatt_east), np.array(Scatt_west)
    SPF,         D_SPF         = np.array(SPF),         np.abs(np.array(D_SPF))
    SPF_east,    D_SPF_east    = np.array(SPF_east),    np.abs(np.array(D_SPF_east))
    SPF_west,    D_SPF_west    = np.array(SPF_west),    np.abs(np.array(D_SPF_west))
    All_disk          = {"Scatt" : Scatt,      "I" : SPF,              "PI" : SPF,        "Err_Scatt" : D_Scatt,      "Err_I" : D_SPF,              "Err_PI" : D_SPF,      "LB" : LB_effect_tot,  "Err_LB" : D_LB_tot}
    East_side         = {"Scatt" : Scatt_east, "I" : SPF_east,         "PI" : SPF_east,   "Err_Scatt" : D_Scatt_east, "Err_I" : D_SPF_east,         "Err_PI" : D_SPF_east, "LB" : LB_effect_east, "Err_LB" : D_LB_east}
    West_side         = {"Scatt" : Scatt_west, "I" : SPF_west,         "PI" : SPF_west,   "Err_Scatt" : D_Scatt_west, "Err_I" : D_SPF_west,         "Err_PI" : D_SPF_west, "LB" : LB_effect_west, "Err_LB" : D_LB_west}
    Dataset = {'All' : All_disk, "East" : East_side, "West" : West_side}
    return Dataset

def New_SPF_AllData(params):
    [IMG, distance, pixelscale, r_beam, (xs, ys), (Xc, Yc), incl, PA, aspect, alpha, D_incl, D_aspect, R_in, R_out, n_bin, Type] = params
    
    Angle, Angle_e, Angle_w = [], [], []
    Flux_0,   Flux_1,   Flux_2,   Flux_3,   Flux_4,   Flux_5   = [], [], [], [], [], []
    Flux_0_e, Flux_1_e, Flux_2_e, Flux_3_e, Flux_4_e, Flux_5_e = [], [], [], [], [], []
    Flux_0_w, Flux_1_w, Flux_2_w, Flux_3_w, Flux_4_w, Flux_5_w = [], [], [], [], [], []

    Sca, Sca_e, Sca_w = [], [], []
    SPF_0,   SPF_1,   SPF_2,   SPF_3,   SPF_4,   SPF_5   = [], [], [], [], [], []
    SPF_0_e, SPF_1_e, SPF_2_e, SPF_3_e, SPF_4_e, SPF_5_e = [], [], [], [], [], []
    SPF_0_w, SPF_1_w, SPF_2_w, SPF_3_w, SPF_4_w, SPF_5_w = [], [], [], [], [], []

    D_Sca, D_Sca_e, D_Sca_w = [], [], []
    D_SPF_0,   D_SPF_1,   D_SPF_2,   D_SPF_3,   D_SPF_4,   D_SPF_5   = [], [], [], [], [], []
    D_SPF_0_e, D_SPF_1_e, D_SPF_2_e, D_SPF_3_e, D_SPF_4_e, D_SPF_5_e = [], [], [], [], [], []
    D_SPF_0_w, D_SPF_1_w, D_SPF_2_w, D_SPF_3_w, D_SPF_4_w, D_SPF_5_w = [], [], [], [], [], []

    FLUX   = [Flux_0,   Flux_1,   Flux_2,   Flux_3,   Flux_4,   Flux_5]
    FLUX_e = [Flux_0_e, Flux_1_e, Flux_2_e, Flux_3_e, Flux_4_e, Flux_5_e]
    FLUX_w = [Flux_0_w, Flux_1_w, Flux_2_w, Flux_3_w, Flux_4_w, Flux_5_w]

    SPF   = [SPF_0,   SPF_1,   SPF_2,   SPF_3,   SPF_4,   SPF_5]
    SPF_e = [SPF_0_e, SPF_1_e, SPF_2_e, SPF_3_e, SPF_4_e, SPF_5_e]
    SPF_w = [SPF_0_w, SPF_1_w, SPF_2_w, SPF_3_w, SPF_4_w, SPF_5_w]

    D_SPF   = [D_SPF_0,   D_SPF_1,   D_SPF_2,   D_SPF_3,   D_SPF_4,   D_SPF_5]
    D_SPF_e = [D_SPF_0_e, D_SPF_1_e, D_SPF_2_e, D_SPF_3_e, D_SPF_4_e, D_SPF_5_e]
    D_SPF_w = [D_SPF_0_w, D_SPF_1_w, D_SPF_2_w, D_SPF_3_w, D_SPF_4_w, D_SPF_5_w]

    pixelscale_au = 648000/np.pi * distance * np.tan(np.radians(pixelscale/3600))

    size = len(IMG[0])/2
    Side = np.zeros_like(IMG[0])
    x0 = y0 = len(IMG[0])/2
    slope = np.tan(PA)
    intercept = y0 - x0*slope
    mask_idx = np.indices((len(IMG[0]), len(IMG[0])))
    mask = mask_idx[0] > slope * mask_idx[1] + intercept
    Side[mask] = 1
    for R in np.linspace(R_in/pixelscale_au, R_out/pixelscale_au, int(R_out - R_in)):
        if Xc == None:
            Phi = np.radians(np.linspace(0, 359, 360))
            xc_p, yc_p = size, size
        else:
            Phi, yc_p, xc_p = Tools.Non_Centered_Star_AzimithalAngle(R, (xs, ys), (Yc, Xc), np.pi/2 -PA)
        for phi in Phi :
            x     = R * np.sin(phi)
            y     = aspect * R**alpha * np.sin(incl) - R * np.cos(phi) * np.cos(incl)
            x_rot = x * np.cos(np.pi - PA) - y * np.sin(np.pi - PA) + xc_p
            y_rot = x * np.sin(np.pi - PA) + y * np.cos(np.pi - PA) + yc_p
            scatt = np.arccos(np.cos(aspect) * np.cos(phi) * np.sin(incl) + np.sin(aspect) *np.cos(incl))

            if Side[int(x_rot), int(y_rot)] == 1:
                Angle_e.append(np.degrees(scatt))
                for idx in range(len(IMG)):
                    FLUX_e[idx].append(IMG[idx][[int(x_rot), int(y_rot)]])
            else :
                Angle_w.append(np.degrees(scatt))
                for idx in range(len(IMG)):
                    FLUX_w[idx].append(IMG[idx][[int(x_rot), int(y_rot)]])
            Angle.append(np.degrees(scatt))
            for idx in range(len(IMG)):
                FLUX[idx].append(IMG[idx][[int(x_rot), int(y_rot)]])
    
    # print(Angle)

    for nb in range(len(FLUX)) :
        FLUX[nb]   = np.array(FLUX[nb])
        FLUX_e[nb] = np.array(FLUX_e[nb])
        FLUX_w[nb] = np.array(FLUX_w[nb])
    
    Angle, Angle_e, Angle_w = np.array(Angle), np.array(Angle_e), np.array(Angle_w)

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
        mask_east = np.where(np.logical_and(Angle_e    >= bin_Scatt[idx], Angle_e    < bin_Scatt[idx+1]))
        mask_west = np.where(np.logical_and(Angle_w    >= bin_Scatt[idx], Angle_w    < bin_Scatt[idx+1]))
        if len(mask[0])!=0:
            Sca.append(np.nanmean(Angle[mask]))
            D_Sca.append(np.nanstd(Angle[mask])/len(Angle[mask]))
            for nb in range(len(SPF)):
                SPF[nb].append(np.nanmean(FLUX[nb][mask]))
                D_SPF[nb].append(np.nanstd(FLUX[nb][mask])/len(FLUX[nb][mask]))

        if len(mask_east[0]) != 0:
            Sca_e.append(np.nanmean(Angle_e[mask_east]))
            D_Sca_e.append(np.nanstd(Angle_e[mask_east])/len(Angle_e[mask_east]))
            for nb in range(len(SPF)):
                SPF_e[nb].append(np.nanmean(FLUX_e[nb][mask_east]))
                D_SPF_e[nb].append(np.nanstd(FLUX_e[nb][mask_east])/len(FLUX_e[nb][mask_east]))
        if len(mask_west[0]) != 0:
            Sca_w.append(np.nanmean(Angle_w[mask_west]))
            D_Sca_w.append(np.nanstd(Angle_w[mask_west])/len(Angle_w[mask_west]))
            for nb in range(len(SPF)):
                SPF_w[nb].append(np.nanmean(FLUX_w[nb][mask_west]))
                D_SPF_w[nb].append(np.nanstd(FLUX_w[nb][mask_west])/len(FLUX_w[nb][mask_west]))

    if alpha == 1:
        chi = 1.00001
    else :
        chi = alpha
    N = np.cos(aspect) * np.sin(chi*aspect - aspect)
    D = np.cos(aspect) * np.sin(chi*aspect - aspect) + np.cos(incl) * np.cos(chi*aspect - aspect) - np.sin(chi*aspect) * np.cos(np.radians(Sca))
    dN_g  = - np.sin(aspect) * np.sin(chi*aspect - aspect) + np.cos(aspect) * (np.cos(chi*aspect - aspect) * (chi - 1))
    dD_g  = - np.sin(aspect) * np.sin(chi*aspect - aspect) + np.cos(aspect) * (np.cos(chi*aspect - aspect) * (chi - 1)) + np.cos(incl) * (-np.sin(chi*aspect - aspect) * (chi-1)) - np.cos(chi*aspect) * chi * np.cos(np.radians(Sca))
    dLB_g = (D * dN_g * N * dD_g)/D**2
    dLB_i = (-N*np.sin(incl)*np.cos(chi*aspect-aspect))/D**2
    dLB_s = (-N*np.sin(chi*aspect)*np.sin(np.radians(Sca)))/D**2

    N_east = np.cos(aspect) * np.sin(chi*aspect - aspect)
    D_east = np.cos(aspect) * np.sin(chi*aspect - aspect) + np.cos(incl) * np.cos(chi*aspect - aspect) - np.sin(chi*aspect) * np.cos(np.radians(Sca_e))
    dN_g_east  = - np.sin(aspect) * np.sin(chi*aspect - aspect) + np.cos(aspect) * (np.cos(chi*aspect - aspect) * (chi - 1))
    dD_g_east  = - np.sin(aspect) * np.sin(chi*aspect - aspect) + np.cos(aspect) * (np.cos(chi*aspect - aspect) * (chi - 1)) + np.cos(incl) * (-np.sin(chi*aspect - aspect) * (chi-1)) - np.cos(chi*aspect) * chi * np.cos(np.radians(Sca_e))
    dLB_g_east = (D_east * dN_g_east * N_east * dD_g_east)/D_east**2
    dLB_i_east = (-N_east*np.sin(incl)*np.cos(chi*aspect-aspect))/D_east**2
    dLB_s_east = (-N_east*np.sin(chi*aspect)*np.sin(np.radians(Sca_e)))/D_east**2

    N_west = np.cos(aspect) * np.sin(chi*aspect - aspect)
    D_west = np.cos(aspect) * np.sin(chi*aspect - aspect) + np.cos(incl) * np.cos(chi*aspect - aspect) - np.sin(chi*aspect) * np.cos(np.radians(Sca_w))
    dN_g_west  = - np.sin(aspect) * np.sin(chi*aspect - aspect) + np.cos(aspect) * (np.cos(chi*aspect - aspect) * (chi - 1))
    dD_g_west  = - np.sin(aspect) * np.sin(chi*aspect - aspect) + np.cos(aspect) * (np.cos(chi*aspect - aspect) * (chi - 1)) + np.cos(incl) * (-np.sin(chi*aspect - aspect) * (chi-1)) - np.cos(chi*aspect) * chi * np.cos(np.radians(Sca_w))
    dLB_g_west = (D_west * dN_g_west * N_west * dD_g_west)/D_west**2
    dLB_i_west = (-N_west*np.sin(incl)*np.cos(chi*aspect-aspect))/D_west**2
    dLB_s_west = (-N_west*np.sin(chi*aspect)*np.sin(np.radians(Sca_w)))/D_west**2

    LB_effect_tot = N/D
    D_LB_tot = np.sqrt((dLB_g * D_aspect)**2 + (dLB_i * D_incl)**2 + (dLB_s * D_Sca)**2)

    LB_effect_east = N_east/D_east
    D_LB_east = np.sqrt((dLB_g_east * D_aspect)**2 + (dLB_i_east * D_incl)**2 + (dLB_s_east * D_Sca_e)**2)

    LB_effect_west = N_west/D_west
    D_LB_west = np.sqrt((dLB_g_west * D_aspect)**2 + (dLB_i_west * D_incl)**2 + (dLB_s_west * D_Sca_w)**2)

    Sca, Sca_e, Sca_w = np.array(Sca),  np.array(Sca_e), np.array(Sca_w)
    for nb in range(len(SPF)):
        SPF[nb]   = np.array(SPF[nb])
        SPF_e[nb] = np.array(SPF_e[nb])
        SPF_w[nb] = np.array(SPF_w[nb])
        D_SPF[nb]   = np.array(D_SPF[nb])
        D_SPF_e[nb] = np.array(D_SPF_e[nb])
        D_SPF_w[nb] = np.array(D_SPF_w[nb])
    
    All_disk          = {"Scatt" : Sca,    "SPF" : SPF,   "Err_Scatt" : D_Sca,   "Err_SPF" : D_SPF,   "LB" : LB_effect_tot,    "Err_LB" : D_LB_tot}
    East_disk         = {"Scatt" : Sca_e,  "SPF" : SPF_e, "Err_Scatt" : D_Sca_e, "Err_SPF" : D_SPF_e, "LB" : LB_effect_east,  "Err_LB" : D_LB_east}
    West_disk         = {"Scatt" : Sca_w,  "SPF" : SPF_w, "Err_Scatt" : D_Sca_w, "Err_SPF" : D_SPF_w, "LB" : LB_effect_west,  "Err_LB" : D_LB_west}
    Dataset = {'All' : All_disk, "East" : East_disk, "West" : West_disk}
    return Dataset

def New_Beam_pSPF(params):
    [img_I, img_PI, distance, pixelscale, r_beam, incl, PA, aspect, alpha, D_incl, D_aspect, R_in, R_out, n_bin, Type] = params
    pixelscale_au = 648000/np.pi * distance * np.tan(np.radians(pixelscale/3600))

    range_R   = r_beam * 180/np.pi * 3600 /pixelscale
    R_in, R_out = R_in/pixelscale, R_out/pixelscale
    Radius = np.arange(R_in, R_out, range_R)

    Beam_X, Beam_Y, Beam_Phi = Tools.BeamSpace(Radius, range_R/2, PA, incl, aspect, alpha)
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
            mask = Tools.create_circular_mask(X, Y, (y, x), range_R/2)
            if Side[int(x), int(y)] == 1:
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

def Compute_SPF(params, File_name, img_name):
    # print("Compute SPF...")
    with Pool(processes=7) as pool_SPF:
        # resultats = pool_SPF.map(New_pSPF, params)
        resultats = pool_SPF.map(New_pSPF_OneData, params)
        # resultats = pool_SPF.map(New_Beam_pSPF, params)
    Data_tot        = resultats[0]
    Data_MinPA      = resultats[1]
    Data_MaxPA      = resultats[2]
    Data_MinIncl    = resultats[3]
    Data_MaxIncl    = resultats[4]
    Data_MinAspect  = resultats[5]
    Data_MaxAspect  = resultats[6]

    Dataset = {}
    Types  = ['All', "East", "West"]
    for T in Types:
        Scatt_tot, I_tot,  Err_I_tot                     = Data_tot[T]["Scatt"],        Data_tot[T]['I'],         Data_tot[T]['Err_I']
        PI_tot, Err_PI_tot              =                                               Data_tot[T]['PI'],        Data_tot[T]['Err_PI']

        Scatt_MinPA, I_MinPA,  Err_I_MinPA               = Data_MinPA[T]["Scatt"],      Data_MinPA[T]['I'],       Data_MinPA[T]['Err_I']
        Scatt_MaxPA, I_MaxPA,  Err_I_MaxPA               = Data_MaxPA[T]["Scatt"],      Data_MaxPA[T]['I'],       Data_MaxPA[T]['Err_I']
        PI_MinPA, Err_PI_MinPA          =                                               Data_MinPA[T]['PI'],      Data_MinPA[T]['Err_PI']
        PI_MaxPA, Err_PI_MaxPA          =                                               Data_MaxPA[T]['PI'],      Data_MaxPA[T]['Err_PI']

        Scatt_MinIncl, I_MinIncl,  Err_I_MinIncl         = Data_MinIncl[T]["Scatt"],    Data_MinIncl[T]['I'],     Data_MinIncl[T]['Err_I']
        Scatt_MaxIncl, I_MaxIncl,  Err_I_MaxIncl         = Data_MaxIncl[T]["Scatt"],    Data_MaxIncl[T]['I'],     Data_MaxIncl[T]['Err_I']
        PI_MinIncl, Err_PI_MinIncl      =                                               Data_MinIncl[T]['PI'],    Data_MinIncl[T]['Err_PI']
        PI_MaxIncl, Err_PI_MaxIncl      =                                               Data_MaxIncl[T]['PI'],    Data_MaxIncl[T]['Err_PI']

        Scatt_MinAspect, I_MinAspect,  Err_I_MinAspect   = Data_MinAspect[T]["Scatt"],  Data_MinAspect[T]['I'],   Data_MinAspect[T]['Err_I']
        Scatt_MaxAspect, I_MaxAspect,  Err_I_MaxAspect   = Data_MaxAspect[T]["Scatt"],  Data_MaxAspect[T]['I'],   Data_MaxAspect[T]['Err_I']
        PI_MinAspect, Err_PI_MinAspect  =                                               Data_MinAspect[T]['PI'],  Data_MinAspect[T]['Err_PI']
        PI_MaxAspect, Err_PI_MaxAspect  =                                               Data_MaxAspect[T]['PI'],  Data_MaxAspect[T]['Err_PI']

        delta_I_MinPA      = np.where((Scatt_tot >= np.min(Scatt_MinPA))     & (Scatt_tot <= np.max(Scatt_MinPA)),      I_tot - np.interp(Scatt_tot,   Scatt_MinPA,      I_MinPA     ), 0)
        delta_I_MaxPA      = np.where((Scatt_tot >= np.min(Scatt_MaxPA))     & (Scatt_tot <= np.max(Scatt_MaxPA)),      I_tot - np.interp(Scatt_tot,   Scatt_MaxPA,      I_MaxPA     ), 0)
        delta_I_MinIncl    = np.where((Scatt_tot >= np.min(Scatt_MinIncl))   & (Scatt_tot <= np.max(Scatt_MinIncl)),    I_tot - np.interp(Scatt_tot,   Scatt_MinIncl,    I_MinIncl   ), 0)
        delta_I_MaxIncl    = np.where((Scatt_tot >= np.min(Scatt_MaxIncl))   & (Scatt_tot <= np.max(Scatt_MaxIncl)),    I_tot - np.interp(Scatt_tot,   Scatt_MaxIncl,    I_MaxIncl   ), 0)
        delta_I_MinAspect  = np.where((Scatt_tot >= np.min(Scatt_MinAspect)) & (Scatt_tot <= np.max(Scatt_MinAspect)),  I_tot - np.interp(Scatt_tot,   Scatt_MinAspect,  I_MinAspect ), 0)
        delta_I_MaxAspect  = np.where((Scatt_tot >= np.min(Scatt_MaxAspect)) & (Scatt_tot <= np.max(Scatt_MaxAspect)),  I_tot - np.interp(Scatt_tot,   Scatt_MaxAspect,  I_MaxAspect ), 0)
        delta_PI_MinPA     = np.where((Scatt_tot >= np.min(Scatt_MinPA))     & (Scatt_tot <= np.max(Scatt_MinPA)),     PI_tot - np.interp(Scatt_tot,   Scatt_MinPA,      PI_MinPA    ), 0)
        delta_PI_MaxPA     = np.where((Scatt_tot >= np.min(Scatt_MaxPA))     & (Scatt_tot <= np.max(Scatt_MaxPA)),     PI_tot - np.interp(Scatt_tot,   Scatt_MaxPA,      PI_MaxPA    ), 0)
        delta_PI_MinIncl   = np.where((Scatt_tot >= np.min(Scatt_MinIncl))   & (Scatt_tot <= np.max(Scatt_MinIncl)),   PI_tot - np.interp(Scatt_tot,   Scatt_MinIncl,    PI_MinIncl  ), 0)
        delta_PI_MaxIncl   = np.where((Scatt_tot >= np.min(Scatt_MaxIncl))   & (Scatt_tot <= np.max(Scatt_MaxIncl)),   PI_tot - np.interp(Scatt_tot,   Scatt_MaxIncl,    PI_MaxIncl  ), 0)
        delta_PI_MinAspect = np.where((Scatt_tot >= np.min(Scatt_MinAspect)) & (Scatt_tot <= np.max(Scatt_MinAspect)), PI_tot - np.interp(Scatt_tot,   Scatt_MinAspect,  PI_MinAspect), 0)
        delta_PI_MaxAspect = np.where((Scatt_tot >= np.min(Scatt_MaxAspect)) & (Scatt_tot <= np.max(Scatt_MaxAspect)), PI_tot - np.interp(Scatt_tot,   Scatt_MaxAspect,  PI_MaxAspect), 0)

        Err_I_min  = np.sqrt((Err_I_tot)**2  + np.array(delta_I_MinPA)**2  + np.array(delta_I_MinIncl)**2  + np.array(delta_I_MinAspect)**2)
        Err_I_max  = np.sqrt((Err_I_tot)**2  + np.array(delta_I_MaxPA)**2  + np.array(delta_I_MaxIncl)**2  + np.array(delta_I_MaxAspect)**2)
        Err_PI_min = np.sqrt((Err_PI_tot)**2 + np.array(delta_PI_MinPA)**2 + np.array(delta_PI_MinIncl)**2 + np.array(delta_PI_MinAspect)**2)
        Err_PI_max = np.sqrt((Err_PI_tot)**2 + np.array(delta_PI_MaxPA)**2 + np.array(delta_PI_MaxIncl)**2 + np.array(delta_PI_MaxAspect)**2)

        Dataset[T] = {'Scatt': Scatt_tot, 'I': I_tot, 'PI': PI_tot, 'Err_Scatt': Data_tot[T]['Err_Scatt'], 'Err_I': [Err_I_min, Err_I_max], 'Err_PI': [Err_PI_min, Err_PI_max], 'LB' : Data_tot[T]['LB'], 'Err_LB' : Data_tot[T]['Err_LB']}
    
    GUI_Folder, Fitting_Folder, SPF_Folder = Tools.Init_Folders()
    # filename = os.getcwd()+'\Results\Phase_Function\MaxSoft/' + img_name + '_' + File_name[:-5] + '.spf'
    filename = f"{SPF_Folder}/{img_name}_{File_name[:-5]}.spf"
    with open(filename, 'wb') as Data_PhF:
        pkl.dump(Dataset, Data_PhF)