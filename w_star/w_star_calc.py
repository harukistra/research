import numpy as np 
import xarray as xr

def omega_w(omega):
    g = 9.8
    Rho = rho()
    w = - omega / (g * Rho)
    return w

def load_data(y, month, date):
    if month < 10:
        month = "0" + str(month)
    if date < 10:
        date = "0" + str(date)
    path = f'/data_raid3/JRA3Q/data/{y}/JRA3Q_{y}{month}{date}.nc'
    data = xr.open_dataset(path)
    return data

def rho():
    p = load_data(2009, 8, 1)['lev']
    p0 = 1013
    Ts = 240
    R = 287
    rho0 = p0*10**2 / (R * Ts)
    rho = rho0 * (p/p0) 
    return rho

# def residual_w(year, month, date):

def w_star(year, month, date):
    data = load_data(year, month, date)
    v, t, omega = data["V"], data["T"], data["W"]
    w = omega_w(omega)
    v_mean = v.mean('lon')
    t_mean = t.mean('lon')
    w_mean = w.mean('lon')
    w_mean["phi"] = ('lat', np.radians(w.lat.values))
    cosphi = np.cos(w_mean["phi"].values)

    R = 287
    H = 7 * 10**3
    N_2 = 5 * 10**(-4)
    a = 6.37 * 10**6

    vt_prime = ((v - v_mean) * (t - t_mean)).mean("lon") * cosphi
    vt_prime["phi"] = ('lat', np.radians(vt_prime.lat))
    w_star = vt_prime.differentiate("phi") * R / (N_2 * H * a * cosphi) + w_mean     

    return w_star
