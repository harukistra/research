import numpy as np 
import xarray as xr
import modulelist as ml 

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

#%%
def calcw(sy, sm, sd, ey, em, ed):
    v, t, omega = ml.load_data_days(sy, sm, sd, ey, em, ed, "V"), \
        ml.load_data_days(sy, sm, sd, ey, em, ed, "T"), \
        ml.load_data_days(sy, sm, sd, ey, em, ed, "W")

    w_s = ml.w_star(v, t, omega)

    return w_s

W1 = calcw(2009, 1, 15, 2009, 2, 15)
#%%
W = W1.sel(lev="100")
print(W)

#%%
import matplotlib.pyplot as plt 
import numpy as np 
import matplotlib.cm as cm 
time = W["time"]

lat = W["lat"][1:143]
fig = plt.figure(1, figsize=(15, 7), dpi=300, facecolor="white")
ax = fig.add_subplot()
ax.set_ylabel('lat', fontsize=20)
ax.set_xlabel('time', fontsize=20)
ax.set_title('W at 100hPa 2009/1/15 ~ 2009/2/15', fontsize=15)
contf = ax.contourf(time, lat, W[:, 1:143].T, levels=np.arange(-0.01, 0.01, 0.001), cmap=cm.bwr, extend="both")
plt.colorbar(contf)
plt.show() 
# fig.savefig('/data_raid/home/haru/research/fig/w_star/lat_time/100hPa/2009_1_15_2_15_100hPa.png')

#%%
# print(time[0])
import datetime 
t = time[0].values
da = datetime.datetime.fromtimestamp(t)
# k = t.strftime("%m%d")
print(da)