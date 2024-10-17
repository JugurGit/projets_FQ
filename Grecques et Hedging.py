import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
from scipy.stats import norm 
import time 

K = 100
T = 1
r = 0.1 
q = 0 
sigmas = [0.1, 0.2, 0.3]

def d_1(t, S, K, T, sigma, r, q) : 
    return (np.log(S/K)+(r-q+sigma**2/2)*(T-t))/(sigma*np.sqrt(T-t))

def d_2(t, S, K, T, sigma, r, q) : 
    return d_1(t, S, K, T, sigma, r, q)-sigma*np.sqrt(T-t)

def call(t, S, K, T, sigma, r, q) : 
    return S*np.exp(-q*(T-t))*norm.cdf(d_1(t, S, K, T, sigma, r, q))-K*np.exp(-r*(T-t))*norm.cdf(d_2(t, S, K, T, sigma, r, q))

def put(t, S, K, T, sigma, r, q) : 
    return -S*np.exp(-q*(T-t))*norm.cdf(-d_1(t, S, K, T, sigma, r, q))+K*np.exp(-r*(T-t))*norm.cdf(-d_2(t, S, K, T, sigma, r, q))

def delta_call(t, S, K, T, sigma, r, q) : 
    return np.exp(-q*(T-t))*norm.cdf(d_1(t, S, K, T, sigma, r, q))

delta_call(0,120,K, T, sigmas[0], r, q)

def delta_put(t, S, K, T, sigma, r, q) : 
    return -np.exp(-q*(T-t))*norm.cdf(-d_1(t, S, K, T, sigma, r, q))
delta_put(0,120,K, T, sigmas[0], r, q)

delta_values = []
S_values = np.linspace(60,140,120)

for sigma in sigmas:
    delta_sigma = [delta_call(0, S, K, T, sigma, r, q) for S in S_values]
    delta_values.append(delta_sigma)

for i, sigma in enumerate(sigmas):
    plt.plot(S_values, delta_values[i], label=fr"$\sigma = {sigma}$")

plt.title(fr"Delta d'un call")
plt.xlabel('Prix de l\'actif sous-jacent (S)')
plt.ylabel('Delta')
plt.legend(); 

delta_values = []
S_values = np.linspace(60,140,120)

for sigma in sigmas:
    delta_sigma = [delta_put(0, S, K, T, sigma, r, q) for S in S_values]
    delta_values.append(delta_sigma)

for i, sigma in enumerate(sigmas):
    plt.plot(S_values, delta_values[i], label=fr"$\sigma = {sigma}$")

plt.title("Delta d'un put")
plt.xlabel('Prix de l\'actif sous-jacent (S)')
plt.ylabel('Delta')
plt.legend(); 

def delta_call_approx(t, S, K, T, sigma, r, q, ds) : 
    return (call(t, S+ds, K, T, sigma, r, q)-call(t, S-ds, K, T, sigma, r, q))/(2*ds)

def delta_put_approx(t, S, K, T, sigma, r, q, ds) : 
    return (put(t, S+ds, K, T, sigma, r, q)-put(t, S-ds, K, T, sigma, r, q))/(2*ds)

t = 0
S = 110
K = 100
T = 1
sigma = 0.2
r = 0.1
q = 0
errors_values = []
ds_values = [10**(-i) for i in range(1,5)]
for ds in ds_values : 
    val = np.abs(delta_call_approx(t, S, K, T, sigma, r, q, ds)-delta_call(t, S, K, T, sigma, r, q))
    errors_values.append(val)

plt.plot(ds_values, errors_values, marker='o')
plt.xscale('log')
plt.yscale('log')
plt.xlabel(fr"$dS$")
plt.ylabel('Erreur')
plt.title(fr"Tracé de l'erreur $|\Delta_{{approx}}-\Delta|$, pour un call")
plt.show()

errors_values_call = []
ds_values = [10**(-i) for i in range(1, 5)]

for ds in ds_values:
    val = np.abs(delta_call_approx(t, S, K, T, sigma, r, q, ds) - delta_call(t, S, K, T, sigma, r, q))
    errors_values_call.append(val)

errors_values_put = []
for ds in ds_values:
    val = np.abs(delta_put_approx(t, S, K, T, sigma, r, q, ds) - delta_put(t, S, K, T, sigma, r, q))
    errors_values_put.append(val)

fig, axs = plt.subplots(1, 2, figsize=(15, 6))

axs[0].plot(ds_values, errors_values_call, marker='o')
axs[0].set_xscale('log')
axs[0].set_yscale('log')
axs[0].set_xlabel(r"$dS$")
axs[0].set_ylabel('Erreur')
axs[0].set_title(r"Pour un call", fontsize = 15)


axs[1].plot(ds_values, errors_values_put, marker='o')
axs[1].set_xscale('log')
axs[1].set_yscale('log')
axs[1].set_xlabel(r"$dS$")
axs[1].set_ylabel('Erreur')
axs[1].set_title("Pour un put", fontsize = 15)

plt.suptitle(fr"Tracé de l'erreur $|\Delta_{{approx}}-\Delta|$", fontsize = 17)
plt.tight_layout()
plt.show()

def gamma(t, S, K, T, sigma, r, q) : 
    return np.exp(-q*(T-t))/(S*sigma*np.sqrt(T-t))*1/(np.sqrt(2*np.pi))*np.exp(-d_1(t, S, K, T, sigma, r, q)**2/2)

sigmas = [0.025,0.05,0.1, 0.2, 0.3]
K = 100
T = 1 
r = 0.1 
q = 0

gamma_values = []
S_values = np.linspace(60,140,120)

for sigma in sigmas:
    gamma_sigma = [gamma(0, S, K, T, sigma, r, q) for S in S_values]
    gamma_values.append(gamma_sigma)

for i, sigma in enumerate(sigmas):
    plt.plot(S_values, gamma_values[i], label=fr"$\sigma = {sigma}$")

plt.title(fr"Gamma d'une option vanille européenne")
plt.xlabel('Prix de l\'actif sous-jacent (S)')
plt.ylabel('Gamma')
plt.legend(); 

def gamma_approx(t, S, K, T, sigma, r, q, ds) : 
    forward = call(t, S+ds, K, T, sigma, r, q)
    backward = call(t, S-ds, K, T, sigma, r, q)
    current = call(t, S, K, T, sigma, r, q)
    return (forward-2*current+backward)/(ds)**2

t = 0
S = 110
K = 100
T = 1
sigma = 0.2
r = 0.1
q = 0
errors_values = []
ds_values = [10**(-i) for i in range(1,3)]
for ds in ds_values : 
    val = np.abs(gamma_approx(t, S, K, T, sigma, r, q, ds)-gamma(t, S, K, T, sigma, r, q))
    errors_values.append(val)

plt.plot(ds_values, errors_values, marker='o')
plt.xscale('log')
plt.yscale('log')
plt.xlabel(fr"$dS$")
plt.ylabel('Erreur')
plt.title(fr"Tracé de l'erreur $|\Gamma_{{approx}}-\Gamma|$")
plt.show()

S_values = [80,100,120]

def theta_call(t, S, K, T, sigma, r, q) : 
    if T == t:
        T = t + 1e-6
    else : 
        return q*S*np.exp(-q*(T-t))*norm.cdf(d_1(t, S, K, T, sigma, r, q))-1/(np.sqrt(2*np.pi))*S*np.exp(-q*(T-t))*sigma/(2*np.sqrt(T-t))*np.exp(-d_1(t, S, K, T, sigma, r, q)**2/2)-r*K*np.exp(-r*(T-t))*norm.cdf(d_2(t, S, K, T, sigma, r, q))

def theta_put(t, S, K, T, sigma, r, q) : 
    if T == t:
        T = t + 1e-6 
    else : 
        return -q*S*np.exp(-q*(T-t))*norm.cdf(-d_1(t, S, K, T, sigma, r, q))-1/(np.sqrt(2*np.pi))*S*np.exp(-q*(T-t))*sigma/(2*np.sqrt(T-t))*np.exp(-d_1(t, S, K, T, sigma, r, q)**2/2)+r*K*np.exp(-r*(T-t))*norm.cdf(-d_2(t, S, K, T, sigma, r, q))

thetas_values = []
T_values = np.linspace(0,10,200)

for S in S_values:
    t = [theta_call(0, S, K, T, sigma, r, q) for T in T_values]
    thetas_values.append(t)

for i, s in enumerate(S_values):
    plt.plot(T_values, thetas_values[i], label=fr"$S = {s}$")

plt.title(fr"Thêta d'un call européen")
plt.xlabel('Maturité T (en années)')
plt.ylim(-20)
plt.ylabel('Theta')
plt.legend(); 

thetas_values = []
T_values = np.linspace(0,10,200)

for S in S_values:
    t = [theta_put(0, S, K, T, sigma, r, q) for T in T_values]
    thetas_values.append(t)

for i, s in enumerate(S_values):
    plt.plot(T_values, thetas_values[i], label=fr"$S = {s}$")

plt.title(fr"Thêta d'un put européen")
plt.xlabel('Maturité T (en années)')
plt.ylim(-20)
plt.ylabel('Theta')
plt.legend(); 

def theta_call_approx(t, S, K, T, sigma, r, q, dT) : 
    forward = call(t, S, K, T + dT, sigma, r, q)
    current = call(t, S, K, T, sigma, r, q)
    return -(forward-current)/dT

theta_call_approx(0, 20,18, 0.5, 0.2, 0.05, 0, 1e-12)

def theta_put_approx(t, S, K, T, sigma, r, q, dT) : 
    forward = put(t, S, K, T + dT, sigma, r, q)
    current = put(t, S, K, T, sigma, r, q)
    return -(forward-current)/dT

t = 0
errors_values_call = []
dT_values = [10**(-i) for i in range(1, 8)]

for dT in dT_values:
    val = np.abs(theta_call_approx(t, S, K, T, sigma, r, q, dT) - theta_call(t, S, K, T, sigma, r, q))
    errors_values_call.append(val)

errors_values_put = []
for dT in dT_values:
    val = np.abs(theta_put_approx(t, S, K, T, sigma, r, q, dT) - theta_put(t, S, K, T, sigma, r, q))
    errors_values_put.append(val)

fig, axs = plt.subplots(1, 2, figsize=(15, 6))

axs[0].plot(dT_values, errors_values_call, marker='o')
axs[0].set_xscale('log')
axs[0].set_yscale('log')
axs[0].set_xlabel(r"$dT$")
axs[0].set_ylabel('Erreur')
axs[0].set_title(r"Pour un call", fontsize = 15)


axs[1].plot(dT_values, errors_values_put, marker='o')
axs[1].set_xscale('log')
axs[1].set_yscale('log')
axs[1].set_xlabel(r"$dT$")
axs[1].set_ylabel('Erreur')
axs[1].set_title("Pour un put", fontsize = 15)

plt.suptitle(fr"Tracé de l'erreur $|\Theta_{{approx}}-\Theta|$", fontsize = 17)
plt.tight_layout()
plt.show()

def rho_call(t, S, K, T, sigma, r, q) : 
    return K*(T-t)*np.exp(-r*(T-t))*norm.cdf(d_2(t, S, K, T, sigma, r, q))

rho_call(0,100,100,1,0.3,0.05, 0)

def rho_put(t, S, K, T, sigma, r, q) : 
    return -K*(T-t)*np.exp(-r*(T-t))*norm.cdf(-d_2(t, S, K, T, sigma, r, q))

T = 1
sigma = 0.2
q = 0

r_values = [0.05, 0.1, 0.2]
rho_values = []

S_values = np.linspace(60,140,120)

for r in r_values:
    rho = [rho_call(0, S, K, T, sigma, r, q) for S in S_values]
    rho_values.append(rho)

for i, r in enumerate(r_values):
    plt.plot(S_values, rho_values[i], label=fr"$r = {r}$")

plt.title(fr"Rho d'un call ($\rho$)")
plt.xlabel('Prix de l\'actif sous-jacent (S)')
plt.ylabel('Rho')
plt.legend(); 

r_values = [0.05, 0.1, 0.2]
rho_values = []

S_values = np.linspace(60,140,120)

for r in r_values:
    rho = [rho_put(0, S, K, T, sigma, r, q) for S in S_values]
    rho_values.append(rho)

for i, r in enumerate(r_values):
    plt.plot(S_values, rho_values[i], label=fr"$r = {r}$")

plt.title(fr"Rho d'un put ($\rho$)")
plt.xlabel('Prix de l\'actif sous-jacent (S)')
plt.ylabel('Rho')
plt.legend(); 

def rho_call_approx(t, S, K, T, sigma, r, q, dr) : 
    forward = call(t, S, K, T, sigma, r + dr, q)
    current = call(t, S, K, T, sigma, r, q)
    return (forward-current)/dr 

def rho_put_approx(t, S, K, T, sigma, r, q, dr) : 
    forward = put(t, S, K, T, sigma, r + dr, q)
    current = put(t, S, K, T, sigma, r, q)
    return (forward-current)/dr 

t = 0
errors_values_call = []
dr_values = [10**(-i) for i in range(1, 9)]

for dr in dr_values:
    val = np.abs(rho_call_approx(t, S, K, T, sigma, r, q, dr) - rho_call(t, S, K, T, sigma, r, q))
    errors_values_call.append(val)

errors_values_put = []
for dr in dr_values:
    val = np.abs(rho_put_approx(t, S, K, T, sigma, r, q, dr) - rho_put(t, S, K, T, sigma, r, q))
    errors_values_put.append(val)

fig, axs = plt.subplots(1, 2, figsize=(15, 6))

axs[0].plot(dr_values, errors_values_call, marker='o')
axs[0].set_xscale('log')
axs[0].set_yscale('log')
axs[0].set_xlabel(r"$dr$")
axs[0].set_ylabel('Erreur')
axs[0].set_title(r"Pour un call", fontsize = 15)


axs[1].plot(dr_values, errors_values_put, marker='o')
axs[1].set_xscale('log')
axs[1].set_yscale('log')
axs[1].set_xlabel(r"$dr$")
axs[1].set_ylabel('Erreur')
axs[1].set_title("Pour un put", fontsize = 15)

plt.suptitle(fr"Tracé de l'erreur $|\rho_{{approx}}-\rho|$", fontsize = 17)
plt.tight_layout()
plt.show()

def vega(t, S, K, T, sigma, r, q) : 
    if(T == t) : # Afin d'éviter la division par 0. 
        T = t + 1e-6
    return 1/np.sqrt(2*np.pi)*S*np.exp(-q*(T-t))*np.exp(-d_1(t, S, K, T, sigma, r, q)**2/2)*np.sqrt(T-t)

S_values = [80,100, 120]
K = 100
T = 1
r = 0.1 
q = 0 
sigmas = [0.1,0.2,0.3]

vega_values = []
S_values = np.linspace(60,140,120)

for sigma in sigmas:
    v = [vega(0, S, K, T, sigma, r, q) for S in S_values]
    vega_values.append(v)

for i, sigma in enumerate(sigmas):
    plt.plot(S_values, vega_values[i], label=fr"$\sigma = {sigma}$")

plt.title(fr"Vega d'une option vanille européenne")
plt.xlabel('Prix de l\'actif sous-jacent (S)')
plt.ylabel('Vega')
plt.legend(); 

K = 100
sigma = 0.3
r = 0.1
q = 0
S_values = np.linspace(60,140,120)
T_values = [0.1, 0.5, 1.0,2]

vega_values = []
for T in T_values:
    vega_s = [vega(0, S, K, T, sigma, r, q) for S in S_values]
    vega_values.append(vega_s)

for i, T in enumerate(T_values):
    plt.plot(S_values, vega_values[i], label=fr"$T = {T}$")

plt.title("Vega d'une option vanille européenne pour différentes maturités")
plt.xlabel('Prix de l\'actif sous-jacent (S)')
plt.ylabel('Vega')
plt.legend();

def vega_approx(t, S, K, T, sigma, r, q, dsigma) : 
    forward = call(t, S, K, T, sigma + dsigma, r, q)
    current = call(t, S, K, T, sigma, r, q)
    return (forward-current)/dsigma

t = 0
S = 110
K = 100
T = 1
sigma = 0.2
r = 0.1
q = 0

errors_values = []
dsigma_values = [10**(-i) for i in range(1,9)]

for dsigma in dsigma_values : 
    val = np.abs(vega_approx(t, S, K, T, sigma, r, q, dsigma) - vega(t, S, K, T, sigma, r, q))
    errors_values.append(val)

plt.plot(dsigma_values, errors_values, marker='o')
plt.xscale('log')
plt.yscale('log')
plt.xlabel(fr"$d\sigma$")
plt.ylabel('Erreur')
plt.title(fr"Tracé de l'erreur $|vega_{{approx}} - vega|$")
plt.show()

def vanna(t, S, K, T, sigma, r, q) : 
    return -np.exp(-q*(T-t))*1/np.sqrt(2*np.pi)*np.exp(-d_1(t, S, K, T, sigma, r, q)**2/2)*d_2(t, S, K, T, sigma, r, q)/sigma

vanna_values = []
S_values = np.linspace(60,140,120)

for sigma in sigmas:
    v = [vanna(0, S, K, T, sigma, r, q) for S in S_values]
    vanna_values.append(v)

for i, sigma in enumerate(sigmas):
    plt.plot(S_values, vanna_values[i], label=fr"$\sigma = {sigma}$")

plt.title(fr"Vanna d'un option vanille européenne")
plt.xlabel('Prix de l\'actif sous-jacent (S)')
plt.ylabel('Vanna')
plt.legend(); 

def volga(t, S, K, T, sigma, r, q) : 
    return S*np.exp(-q*(T-t))*np.sqrt(T-t)*1/np.sqrt(2*np.pi)*np.exp(-d_1(t, S, K, T, sigma, r, q)**2/2)*d_1(t, S, K, T, sigma, r, q)*d_2(t, S, K, T, sigma, r, q)/sigma

volga_values = []
S_values = np.linspace(60,140,120)

for sigma in sigmas:
    v = [volga(0, S, K, T, sigma, r, q) for S in S_values]
    volga_values.append(v)

for i, sigma in enumerate(sigmas):
    plt.plot(S_values, volga_values[i], label=fr"$\sigma = {sigma}$")

plt.title(fr"Volga d'un option vanille européenne")
plt.xlabel('Prix de l\'actif sous-jacent (S)')
plt.ylabel('Volga')
plt.legend(); 

import pandas as pd 

S = 50 
T = 0.5
sigma = 0.3
r = 0.04
q = 0
quotient_values = [0.5, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.5]

data = []
for quotient in quotient_values:
    val = 1 + (sigma**2 * S**2 / 2) * gamma(0, S, S/quotient, T, sigma, r, q) / theta_call(0, S, S/quotient, T, sigma, r, q)
    data.append([quotient, val])

df = pd.DataFrame(data, columns=['S/K', "Valeur"])

print(df) 

