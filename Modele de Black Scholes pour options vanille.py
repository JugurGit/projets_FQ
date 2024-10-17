import numpy as np 
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import norm 
import time # Pour mesurer les durées.

def zelen_severo(t) : 
    z = np.abs(t)
    y = 1 / (1 + 0.2316419 * z)
    a1 = 0.319381530 
    a2 = -0.356563782
    a3 = 1.781477937
    a4 = -1.821255978
    a5 = 1.330274429
    m = 1 - np.exp(-t*t/2)*(a1*y + a2*pow(y,2) + a3*pow(y,3) + a4*pow(y,4) + a5*pow(y,5))/np.sqrt(2*np.pi)
    if(t > 0) : 
        resultat = m
    else : 
        resultat = 1 - m
    return resultat

zelen_severo(0)

def byrc2002B(t) : 
    return 1 - (t**2 + 5.575192695*t+12.77436324)/(t**3*np.sqrt(2*np.pi)+14.38718147*t**2+31.53531977*t + 25.548726)*np.exp(-t**2/2)

byrc2002B(0)

def page(t) : 
    y = np.sqrt(2/np.pi)*t*(1+0.044715*t**2)
    return 0.5*(1+np.tanh(y))

page(0)

def bagby(t) : 
    if(t <= 0) : return 0
    return 0.5 + 0.5*np.sqrt(1 - 1/30*(7*np.exp(-t*t/2)+16*np.exp(-t*t*(2-np.sqrt(2)))+(7+np.pi*t*t/4)*np.exp(-t*t)))

bagby(1e-15)

def cum_dist_normal_with_cdf(t) : 
    return norm.cdf(t)

t_values = np.linspace(0, 6, 1000)

cdf_approx_1 = np.array([zelen_severo(t) for t in t_values])
cdf_approx_2 = np.array([byrc2002B(t) for t in t_values])
cdf_approx_3 = np.array([page(t) for t in t_values])
cdf_approx_4 = np.array([bagby(t) for t in t_values])
cdf_exact = cum_dist_normal_with_cdf(t_values)

error_1 = np.abs(cdf_approx_1 - cdf_exact)
error_2 = np.abs(cdf_approx_2 - cdf_exact)
error_3 = np.abs(cdf_approx_3 - cdf_exact)
error_4 = np.abs(cdf_approx_4 - cdf_exact)

plt.figure(figsize=(10, 6))
plt.plot(t_values, error_1, label='Zelen & Severo')
plt.plot(t_values, error_2, label='Byrc (2002B)')
plt.plot(t_values, error_3, label='Page')
plt.plot(t_values, error_4, label='Bagby')
plt.yscale('log')
plt.ylim(top = 1e-3)
plt.xlabel('t')
plt.ylabel('Erreur en valeur absolue')
plt.title('Approximation de la fonction de répartition d\'une Gausienne standard')
plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.show()

# ------ Zelen & Severo ------ # 
start_time = time.time() 
for i in range(0,100) : 
    zelen_severo(i)
    
end_time = time.time() 
elapsed_time = end_time - start_time 
print(f"Zelen & Severo: {elapsed_time} secondes")

# ------ Byrc2002B ------ # 
start_time = time.time() 
for i in range(0,100) : 
    byrc2002B(i)
    
end_time = time.time() 
elapsed_time = end_time - start_time 
print(f"Byrc: {elapsed_time} secondes")

# ------ Page ------ # 
start_time = time.time() 
for i in range(0,100) : 
    page(i)
    
end_time = time.time() 
elapsed_time = end_time - start_time 
print(f"Page: {elapsed_time} secondes")

# ------ Bagby ------ # 
start_time = time.time() 
for i in range(0,100) : 
    bagby(i)
    
end_time = time.time() 
elapsed_time = end_time - start_time 
print(f"Bagby: {elapsed_time} secondes")

# ------ norm.cdf ------ # 
start_time = time.time() 
for i in range(0,100) : 
    cum_dist_normal_with_cdf(i)
    
end_time = time.time() 
elapsed_time = end_time - start_time 
print(f"norm.cdf : {elapsed_time} secondes") 

def box_muller():
    U1 = np.random.uniform(0, 1)
    U2 = np.random.uniform(0, 1)
    Z0 = np.sqrt(-2 * np.log(U1)) * np.cos(2 * np.pi * U2)
    Z1 = np.sqrt(-2 * np.log(U1)) * np.sin(2 * np.pi * U2)
    return Z0, Z1

Z0, Z1 = box_muller()
Z0, Z1

def several_box_muller(n) : 
    R = np.sqrt(-2 * np.log(stats.uniform.rvs(size = n//2))) #Retourne le quotient entier de la division. 
    theta = 2 * np.pi * stats.uniform.rvs(size = n//2)
    X = np.concatenate((R * np.cos(theta), R * np.sin(theta)))
    return X 
several_box_muller(4)

n = 10**3 
unif = stats.uniform(loc = -1, scale = 2)
z = unif.rvs(size = (2,n))
s = np.sum(z**2, axis = 0)
z = z[:, np.logical_and(0 < s, s <= 1)] #Filtrage des points, on garde ceux dans le cercle unité. 
print("Taille de l'échantillon : ", z.size)

t = np.linspace(0, 2*np.pi, 50)
plt.figure(figsize = (10,6))
plt.scatter(z[0], z[1])
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.plot(np.cos(t), np.sin(t), 'r', lw = 2, label = fr"$D(0,1)$")
plt.axis('image')
plt.title("Échantillonage par méthode de rejet")
plt.legend();

n = 10**7 

unif = stats.uniform(loc = -1, scale = 2)
z = unif.rvs(size = (2, int(n/2*1.5))) # On en génère un peu plus pour être certain d'en avoir assez 
                                       # après le filtrage. 
s = np.sum(z**2, axis = 0)
z = z[:, np.logical_and(0 < s, s <= 1)][:, :n//2] # Filtrage. On en conserve seulement n//2. 
print("Taille de l'échantillon : ", z.size)

s = np.sum(z**2, axis = 0)
X = (np.sqrt(-2*np.log(s))* z/np.sqrt(s)).ravel() # On met dim(X) = 1

norm = stats.norm 
plt.figure(figsize =(10,6))
x = np.linspace(-3,3)
plt.hist(X, bins = 30, density = True)
plt.plot(x,norm.pdf(x), 'r',label=fr"$N(0,1)$")
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.legend();

def marsaglia():
    while True:
        U = np.random.uniform(-1, 1)
        V = np.random.uniform(-1, 1)
        S = U**2 + V**2
        if S > 0 and S < 1:
            factor = np.sqrt(-2 * np.log(S) / S)
            Z0 = U * factor
            Z1 = V * factor
            return Z0, Z1

def several_marsaglia(n):
    unif = stats.uniform(loc = -1, scale = 2)
    z = unif.rvs(size = (2, int(n/2*1.5))) # On en génère un peu plus pour être certain d'en avoir assez 
                                       # après le filtrage. 
    s = np.sum(z**2, axis = 0)
    z = z[:, np.logical_and(0 < s, s <= 1)][:, :n//2] # Filtrage. On en conserve seulement n//2. 
    #print("Taille de l'échantillon : ", z.size)

    s = np.sum(z**2, axis = 0)
    X = (np.sqrt(-2*np.log(s))* z/np.sqrt(s)).ravel() # On met dim(X) = 1
    return X

def several_box_muller(n) : 
    R = np.sqrt(-2 * np.log(stats.uniform.rvs(size = n//2))) #Retourne le quotient entier de la division. 
    theta = 2 * np.pi * stats.uniform.rvs(size = n//2)
    X = np.concatenate((R * np.cos(theta), R * np.sin(theta)))
    return X 
several_box_muller(4)

n = 2

# ------ scipy.norm.rvs ------ # 
start_time = time.time() 

X = stats.norm.rvs(size = n) 

end_time = time.time() 
elapsed_time = end_time - start_time 
print(f"scipy.norm.rvs :", elapsed_time)

# ------ box_muller() ------ # 
start_time = time.time() 

X = box_muller()

end_time = time.time() 
elapsed_time = end_time - start_time 
print(f"Box muller :", elapsed_time)

# ------ marsaglia() ------ # 
start_time = time.time() 

X = marsaglia()

end_time = time.time() 
elapsed_time = end_time - start_time 
print(f"Marsaglia :", elapsed_time)

mu = 0.2
sigma = 0.4
npaths = 20000
nsteps = 20
npathsplot = 20

T = 1 
dt = T/nsteps
S0 = 100 

t = np.linspace(0,7, nsteps+1)

dX = (mu - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*np.random.normal(size = (npaths, nsteps))

X = np.concatenate((np.zeros((npaths,1)), np.cumsum(dX, axis = 1)), axis = 1)

S = S0*np.exp(X)

fig = plt.figure(figsize = (10,7))
ax = fig.add_subplot(111)

for i in range(0, npaths, round(npaths/npathsplot)) : 
    ax.plot(t, S[i,:])
ax.plot(t, np.mean(S,axis = 0), 'k--', label = "Tendance moyenne")
ax.set_title("Trajectoires des prix d'un actif régit par le modèle de Black Scholes")
ax.set_xlabel(fr"$t$")
ax.set_ylabel(fr"$S_t$")
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
ax.legend();

def black_scholes_call(t, S, K, T, sigma, r, q) : 
    d1 = (np.log(S/K) + (r-q+sigma*sigma/2)*(T-t))/(sigma*np.sqrt(T-t))
    d2 = d1 - sigma*np.sqrt(T-t) 
    C = S*np.exp(-q*(T-t))*zelen_severo(d1) - K*np.exp(-r*(T-t))*zelen_severo(d2)
    return C 

def black_scholes_put(t, S, K, T, sigma, r, q) : 
    d1 = (np.log(S/K) + (r-q+sigma*sigma/2)*(T-t))/(sigma*np.sqrt(T-t))
    d2 = d1 - sigma*np.sqrt(T-t) 
    P = -S*np.exp(-q*(T-t))*zelen_severo(-d1) + K*np.exp(-r*(T-t))*zelen_severo(-d2)
    return P 

t = 0
K = 100
T = 1
sigma = 0.2
r = 0.05
q = 0 
print("C_0 = ", black_scholes_call(t, K, K, T, sigma, r, q))
print("P_0 = ", black_scholes_put(t, K, K, T, sigma, r, q))

S_expiration = np.linspace(50, 150, 100)

call_initial_price = black_scholes_call(t, K, K, T, sigma, r, q)
put_initial_price = black_scholes_put(t, K, K, T, sigma, r, q)

profit_values_call = [max(S - K, 0) - call_initial_price for S in S_expiration]
profit_values_put = [max(K - S, 0) - put_initial_price for S in S_expiration]

plt.figure(figsize=(10, 6))
plt.plot(S_expiration, profit_values_call, label='Profit du Call')
plt.plot(S_expiration, profit_values_put, label='Profit du Put')
plt.scatter(K, 0, color='r', label=fr"Strike $K$", alpha = 1)
plt.axhline(0, color='black', linestyle='--', linewidth=1)
plt.xlabel('Prix du Sous-jacent à Expiration (S)')
plt.ylabel('Profit')
plt.title('Profit lors de l\'achat d\'une option européenne en fonction du prix du sous-jacent à expiration')
plt.legend()
plt.grid(True)
plt.show()

S_expiration = np.linspace(50, 150, 100)

call_initial_price = black_scholes_call(t, K, K, T, sigma, r, q)
put_initial_price = black_scholes_put(t, K, K, T, sigma, r, q)

profit_values_call = [call_initial_price - max(S - K, 0) for S in S_expiration]
profit_values_put = [put_initial_price - max(K - S, 0) for S in S_expiration]

plt.figure(figsize=(10, 6))
plt.plot(S_expiration, profit_values_call, label='Profit du Call')
plt.plot(S_expiration, profit_values_put, label='Profit du Put')
plt.axhline(0, color='black', linestyle='--', linewidth=1)
plt.scatter(K, 0, color='r', label=fr"Strike $K$", alpha = 1)
plt.xlabel('Prix du Sous-jacent à Expiration (S)')
plt.ylabel('Profit')
plt.title('Profit lors de la vente d\'une option européenne en fonction du prix du sous-jacent à expiration')
plt.legend()
plt.grid(True)
plt.show()

t = 0
S_0 = 32.5
K_1 = 30
K_2 = 35 
T = 1 
sigma = 0.2 
r = 0.05
q = 0 
print(f"V_0 = ", black_scholes_call(t, S_0, K_1, T, sigma, r, q)-black_scholes_call(t, S_0, K_2, T, sigma, r, q))

S_expiration = np.linspace(22.5, 42.5, 100)

call_1_initial_price = black_scholes_call(t, S_0, K_1, T, sigma, r, q)
call_2_initial_price = black_scholes_call(t, S_0, K_2, T, sigma, r, q)

profit_achat_actif = [(S - S_0) for S in S_expiration]
profit_bull_spread = [(max(S - K_1, 0) - max(S - K_2, 0) - call_1_initial_price + call_2_initial_price) for S in S_expiration]

plt.figure(figsize=(10, 6))
plt.plot(S_expiration, profit_bull_spread, label='Profit Bull Spread')
plt.plot(S_expiration, profit_achat_actif, label=fr"Profit Achat $S_0$")
plt.scatter(K_1, 0, color='r', label=fr"Strike $K_1$", alpha = 1)
plt.scatter(K_2, 0, color='black', label=fr"Strike $K_2$", alpha = 1)
plt.axhline(0, color='black', linestyle='--', linewidth=1)
plt.xlabel('Prix du Sous-jacent à Expiration (S)')
plt.ylabel('Profit')
plt.title('Bull Spread')
plt.legend()
plt.grid(True)
plt.show()

t = 0
S_0 = 32.5
K_1 = 30
K_2 = 35 
T = 1 
sigma = 0.2 
r = 0.05
q = 0 
print(f"V_0 = ", black_scholes_put(t, S_0, K_2, T, sigma, r, q)-black_scholes_put(t, S_0, K_1, T, sigma, r, q))

S_expiration = np.linspace(22.5, 42.5, 100)

put_1_initial_price = black_scholes_put(t, S_0, K_2, T, sigma, r, q)
put_2_initial_price = black_scholes_put(t, S_0, K_1, T, sigma, r, q)

profit_vente_actif = [(S_0 - S) for S in S_expiration]
profit_bear_spread = [(max(K_2-S, 0) - max(K_1-S, 0) - put_1_initial_price + put_2_initial_price) for S in S_expiration]

plt.figure(figsize=(10, 6))
plt.plot(S_expiration, profit_bear_spread, label='Profit Bear Spread')
plt.plot(S_expiration, profit_vente_actif, label=fr"Profit Vente $S_0$")
plt.scatter(K_1, 0, color='r', label=fr"Strike $K_1$", alpha = 1)
plt.scatter(K_2, 0, color='black', label=fr"Strike $K_2$", alpha = 1)
plt.axhline(0, color='black', linestyle='--', linewidth=1)
plt.xlabel('Prix du Sous-jacent à Expiration (S)')
plt.ylabel('Profit')
plt.title('Bear Spread')
plt.legend()
plt.grid(True)
plt.show()

t = 0
S_0 = 32.5
K_1 = 30
K_2 = 35 
K_3 = 40
T = 1 
sigma = 0.2 
r = 0.05
q = 0 
print(f"V_0 = ", black_scholes_call(t, S_0, K_1, T, sigma, r, q)-2*black_scholes_call(t, S_0, K_2, T, sigma, r, q)+black_scholes_call(t, S_0, K_3, T, sigma, r, q))

S_expiration = np.linspace(22.5, 47.5, 100)

call_1_initial_price = black_scholes_call(t, S_0, K_1, T, sigma, r, q)
call_2_initial_price = black_scholes_call(t, S_0, K_2, T, sigma, r, q)
call_3_initial_price = black_scholes_call(t, S_0, K_3, T, sigma, r, q)

profit_butterfly_spread = [(max(S - K_1, 0) - 2*max(S - K_2, 0) + max(S - K_3, 0) - call_1_initial_price + 2* call_2_initial_price - call_3_initial_price) for S in S_expiration]

plt.figure(figsize=(10, 6))
plt.plot(S_expiration, profit_butterfly_spread, label='Profit Butterfly Spread')
plt.axhline(0, color='black', linestyle='--', linewidth=1)
plt.scatter(K_1, 0, color='r', label=fr"Strike $K_1$", alpha = 1)
plt.scatter(K_2, 0, color='black', label=fr"Strike $K_2$", alpha = 1)
plt.scatter(K_3, 0, color='purple', label=fr"Strike $K_3$", alpha = 1)
plt.xlabel('Prix du Sous-jacent à Expiration (S)')
plt.ylabel('Profit')
plt.title('Butterfly Spread')
plt.legend()
plt.grid(True)
plt.show()

t = 0
S_0 = 32.5
K_1 = 30
K_2 = 35 
K_3 = 40
T = 1 
sigma = 0.2 
r = 0.05
q = 0 
print(f"V_0 = ", -black_scholes_call(t, S_0, K_1, T, sigma, r, q)+2*black_scholes_call(t, S_0, K_2, T, sigma, r, q)-black_scholes_call(t, S_0, K_3, T, sigma, r, q))

S_expiration = np.linspace(22.5, 47.5, 100)

call_1_initial_price = black_scholes_call(t, S_0, K_1, T, sigma, r, q)
call_2_initial_price = black_scholes_call(t, S_0, K_2, T, sigma, r, q)
call_3_initial_price = black_scholes_call(t, S_0, K_3, T, sigma, r, q)

profit_butterfly_spread = [(-max(S - K_1, 0) + 2*max(S - K_2, 0) - max(S - K_3, 0) + call_1_initial_price - 2* call_2_initial_price + call_3_initial_price) for S in S_expiration]

plt.figure(figsize=(10, 6))
plt.plot(S_expiration, profit_butterfly_spread, label='Profit Butterfly Spread')
plt.axhline(0, color='black', linestyle='--', linewidth=1)
plt.scatter(K_1, 0, color='r', label=fr"Strike $K_1$", alpha = 1)
plt.scatter(K_2, 0, color='black', label=fr"Strike $K_2$", alpha = 1)
plt.scatter(K_3, 0, color='purple', label=fr"Strike $K_3$", alpha = 1)
plt.xlabel('Prix du Sous-jacent à Expiration (S)')
plt.ylabel('Profit')
plt.title('Butterfly Spread')
plt.legend()
plt.grid(True)
plt.show()

t = 0
S_0 = 32.5
K_1 = 30
K_2 = 35 
T = 1 
sigma = 0.2 
r = 0.05
q = 0 
print(f"V_0 = ", black_scholes_call(t, S_0, K_1, T, sigma, r, q)-black_scholes_call(t, S_0, K_2, T, sigma, r, q)+black_scholes_put(t, S_0, K_2, T, sigma, r, q)-black_scholes_put(t, S_0, K_1, T, sigma, r, q))

t = 0
S_0 = 32.5
K = 35 
T_1 = 0.5
T_2 = 1 
sigma = 0.2 
r = 0.05
q = 0

S_expiration = np.linspace(22.5, 47.5, 100)

call_1_initial_price = black_scholes_call(t, S_0, K, T_1, sigma, r, q)
call_2_initial_price = black_scholes_call(t, S_0, K, T_2, sigma, r, q)

profit_calendar_spread = [( - max(S - K, 0) + black_scholes_call(T_1, S, K, T_2, sigma, r, q) + call_1_initial_price - call_2_initial_price) for S in S_expiration]
short_call = [(- max(S-K,0) + call_1_initial_price) for S in S_expiration]
long_call = [(black_scholes_call(T_1, S, K, T_2, sigma, r, q)-call_2_initial_price) for S in S_expiration]

plt.figure(figsize=(10, 6))
plt.plot(S_expiration, profit_calendar_spread, label='Profit Calendar Spread')
plt.plot(S_expiration, short_call, linestyle='--', linewidth = 1, label=fr"Profit Short call (Maturité $T_1$)")
plt.plot(S_expiration, long_call, linestyle='--', linewidth = 1, label=fr"Profit Long call (Maturité $T_2$)")
plt.axhline(0, color='black', linestyle='--', linewidth= 1)
plt.scatter(K, 0, color='r', label=fr"$K$", alpha = 1)
plt.xlabel('Prix du Sous-jacent à Expiration (S)')
plt.ylabel('Bénéfice')
plt.title('Calendar Spread with Call option')
plt.legend()
plt.grid(True)
plt.show()

S_expiration = np.linspace(22.5, 47.5, 100)

put_1_initial_price = black_scholes_put(t, S_0, K, T_1, sigma, r, q)
put_2_initial_price = black_scholes_put(t, S_0, K, T_2, sigma, r, q)

profit_calendar_spread = [( - max(K - S, 0) + black_scholes_put(T_1, S, K, T_2, sigma, r, q) + put_1_initial_price - put_2_initial_price) for S in S_expiration]
short_put = [(- max(K - S,0) + put_1_initial_price) for S in S_expiration]
long_put = [(black_scholes_put(T_1, S, K, T_2, sigma, r, q) - put_2_initial_price) for S in S_expiration]

plt.figure(figsize=(10, 6))
plt.plot(S_expiration, profit_calendar_spread, label='Profit Calendar Spread')
plt.plot(S_expiration, short_put, linestyle='--', linewidth = 1, label=fr"Profit Short put (Maturité $T_1$)")
plt.plot(S_expiration, long_put, linestyle='--', linewidth = 1, label=fr"Profit Long put (Maturité $T_2$)")
plt.axhline(0, color='black', linestyle='--', linewidth=1)
plt.scatter(K, 0, color='r', label=fr"$K$", alpha = 1)
plt.xlabel('Prix du Sous-jacent à Expiration (S)')
plt.ylabel('Bénéfice')
plt.title('Calendar Spread with Put option')
plt.legend()
plt.grid(True)
plt.show()

t = 0
S_0 = 32.5
K = 35 
T_1 = 0.5
T_2 = 1 
sigma = 0.2 
r = 0.05
q = 0

S_expiration = np.linspace(22.5, 47.5, 100)

call_1_initial_price = black_scholes_call(t, S_0, K, T_1, sigma, r, q)
call_2_initial_price = black_scholes_call(t, S_0, K, T_2, sigma, r, q)

profit_calendar_spread = [(max(S - K, 0) - black_scholes_call(T_1, S, K, T_2, sigma, r, q) - call_1_initial_price + call_2_initial_price) for S in S_expiration]
long_call = [ (max(S-K,0) - call_1_initial_price) for S in S_expiration]
short_call = [(-black_scholes_call(T_1, S, K, T_2, sigma, r, q)+call_2_initial_price) for S in S_expiration]

plt.figure(figsize=(10, 6))
plt.plot(S_expiration, profit_calendar_spread, label='Profit Calendar Spread')
plt.plot(S_expiration, short_call, linestyle='--', linewidth = 1, label=fr"Profit Short call (Maturité $T_2$)")
plt.plot(S_expiration, long_call, linestyle='--', linewidth = 1, label=fr"Profit Long call (Maturité $T_1$)")
plt.axhline(0, color='black', linestyle='--', linewidth= 1)
plt.scatter(K, 0, color='r', label=fr"Strike $K$", alpha = 1)
plt.xlabel('Prix du Sous-jacent à Expiration (S)')
plt.ylabel('Bénéfice')
plt.title('Reverse Calendar Spread with Call option')
plt.legend()
plt.grid(True)
plt.show()

t = 0
S_0 = 32.5 
K_1 = 32 
K_2 = 35 
T_1 = 1
T_2 = 2 
sigma = 0.2 
r = 0.05
q = 0

S_expiration = np.linspace(22.5, 47.5, 100)

call_1_initial_price = black_scholes_call(t, S_0, K_2, T_1, sigma, r, q)
call_2_initial_price = black_scholes_call(t, S_0, K_1, T_2, sigma, r, q)

profit_diagonal_spread = [( -max(S - K_2, 0)+black_scholes_call(T_1, S, K_1, T_2, sigma, r, q) + call_1_initial_price  - call_2_initial_price) for S in S_expiration]
short_call = [(-max(S-K_1, 0) + call_1_initial_price) for S in S_expiration]
long_call = [(black_scholes_call(T_1, S, K_1, T_2, sigma, r, q)-call_2_initial_price) for S in S_expiration]


plt.figure(figsize=(10, 6))
plt.plot(S_expiration, profit_diagonal_spread, label='Profit Diagonal Spread')
plt.plot(S_expiration, short_call, linestyle='--', linewidth = 1, label=fr"Profit Short call (Strike $K_2$, Maturité $T_1$)")
plt.plot(S_expiration, long_call, linestyle='--', linewidth = 1, label=fr"Profit Long call (Strike $K_1$, Maturité $T_2$)")
plt.axhline(0, color='black', linestyle='--', linewidth= 1)
plt.scatter(K_1, 0, color='r', label=fr"Strike $K_1$", alpha = 1)
plt.scatter(K_2, 0, color='black', label=fr"Strike $K_2$", alpha = 1)
plt.xlabel('Prix du Sous-jacent à Expiration (S)')
plt.ylabel('Bénéfice')
plt.title('Diagonal Spread with Call option')
plt.legend()
plt.grid(True)
plt.show()

t = 0
S_0 = 32.5
K = 35 
T = 1
sigma = 0.2 
r = 0.05
q = 0

S_expiration = np.linspace(22.5, 47.5, 100)

call_initial_price = black_scholes_call(t, S_0, K, T, sigma, r, q)
put_initial_price = black_scholes_put(t, S_0, K, T, sigma, r, q)

profit_straddle = [(max(S - K, 0) + max(K - S, 0) - call_initial_price - put_initial_price ) for S in S_expiration]
long_call = [(max( S - K, 0) - call_initial_price) for S in S_expiration]
long_put = [(max(K - S, 0) - put_initial_price) for S in S_expiration]

plt.figure(figsize=(10, 6))
plt.plot(S_expiration, profit_straddle, label='Profit Straddle')
plt.plot(S_expiration, long_call, linestyle='--', linewidth = 1, label=fr"Profit Long call")
plt.plot(S_expiration, long_put, linestyle='--', linewidth = 1, label=fr"Profit Long put")
plt.axhline(0, color='black', linestyle='--', linewidth = 1)
plt.scatter(K, 0, color='r', label=fr"$K$", alpha = 1)
plt.xlabel('Prix du Sous-jacent à Expiration (S)')
plt.ylabel('Profit')
plt.title('Profit Bottom Straddle')
plt.legend()
plt.grid(True)
plt.show()

t = 0
S_0 = 32.5
K = 35 
T = 1
sigma = 0.2 
r = 0.05
q = 0

S_expiration = np.linspace(22.5, 47.5, 100)

call_initial_price = black_scholes_call(t, S_0, K, T, sigma, r, q)
put_initial_price = black_scholes_put(t, S_0, K, T, sigma, r, q)

profit_straddle = [(- max(S - K, 0) - max(K - S, 0) + call_initial_price + put_initial_price ) for S in S_expiration]
short_call = [(- max( S - K, 0) + call_initial_price) for S in S_expiration]
short_put = [(- max(K - S, 0) + put_initial_price) for S in S_expiration]

plt.figure(figsize=(10, 6))
plt.plot(S_expiration, profit_straddle, label='Profit Straddle')
plt.plot(S_expiration, short_call, linestyle='--', linewidth = 1, label=fr"Profit Short call")
plt.plot(S_expiration, short_put, linestyle='--', linewidth = 1, label=fr"Profit Short put")
plt.axhline(0, color='black', linestyle='--', linewidth= 1)
plt.scatter(K, 0, color='r', label=fr"$K$", alpha = 1)
plt.xlabel('Prix du Sous-jacent à Expiration (S)')
plt.ylabel('Profit')
plt.title('Profit Top Straddle')
plt.legend()
plt.grid(True)
plt.show()

t = 0
S_0 = 32.5
K = 35 
T = 1
sigma = 0.2 
r = 0.05
q = 0

S_expiration = np.linspace(22.5, 47.5, 100)

call_initial_price = black_scholes_call(t, S_0, K, T, sigma, r, q)
put_initial_price = black_scholes_put(t, S_0, K, T, sigma, r, q)

profit_strip = [(max(S - K, 0) + 2 * max(K - S, 0) - call_initial_price - 2*put_initial_price) for S in S_expiration]
profit_strap = [(2*max(S - K, 0) + max(K - S, 0) - 2*call_initial_price - put_initial_price) for S in S_expiration]

plt.figure(figsize=(10, 6))
plt.plot(S_expiration, profit_strip, label='Profit Strip')
plt.plot(S_expiration, profit_strap, label='Profit Strap')
plt.scatter(K, 0, color='r', label=fr"Strike $K$", alpha = 1)
plt.axhline(0, color='black', linestyle='--', linewidth= 1)
plt.xlabel('Prix du Sous-jacent à Expiration (S)')
plt.ylabel('Profit')
plt.title('Profit/Perte d\'un strip et d\'un strap')
plt.legend()
plt.grid(True)
plt.show()

t = 0
S_0 = 32.5
K_1 = 30
K_2 = 35
T = 1
sigma = 0.2 
r = 0.05
q = 0

S_expiration = np.linspace(22.5, 47.5, 100)

call_initial_price = black_scholes_call(t, S_0, K_2, T, sigma, r, q)
put_initial_price = black_scholes_put(t, S_0, K_1, T, sigma, r, q)

profit_strangle = [(max(S - K_2, 0) + max(K_1 - S, 0) - call_initial_price - put_initial_price ) for S in S_expiration]
long_call = [(max( S - K_2, 0) - call_initial_price) for S in S_expiration]
long_put = [(max(K_1 - S, 0) - put_initial_price) for S in S_expiration]

plt.figure(figsize=(10, 6))
plt.plot(S_expiration, profit_strangle, label='Profit Strangle')
plt.plot(S_expiration, long_call, linestyle='--', linewidth = 1, label=fr"Profit Long call")
plt.plot(S_expiration, long_put, linestyle='--', linewidth = 1, label=fr"Profit Long put")
plt.axhline(0, color='black', linestyle='--', linewidth= 1)
plt.scatter(K_1, 0, color='r', label=fr"$K_1$", alpha = 1)
plt.scatter(K_2, 0, color='black', label=fr"$K_2$", alpha = 1)
plt.xlabel('Prix du Sous-jacent à Expiration (S)')
plt.ylabel('Profit')
plt.title('Profit/perte d\'un strangle')
plt.legend()
plt.grid(True)
plt.show()

