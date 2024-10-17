import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
from scipy.stats import norm 

def f(x) : 
    return x**4 - 5*x**2 + 4 - 1/(1+np.exp(x**3))

x = np.linspace(-3, 3, 400)
y = f(x)
zeros = [-2.0743, -0.889642, 0.950748, 2.00003]
plt.figure(figsize=(8, 4))
plt.plot(x, y, label=fr"$f(x)$")
plt.scatter(zeros, [0,0,0,0], color='r', label=fr"Zéros de $f$", alpha = 1)
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend(loc='upper center')
plt.grid(True)
plt.axhline(0, color='black',linewidth=1.5)
plt.axvline(0, color='black',linewidth=1.5)
plt.show()

def bisection_method(a,b,f,seuil_int, seuil_approx) : 
    # Vérification de l'hypothèse du signe des images par f des bornes de l'intervalle : 
    if(f(a) * f(b) > 0) : 
        print("Erreur : les images sont de même signe.")
        return 0 
    
    # Au cas où les bornes sont solutions. 
    if(np.abs(f(a)) < seuil_approx) : return a
    if(np.abs(f(b)) < seuil_approx) : return b 
    
    left = a 
    right = b
    while max(np.abs(f(left)), np.abs(f(right))) > seuil_approx or right - left > seuil_int:
        mid = (left + right) / 2
        if(f(left) * f(mid) < 0) : 
            right = mid 
        else : 
            left = mid 
    return mid 

seuil_approx = 1e-9
seuil_int = 1e-6 
a = -2
b = 3 
bisection_method(a,b,f,seuil_int, seuil_approx)

def f_prime(x) : 
    return 4*x**3 + 3*x**2*np.exp(x**3)/(np.exp(x**3) + 1)**2 - 10*x

def newton_method(x_0, f, f_prime, seuil_int, seuil_approx) : 
    x_new = x_0
    x_old = x_0 - 1 
    nb_iterations = 0 
    while np.abs(f(x_new)) > seuil_approx or np.abs(x_new - x_old) > seuil_int:
        x_old = x_new
        x_new = x_old - f(x_old)/f_prime(x_old)
        nb_iterations += 1
    return x_new, nb_iterations

print(newton_method(-3, f, f_prime, seuil_int, seuil_approx))
print(newton_method(-0.5, f, f_prime, seuil_int, seuil_approx))
print(newton_method(0.5, f, f_prime, seuil_int, seuil_approx))
print(newton_method(3, f, f_prime, seuil_int, seuil_approx))

def secant_method(x_minus1, x_0, f, seuil_int, seuil_approx) : 
    x_new = x_0
    x_old = x_minus1
    nb_iterations = 0
    while np.abs(f(x_new)) > seuil_approx or np.abs(x_new - x_old) > seuil_int : 
        x_oldest = x_old
        x_old = x_new
        x_new = x_old - f(x_old)*(x_old-x_oldest)/(f(x_old)-f(x_oldest))
        nb_iterations += 1
    return x_new, nb_iterations

print(secant_method(-2.5,-2.25, f, seuil_int, seuil_approx))
print(secant_method(-1,-0.75, f, seuil_int, seuil_approx))
print(secant_method(0.75,1, f, seuil_int, seuil_approx))
print(secant_method(2.25,2.5, f, seuil_int, seuil_approx))

def F(x):
    return np.array([
        x[0]**3 + 2*x[0]*x[1] + x[2]**2 - x[1]*x[2] + 9, 
        2*x[0]**2 + 2*x[0]*x[1]**2 + x[1]**3*x[2]**2 - x[1]**2*x[2] - 2, 
        x[0]*x[1]*x[2] + x[0]**3 - x[2]**2 - x[0]*x[1]**2 - 4 ])

def gradient_F(x):
    return np.array([[3*x[0]**2 + 2*x[1], 2*x[0] - x[2], 2*x[2] - x[1]],
                     [4*x[0] + 2*x[1]**2, 4*x[0]*x[1] + 3*x[1]**2*x[2]**2 - 2*x[1]*x[2], 2*x[1]**2*x[2] - x[1]**2],
                     [x[1]*x[2] + 3*x[0]**2 - x[1]**2, x[0]*x[2] - 2*x[0]*x[1], x[0]*x[1] - 2*x[2]]])

result = gradient_F(np.array([-1, 3, 1]))
print(result)

def newton_N_method(x0, F, gradient_F, seuil_int, seuil_approx) : 
    x_new = x0
    x_old = x0 - np.ones(x0.shape) # Ou len(x0) pour une liste.
    nb_iterations = 0 
    while LA.norm(F(x_new)) > seuil_approx or LA.norm(x_new-x_old) > seuil_int : 
        x_old = x_new 
        grad = gradient_F(x_old)
        x_new = x_old - LA.solve(grad, F(x_old))
        nb_iterations += 1
    return x_new, nb_iterations

print(newton_N_method(np.array([1,2,3]), F, gradient_F, seuil_int, seuil_approx))

def compute_approx_grad(F, N, x, h) : 
    res = []
    for i in range(1,N+1) :
        line = []
        for j in range(1, N+1) : 
            e = np.zeros(N) 
            e[j-1] = 1 
            line.append((F(x+e*h)[i-1]-F(x)[i-1])/h)
        res.append(line)      
    return res

print("Approx. avec diff. finies : \n", compute_approx_grad(F,3, [1,1,1],0.001))
print("\n")
print("`Vrai` gradient : \n", gradient_F([1,1,1]))


def newton_N_method_approx(x0, F, seuil_int, seuil_approx) : 
    x_new = x0
    x_old = x0 - np.ones(len(x0))
    h = seuil_int 
    N = len(x0)
    nb_iterations = 0 
    while LA.norm(F(x_new)) > seuil_approx or LA.norm(x_new-x_old) > seuil_int : 
        x_old = x_new 
        grad_approx = compute_approx_grad(F, N, x_old, h)
        x_new = x_old - LA.solve(grad_approx, F(x_old))
        nb_iterations += 1
    return x_new, nb_iterations

x_new, nb_iterations = newton_N_method_approx(np.array([2,2,2]), F, seuil_int, seuil_approx)
print(x_new, " trouvé avec ", nb_iterations, " itérations.")
x_new, nb_iterations = newton_N_method_approx(np.array([1,2,3]), F, seuil_int, seuil_approx)
print(x_new, " trouvé avec ", nb_iterations, " itérations.")

def f(x, B, n, t_cash_flow, v_cash_flow) : 
    t_cash_flow = np.array(t_cash_flow)
    c_cash_flow = np.array(v_cash_flow)
    return np.sum(v_cash_flow * np.exp(-x * t_cash_flow)) - B

def f_prime(x, B, n, t_cash_flow, v_cash_flow) : 
    t_cash_flow = np.array(t_cash_flow)
    c_cash_flow = np.array(v_cash_flow)
    return -np.sum(v_cash_flow * t_cash_flow * np.exp(-x * t_cash_flow))

def bond_yield(x0, B, n, t_cash_flow, v_cash_flow) : 
    x_new = x0
    x_old = x0 - 1 
    seuil = 1e-6
    while np.abs(x_new - x_old) > seuil : 
        x_old = x_new 
        x_new = x_old - f(x_old, B, n, t_cash_flow, v_cash_flow)/f_prime(x_old, B, n, t_cash_flow, v_cash_flow)
    return x_new

n = 10
B = 100+1/32
C = 0.03375
F = 100
t_cash_flow = [6/12, 12/12, 18/12, 24/12, 30/12, 36/12, 42/12, 48/12, 54/12, 60/12]
c_cash_flow = [C/2*F] * (n-1) + [(1 + C/2)*F]

print("y = ", bond_yield(0.1, B, n, t_cash_flow, c_cash_flow)) # y = 3.3401 % 

def price_duration_convexity(n,t_cash_flow, c_cash_flow, y) : 
    B = 0
    D = 0
    C = 0
    for i in range (0,n) : 
        discount_factor = np.exp(-t_cash_flow[i]*y)
        B += c_cash_flow[i]*discount_factor
        D += t_cash_flow[i]*c_cash_flow[i]*discount_factor
        C += t_cash_flow[i]**2*c_cash_flow[i]*discount_factor 
    D /= B 
    C /= B
    return B,D,C

n = 10
B = 100+1/32
C = 0.03375
F = 100
t_cash_flow = [6/12, 12/12, 18/12, 24/12, 30/12, 36/12, 42/12, 48/12, 54/12, 60/12]
c_cash_flow = [C/2*F] * (n-1) + [(1 + C/2)*F]
y = bond_yield(0.1, B, n, t_cash_flow, c_cash_flow)

new_B, D, C = price_duration_convexity(n, t_cash_flow, c_cash_flow, y)
print("B = ", B)
print("new_B = ", new_B) # Erreur d'arrondi suite à l'approximation de y par la routine `bond_yield`. 
print("D = ", D)
print("C = ", C)

def f_call(x,C,S,K,T,q,r) : 
    d1 = (np.log(S/K) + (r-q+x*x/2)*(T))/(x*np.sqrt(T))
    d2 = d1-x*np.sqrt(T)
    return S*np.exp(-q*T)*norm.cdf(d1)-K*np.exp(-r*T)*norm.cdf(d2) - C

def vega(x,S,K,T,q,r) : 
    d1 = (np.log(S/K) + (r-q+x*x/2)*(T))/(x*np.sqrt(T))
    return S*np.exp(-q*T)*np.sqrt(T)*1/np.sqrt(2*np.pi)*np.exp(-d1*d1/2)

T = 1/4 
K = 30 
S = 30 
C = 2.5
q = 0.02 
r = 0.06 
seuil_approx = 1e-9
seuil_int = 1e-6 
a = 0.0001
b = 1

a = 0.0001
b = 1

def bisection_method_call_option(a, b, f, C, S, K, T, q, r, seuil_int, seuil_approx) : 
    if(f(a,C,S,K,T,q,r) * f(b,C,S,K,T,q,r)  > 0) : 
        print("Erreur : les images sont de même signe.")
        return 0 
    
    if(np.abs(f(a,C,S,K,T,q,r)) < seuil_approx) : return a
    if(np.abs(f(b,C,S,K,T,q,r)) < seuil_approx) : return b 
    
    left = a 
    right = b
    compteur = 0
    while max(np.abs(f(left,C,S,K,T,q,r)), np.abs(f(right,C,S,K,T,q,r))) > seuil_approx or right - left > seuil_int:
        mid = (left + right) / 2
        if(f(left,C,S,K,T,q,r) * f(mid,C,S,K,T,q,r) < 0) : 
            right = mid 
        else : 
            left = mid 
        compteur+=1
    return mid, compteur  

mid, compteur = bisection_method_call_option(a, b, f_call, C, S, K, T, q, r, seuil_int, seuil_approx)
print("Approx. = ", mid, " avec ", compteur, " itérations.")

x_0 = 0.5
x_minus1 = 0.6

def secant_method_call_option(x_minus1, x_0, f, C, S, K, T, q, r, seuil_int, seuil_approx) : 
    x_new = x_0
    x_old = x_minus1
    nb_iterations = 0
    while np.abs(f(x_new,C,S,K,T,q,r)) > seuil_approx or np.abs(x_new - x_old) > seuil_int : 
        x_oldest = x_old
        x_old = x_new
        x_new = x_old - f(x_old,C,S,K,T,q,r)*(x_old-x_oldest)/(f(x_old,C,S,K,T,q,r)-f(x_oldest,C,S,K,T,q,r))
        nb_iterations += 1
    return x_new, nb_iterations

x_new, nb_iterations = secant_method_call_option(x_minus1, x_0, f_call, C, S, K, T, q, r, seuil_int, seuil_approx)
print("Approx. = ", x_new, " avec ", nb_iterations, " itérations.")

def implied_volatility_call(x0,C,S,K,T,q,r) :  
    x_new = x0
    x_old = x0 - 1
    seuil = 1e-6
    nb_iterations = 0 
    while np.abs(x_new-x_old) > seuil : 
        x_old = x_new 
        x_new -= f_call(x_new, C,S,K,T,q,r)/vega(x_new,S,K,T,q,r)
        nb_iterations +=1 
    return x_new, nb_iterations 

x_new, nb_iterations = implied_volatility_call(x_0,C,S,K,T,q,r)
print("Approx. = ", x_new, " avec ", nb_iterations, " itérations.")

def f_put(x,P,S,K,T,q,r) : 
    d1 = (np.log(S/K) + (r-q+x*x/2)*(T))/(x*np.sqrt(T))
    d2 = d1-x*np.sqrt(T)
    return K*np.exp(-r*T)*norm.cdf(-d2)-S*np.exp(-q*T)*norm.cdf(-d1)-P


f_put(0.75, 3, 25, 25, 1, 0, 0.05)

T = 1
K = 25 
S = 25 
P = 3
q = 0
r = 0.05 
seuil_approx = 1e-9
seuil_int = 1e-6 

x_0 = 0.5

def implied_volatility_put(x0,P,S,K,T,q,r) : 
    x_new = x0
    x_old = x0 - 1
    seuil = 1e-6
    nb_iterations = 0 
    while np.abs(x_new-x_old) > seuil : 
        x_old = x_new 
        x_new -= f_put(x_new, P,S,K,T,q,r)/vega(x_new,S,K,T,q,r)
        nb_iterations +=1 
    return x_new, nb_iterations 

x_new, nb_iterations = implied_volatility_put(x_0,P,S,K,T,q,r) 
print("Approx. = ", x_new, " avec ", nb_iterations, " itérations.")

C = 0.02

def f_02(x) : 
    return 100*C/2*(np.exp(-0.5*0.0109)+np.exp(-0.0139)+np.exp(-0.75*(0.0139+x)))+100*(1+C/2)*np.exp(-2*x)-101-17.5/32

def df_02(x) :
    term1 = 100 * C / 2 * (-0.75) * np.exp(-0.75 * (0.0139 + x))
    term2 = 100 * (1 + C / 2) * (-2) * np.exp(-2 * x)
    return term1 + term2

res, nb = newton_method(0.0139, f_02, df_02, seuil_int, seuil_approx)
print("r(0,2) approx. ", res, " obtenu avec ", nb, " iterations.")

C = 0.03125
seuil_int = seuil_approx = 1e-12

def f_05(x) : 
    return 100*C/2*(np.exp(-0.5*0.0109) + np.exp(-0.0139) + 
                    np.exp(-0.75*(0.0139+x))+ np.exp(-2*0.012099) + np.exp(-2.5/3*(2.5*0.012099+0.5*x)) +  np.exp(-2*0.012099-x)
                    + np.exp(-3.5/3*(1.5*0.012099+1.5*x)) + np.exp(-4/3*(0.012099+2*x))
                    + np.exp(-4.5/3*(0.5*0.012099+2.5*x)))+100*(1+C/2)*np.exp(-5*x)-102-8/32

def df_05(x):
    const1 = 100 * C / 2
    const2 = 100 * (1 + C / 2)
    
    term1 = const1 * (-0.75 * np.exp(-0.75 * (0.0139 + x)))
    term2 = const1 * (-2.5/3 * np.exp(-2.5/3 * (2.5 * 0.012099 + 0.5 * x)))
    term3 = const1 * (-np.exp(-2 * 0.012099 - x))
    term4 = const1 * (-3.5/3 * np.exp(-3.5/3 * (1.5 * 0.012099 + 1.5 * x)))
    term5 = const1 * (-4/3 * np.exp(-4/3 * (0.012099 + 2 * x)))
    term6 = const1 * (-4.5/3 * np.exp(-4.5/3 * (0.5 * 0.012099 + 2.5 * x)))
    term7 = const2 * (-5 * np.exp(-5 * x))

    derivative = term1 + term2 + term3 + term4 + term5 + term6 + term7
    
    return derivative

C = 0.03125
res, nb = newton_method(0.012099, f_05, df_05, seuil_int, seuil_approx)
print("r(0,5) approx. ", res, " obtenu avec ", nb, " iterations.")

def zero_rate_curve(t) : 
    if(t < 0.5) : return zero_rate_curve(0.5)
    if(0.5 <= t and t<1) : return 2*(t-0.5)*0.0139+2*(1-t)*0.0109
    if(1 <= t and t < 2) : return (t-1)*0.012099+(2-t)*0.0139
    if(2 <= t and t < 5) : return ((t-2)*0.0268+(5-t)*0.012099)/3
    if(5 <= t and t <= 10) : return ((t-5)*0.03771+(10-t)*0.0268)/5
    return zero_rate_curve(10) 

t = np.linspace(0, 10.5, 1000)
y = np.array([zero_rate_curve(ti) for ti in t])

plt.figure(figsize=(10, 6))
plt.plot(t, y)
plt.xlabel(fr"$t$")
plt.ylabel(fr"$r(0,t)$")
plt.title("Courbe des taux zéro-coupon")
plt.grid(linestyle = '--', linewidth = 0.5)
plt.show()

def bond_price_with_zero_rate_curve(n,t_cash_flow,c_cash_flow,r_zero) : 
    B = 0 
    for i in range (0,n) : 
        discount_factor = np.exp(-t_cash_flow[i]*r_zero(t_cash_flow[i]))
        B += c_cash_flow[i]*discount_factor
    return B 

n = 6
t_cash_flow = [6/12, 12/12, 18/12, 24/12, 30/12, 36/12]
F = 100
C = 0.045
c_cash_flow = [C/2*F] * (n-1) + [(1 + C/2)*F]

B_approx = bond_price_with_zero_rate_curve(n, t_cash_flow, c_cash_flow, zero_rate_curve)
print("B_approx = ", B_approx)

B = 107+8/32
print("Erreur relative = ", np.abs(B-B_approx)/B, "soit ",np.abs(B-B_approx)/B*100, " %.") 

