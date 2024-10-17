import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
from scipy.stats import norm 
import time 

np.sqrt(np.pi)*(norm.cdf(2*np.sqrt(2))-norm.cdf(0))

def f(x) : 
    return np.exp(-x*x) 

def midpoint_rule(a,b,n,f) :
    res = 0
    h = (b-a)/n
    for i in range(1,n+1) : 
        res += f(a+(i-1/2)*h)
    return h*res 

midpoint_rule(0,2,100,f)

a = 0
b = 2
n_values = [4,8,16,32,64,128,256]
errors = []
exact_value = np.sqrt(np.pi)*(norm.cdf(2*np.sqrt(2))-norm.cdf(0))

for n in n_values:
    approximation = midpoint_rule(a, b, n, f)
    error = np.abs(approximation - exact_value)
    errors.append(error)

plt.figure(figsize=(8, 6))
plt.plot(n_values, errors, label='Erreur de l\'approximation')
plt.axhline(y=0, color='r', linestyle='--', label='Valeur exacte')
plt.xlabel('Nombre de subdivisions (n)')
plt.ylabel('Erreur')
plt.title('Convergence de l\'erreur par la règle du point médian')
#plt.yscale('log')  #Si une échelle logarithmique est souhaitée. 
plt.legend()
plt.grid(True)

def trapezoidal_rule(a,b,n,f) : 
    h = (b-a)/n 
    res = f(a)/2 + f(b)/2 
    for i in range(1,n) : 
        res += f(a+i*h)
    return h*res

trapezoidal_rule(0,2,100,f)

a = 0
b = 2
n_values = [4,8,16,32,64,128,256]
errors = []
exact_value = np.sqrt(np.pi)*(norm.cdf(2*np.sqrt(2))-norm.cdf(0))

for n in n_values:
    approximation = trapezoidal_rule(a, b, n, f)
    error = np.abs(approximation - exact_value)
    errors.append(error)

plt.figure(figsize=(8, 6))
plt.plot(n_values, errors, label='Erreur de l\'approximation')
plt.axhline(y=0, color='r', linestyle='--', label='Valeur exacte')
plt.xlabel('Nombre de subdivisions (n)')
plt.ylabel('Erreur')
plt.title('Convergence de l\'erreur par la règle des trapèzes')
#plt.yscale('log')  #Si une échelle logarithmique est souhaitée. 
plt.legend()
plt.grid(True)

def simpson_rule(a,b,n,f) : 
    h = (b-a)/n 
    res = f(a)/6 + f(b)/6
    for i in range(1,n) : 
        res += f(a+i*h)/3 
    for i in range(1,n+1) : 
        res += 2*f(a+(i-1/2)*h)/3
    return h*res 

simpson_rule(0,2,100,f)

a = 0
b = 2
n_values = [4,8,16,32,64,128,256]
errors = []
exact_value = np.sqrt(np.pi)*(norm.cdf(2*np.sqrt(2))-norm.cdf(0))

for n in n_values:
    approximation = simpson_rule(a, b, n, f)
    error = np.abs(approximation - exact_value)
    errors.append(error)

plt.figure(figsize=(8, 6))
plt.plot(n_values, errors, label='Erreur de l\'approximation')
plt.axhline(y=0, color='r', linestyle='--', label='Valeur exacte')
plt.xlabel('Nombre de subdivisions (n)')
plt.ylabel('Erreur')
plt.title('Convergence de l\'erreur par la règle de Simpson')
#plt.yscale('log')  #Si une échelle logarithmique est souhaitée. 
plt.legend()
plt.grid(True)

simpson_rule(0,2,100,f)-(2/3*midpoint_rule(0,2,100,f)+1/3*trapezoidal_rule(0,2,100,f))

def tolerance_approximation(tol, a, b, f, numerical_approx) : 
    n = 4 
    integral0 = numerical_approx(a,b,n,f)
    n *= 2 
    integral1 = numerical_approx(a,b,n,f)
    while np.abs(integral1 - integral0) > tol : 
        integral0 = integral1 
        n *= 2 
        integral1 = numerical_approx(a,b,n,f)
    return integral1,n 

print('Erreur comise :',np.sqrt(np.pi)*(norm.cdf(2*np.sqrt(2))-norm.cdf(0))-tolerance_approximation(1e-12, 0, 2, f, simpson_rule)[0])
print('En {} subdivisions.'.format(tolerance_approximation(1e-12, 0, 2, f, simpson_rule)[1]))

def g(x) : 
    return np.sqrt(x)*np.exp(-x)

tolerance_approximation(1e-6, 1, 3, g, midpoint_rule)

tolerance_approximation(1e-6, 1, 3, g, trapezoidal_rule)

tolerance_approximation(1e-6, 1, 3, g, simpson_rule)

def r_inst(t) :
    return 0.05+0.005*np.log(1+t)

def r_zero(t) : 
    return 0.045+0.005*(1+t)*np.log(1+t)/t 

t_values = np.linspace(0.001,10,1000)
plt.figure(figsize=(8, 6)) 
plt.plot(t_values, r_inst(t_values), label='Taux instantané', color='blue') 
plt.plot(t_values, r_zero(t_values), label='Taux zéro coupon', color='red') 
plt.title(fr"Graphes de $r(t)$ et $r(0,t)$")  
plt.xlabel('t')  
plt.ylabel('Valeur de taux d\'intérêt')
plt.grid(True)  
plt.legend()  
plt.show()  

def bond_price_with_zero_rate_curve(n,t_cash_flow,c_cash_flow,r_zero) : 
    B = 0 
    for i in range (0,n) : 
        discount_factor = np.exp(-t_cash_flow[i]*r_zero(t_cash_flow[i]))
        B += c_cash_flow[i]*discount_factor
    return B 

def bond_price_with_instantaneous_rate_curve(n, t_cash_flow, c_cash_flow, r_inst) : 
    B = 0
    for i in range (0,n) : 
        integral = tolerance_approximation(1e-12, 0, t_cash_flow[i], r_inst, simpson_rule)
        discount_factor = np.exp(-integral[0])
        B += c_cash_flow[i]*discount_factor
    return B

n = 4    # Nombre de flux. 
C = 0.05 # Taux de coupon.
F = 100  # Nominal. 
c_cash_flow = [C/2*F] * (n-1) + [(1 + C/2)*F] # = [2.5, 2.5, 2.5, 102.5] après calculs.
t_cash_flow = [1/2, 1, 3/2, 2]

start_time = time.time()
print("En utilisant la courbe de taux zéro-coupon : B = ", bond_price_with_zero_rate_curve(n,t_cash_flow,c_cash_flow,r_zero))
end_time = time.time()
execution_time = end_time - start_time
print("Éxécuté en : ", execution_time, "secondes.")

start_time2 = time.time()
print("En utilisant la courbe de taux instantanés : B = ", bond_price_with_instantaneous_rate_curve(n, t_cash_flow, c_cash_flow, r_inst))
end_time2 = time.time()
execution_time2 = end_time2 - start_time2 
print("Éxécuté en : ", execution_time2, "secondes.")

def price_duration_convexity(n,t_cash_flow, v_cash_flow, y) : 
    B = 0
    D = 0
    C = 0
    for i in range (0,n) : 
        discount_factor = np.exp(-t_cash_flow[i]*y)
        B += v_cash_flow[i]*discount_factor
        D += t_cash_flow[i]*v_cash_flow[i]*discount_factor
        C += t_cash_flow[i]**2*v_cash_flow[i]*discount_factor 
    D /= B 
    C /= B
    return B,D,C

n = 5
C = 0.06
y = 0.09
F = 100
t_cash_flow = [6/12, 12/12, 18/12, 24/12, 30/12]
c_cash_flow = [C/2*F] * (n-1) + [(1 + C/2)*F] # = [3.0, 3.0, 3.0, 3.0, 103.0] après calculs.
print('B = ', price_duration_convexity(n,t_cash_flow, c_cash_flow, y)[0])
print('D = ', price_duration_convexity(n,t_cash_flow, c_cash_flow, y)[1])
print('C = ', price_duration_convexity(n,t_cash_flow, c_cash_flow, y)[2])

