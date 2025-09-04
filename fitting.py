import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, minimize
import math
import iminuit
import sys


df = pd.read_table('protoncalet2022.dat', sep='\s+', header=None)
darray = df.values
df1 = np.zeros((darray.shape[0], 2))
xerr = [[],[]]
yerr = [[],[]]
i=0
while i < df.shape[0]:
    df1[i,0] = darray[i,3]*10**darray[i,6]
    df1[i,1] = darray[i,7]*10**(-darray[i,12])
    xerr[0].append((darray[i,3]-darray[i,0])*10**darray[i,6])
    xerr[1].append((darray[i,1]-darray[i,3])*10**darray[i,6])
    yerr[0].append(math.sqrt(darray[i,8]**2 + darray[i,9]**2 + darray[i,11]**2)*10**(-darray[i,12]))
    yerr[1].append(math.sqrt(darray[i,8]**2 + darray[i,9]**2 + darray[i,10]**2)*10**(-darray[i,12]))
    i += 1
    
    
# C = 20000.0
# g = -2.83
# s = 2.4
# dg = 0.28
# E0 = 584.0
# dg1 = -0.34
# E1 = 9.3*10**3
# s1 = 30.0

def F(E, C, g, s, dg, E0, dg1, E1, s1):
    return C*E**g*(1+(E/E0)**s)**(dg/s)*(1+(E/E1)**s1)**(dg1/s1)
def chi2(par, df1, yerr):
    Chi2 = 0
    for e, m, yru, yrl in zip(df1[:,0], df1[:,1], yerr[0], yerr[1]):
        f = F(e, *par)
        if f > m:
            Chi2 += ((m-f)/yru)**2
        else:
            Chi2 += ((m-f)/yrl)**2
    return Chi2
#par, cov = curve_fit(F, df1[:,0], df1[:,1], sigma=yerr[1], absolute_sigma=True, p0=[20000.0, -2.83, 2.4, 0.28, 584.0, -0.34, 9.3*10**3, 30.0])
res = minimize(chi2, [20000.0, -2.83, 2.4, 0.28, 584.0, -0.34, 9.3*10**3, 30.0], args=(df1, yerr), method='Nelder-Mead')
print(res)
# sys.exit()

lnE = np.arange(1, 5, 0.01)
E = 10**lnE
Phi = F(E, *res['x'])
Chi2 = chi2(res['x'], df1, yerr)
par = res['x']
#print(Chi2)

#Chi2 = np.sum(((df1[:,1]-F(df1[:,0], *par))/yerr[0])**2)
ndf = df1.shape[0] - len(par)
#errs = np.sqrt(cov.diagonal()/Chi2*ndf)
formatted_par = [f'{p:.2e}' for p in par]
#formatted_errs = [f'{e:.2e}' for e in errs]
print(f'C = {formatted_par[0]}')
print(f'g = {formatted_par[1]}')
print(f's = {formatted_par[2]}')
print(f'dg = {formatted_par[3]}')
print(f'E0 = {formatted_par[4]}')
print(f'dg1 = {formatted_par[5]}')
print(f'E1 = {formatted_par[6]}')
print(f's1 = {formatted_par[7]}')
print(f'Chi2 = {Chi2}')
print(f'Chi2/ndf = {Chi2/ndf}')

plt.errorbar(df1[:,0], df1[:,1]*df1[:,0]**(2.7), xerr=xerr, yerr=yerr*df1[:,0]**(2.7), fmt='o', color='black', markersize=3, label='data', capsize=1, elinewidth=0.5, capthick=0.5)
plt.plot(E, Phi*E**(2.7), color='red', label='fit')
plt.xlim(50, 70000)
plt.ylim(6000, 16000)
plt.xscale('log')
plt.show()