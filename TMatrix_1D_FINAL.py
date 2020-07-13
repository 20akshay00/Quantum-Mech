import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema


#constants
m=1.0 #mass of particle
h=1.0 #plancks constant/ natural units

#incident energy range of particle
erange=np.concatenate([np.arange(0,0.621,0.01),np.arange(0.620,0.623,0.000001), np.arange(0.623,2.5,0.001), np.arange(2.5, 5, 0.1)])

dx=0.005 #step size in discrete grid
x=np.arange(-15.0,15.0,dx) #discretized position grid 

''' #GAUSSIAN POTENTIAL BARRIER
def convert(x): 
    f=[ 3*np.exp(-val**2)/np.pi**0.5 for val in x]
    potential=[]
    for i in range(0,len(f),2):
        potential.append((f[i]+f[i+1])/2.0)
        potential.append((f[i]+f[i+1])/2.0)
        
    return potential
'''

''' #DOUBLE BARRIER POTENTIAL (SQUARE)
def convert(position):
    v=[]
    for x in position:
        if x>-12 and x<-10:
            v.append(5)
        elif x>10 and x<12:
            v.append(5)
        else:
            v.append(0)
            
    return np.array(v)    
'''

def convert(position): #DOUBLE BARRIER POTENTIAL
    f=[(0.5*val**2 - 0.8)* np.exp(-0.1*val**2) for val in position]
    potential=[]
    for i in range(0,len(f),2): #approximate continuous function as series of step functions
        potential.append((f[i]+f[i+1])/2.0)
        potential.append((f[i]+f[i+1])/2.0)
        
    return potential

def mMatrix(E,V,a1,a2): #aids in computation of the transfer matrix
    delta=a2-a1
    if E>V:
        k=(2*m*(E-V)/h**2)**0.5
        M=np.reshape(np.array([np.cos(k*delta),-(1/k)*np.sin(k*delta),k*np.sin(k*delta),np.cos(k*delta)]),(2,2))
        return M
    elif E==V:
        return np.reshape(np.array([1,-delta,0,1]),(2,2))
    
    elif E<V:
        k=(2*m*(V-E)/h**2)**0.5
        M=np.reshape(np.array([np.cosh(k*delta),-(1/k)*np.sinh(k*delta),-k*np.sinh(k*delta),np.cosh(k*delta)]),(2,2))
        return M
    
def trans_coeff(E,boundary): #calculates transmission coefficient after finding overall m-Matrix
    mMat=np.reshape(np.array([1,0,0,1]),(2,2))
    
    for val in boundary:
        mMat=np.dot(mMat,mMatrix(E,val[0],val[1],val[2]))
        
    kl=(2*m*abs(E-boundary[0][0])/h**2)**0.5
    kr=(2*m*abs(E-boundary[-1][0])/h**2)**0.5
    
    if kl==0:
        kl=(2*m*abs(E-0.01-boundary[0][0])/h**2)**0.5
        
    tcoeff=4/((mMat[0][0]+(kr/kl)*mMat[1][1])**2+(kr*mMat[0,1]-(1/kl)*mMat[1][0])**2) #finds the transmission coefficient
    return tcoeff

def driver(): #driver code that finds transmission coefficients over given energy range
    potential = convert(x)
    boundaryval=[]

    start=x[0]
    for i in range(len(potential)-1):
        if not (potential[i]==potential[i+1]):
            boundaryval.append([potential[i],round(start,5),round(x[i+1],5)])
            start=x[i+1]
            
        elif i==len(potential)-2 and potential[i]==potential[i+1]:
            boundaryval.append([potential[i],round(start,5),round(x[i+1],5)])
            
    
    coeff=[trans_coeff(e,boundaryval) for e in erange]
    
    return coeff

def plotting(): #plots the results and outputs maxima/minima points
    fig, ax1 = plt.subplots()

    left, bottom, width, height = [0.5, 0.2, 0.3, 0.3]
    plt.xlabel("Energy")
    plt.ylabel("Transmission probability")
    ax2 = fig.add_axes([left, bottom, width, height])
    coefflist=driver()
    
    print(erange[argrelextrema(np.array(coefflist), np.greater)[0]]) #gives local maxima
    print(erange[argrelextrema(np.array(coefflist), np.less)[0]]) #gives local minima
    
    ax1.plot(erange, coefflist)
    axes = plt.gca()
    axes.set_xlim([np.min(erange)-1,np.min(erange)+1])
    axes.set_ylim([0,1.1])
    ax1.grid()
    
    axes2 = plt.gca()
    axes2.set_xlim([np.min(x),np.max(x)])
    axes2.set_ylim([np.min(convert(x))-1,np.max(convert(x))+1])
    ax2.plot(x, convert(x),color="black")
    ax2.grid()
    plt.show()
    
    