import numpy as np
import scipy.sparse.linalg
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.offsetbox import AnchoredText
import time

#change values of xi, xf to set boundaries; M to determine time steps, dx and dt for precision, ma and ee for particle attributes

fig = plt.figure()
xi=-90 # left end of boundary
xf=90 # right end of boundary

# setting up to plot the data
ax = plt.axes(xlim=(xi+10, xf-10), ylim=(-0.5, 0.5))
line, = ax.plot([], [], lw=1.5)
ax.set_title("TDSE; 1D potential barrier")
ax.set_ylabel("Potential")
ax.set_xlabel("Position")


dx=0.005 #step size of space discretization
dt=0.05 #step size of time discretization
ma=1 # mass of particle

xgrid=np.arange(xi, xf, dx) #discrete position values
M=1000 #number of time steps
N=len(xgrid) #number of discrete position values
ee=12.5 #energy of particle 

#matrix elements used in calculating expectation value of momentum
diags=np.array([-1,1])
a=np.ones((N-1))
a[0]=0
b=np.ones((N-1))
b[N-2]=0
c=-2*np.ones((N-1))
c[0]=0
c[N-2]=0

xmatrix=scipy.sparse.spdiags(np.array([-a, b]), diags, N, N)
xmatrix=xmatrix.tocsc()
x2matrix=scipy.sparse.spdiags(np.array([a, c, b]),np.array([-1, 0, 1]), N, N)
x2matrix=x2matrix.tocsc()


def probability(wavef): # returns discretized integral of wavefunc
    return dx*np.sum(wavef)

def p_expect(wavefunc): #calculates expectation and stddev of momentum
    vec1=np.conj(wavefunc) #complex conjugate of wavefunc
    mean=-0.5j*np.sum(np.dot(vec1, xmatrix.dot(wavefunc))) #gives expectation of momentum (using finite difference of first derivative, written in matrix form)
    mean2=-(1/dx)*np.sum(np.dot(vec1, x2matrix.dot(wavefunc))) #gives expectation of momentum**2 (using finite difference of second derivative, written in matrix form)
    return [mean.real, (mean2.real-(mean.real)**2)**0.5] #returns average and stddev (img parts are tiny, and ignored)

def x_expect(wavefunc): #calculates expectation and stddev of position

    av=np.sum(np.multiply(abs(wavefunc)**2, xgrid))*dx 
    av2=np.sum(np.multiply(abs(wavefunc)**2, xgrid**2))*dx
    
    var=av2-av**2
    return [av,var**0.5]

    
def potential(position): #defines the potential energy distribution of the region
    v=[] 
    for x in position:
        if x<xi+10: # do not change; these are complex potentials at left boundary to prevent reflections
            v.append(-(x-xi)*ee/20*1j)
        elif x>xf-10: # do not change; these are complex potentials at right boundary to prevent reflections
            v.append(-(x-xf+10)*ee/20*1j)
        
        #change below this to create any potential setup as required;
        elif x>-22 and x<-20: 
            v.append(5)
        elif x>20 and x<22:
            v.append(5)
        else:
            v.append(0)
            
    return np.array(v)

'''
def final_config(wavef): #returns probability density at different parts of varying potential at end of simulation; used as diagnostic tool to check sim accuracy; does not work if potential is not piecewise step functions
    boundaryval=[]
    start=0
    V=potential(xgrid).real
    
    for i in range(len(V)-1): #finds different ranges where potential changes 
        if not (V[i]==V[i+1]):
            boundaryval.append([start,i+1])
            start=i+1
            
        elif i==len(V)-2 and V[i]==V[i+1]:
            boundaryval.append([start,i+1])
    print(boundaryval)
    ratio=[]
    
    for val in boundaryval:
        ratio.append(probability(wavef[val[0]:val[1]]))
        
    return ratio
'''


def psi_initial(mu=-50,sigma=10): #define initial wavefunction here; mu is mean and sigma is stddev of gaussian wavepacket; energy (ee) assigned above determines speed of travel (0 is stationary) 
    return (1/(2*np.pi*sigma**2)**0.25)*(np.exp((2*ma*ee)**0.5*1j*xgrid-(xgrid-mu)**2/(4*sigma**2)))

def TDSE(): #code to simulate time evolution of the wavefunction
    V=potential(xgrid) #define potential
    
    #define arrays having values of the diagonal elements of the matrix; 'j' is default complex number sqrt(-1) in python
    i = np.ones((N),complex) 
    alpha = ((1j)*dt/(4*ma*dx**2))*i
    xi = i + (1j*dt/2)*((1/(ma*dx**2))*i + V)
    gamma = i - (1j*dt/2)*((1/(ma*dx**2))*i + V)
    
    diags=np.array([-1,0,1]) #indices of the diagonal relative to central diagonal (0 is central; -1 is lower diagonal; +1 is upper diagonal, etc)
    
    vecs1=np.array([-alpha,xi,-alpha]) #tridiagonal values
    U1=scipy.sparse.spdiags(vecs1,diags,N,N) #filling a sparse matrix using diagonal values
    U1=U1.tocsc() #converting to compressed sparse column format, for LU decomposition
    
    vecs2=np.array([alpha,gamma,alpha])
    U2=scipy.sparse.spdiags(vecs2,diags,N,N)
    U2=U2.tocsc()
    
    PSI = np.zeros((N,M),complex) # N- discrete space, M- discrete time; PSI is 2d array, rows represent space and columns represent time
    PSI[:,0] = psi_initial() #assign initial wavefunction 
    LU=scipy.sparse.linalg.splu(U1) # LU decomposition of U1 <- returns object with 'solve' method
    
    for m in range(0,M-1): #solve system of equations; for each time step
        b=U2.dot(PSI[:, m]) 
        PSI[:, m+1] = LU.solve(b)
        
    prob=abs(PSI)**2 #probability density 
    print(probability(prob[:, 0])) #checking initial normalization; should be 1; 
    expectation_x= [x_expect(PSI[:, m]) for m in range(M)] #calculate expectation of x, for each time step
    expectation_p= [p_expect(PSI[:, m]) for m in range(M)] #calculate expectation of momentum, for each time step
    
    return prob, expectation_x, expectation_p #returns probability density matrix with values at each space point and each time point; and expectation values at each time step

#everything below here is to animate the data

start_time=time.time()
y, y1, y2=TDSE()
print("--- %s seconds ---" % (time.time() - start_time))
hproduct=[y1[i][1]*y2[i][1] for i in range(M)]

def init():
    line.set_data([], [])
    return line, 

timeclock=dt*np.arange(M)
def animate(i):
    line.set_data(xgrid, y[:,2*i])
    time_text.set_text("t= %s" % round(timeclock[2*i], 3))
    x_text.set_text("<x>= %s" % round(y1[2*i][0], 1))
    delx_text.set_text("$\sigma_x$= %s" % round(y1[2*i][1], 1))
    p_text.set_text("<p>= %s" % round(y2[2*i][0], 2))
    delp_text.set_text("$\sigma_p$= %s" % round(y2[2*i][1], 3))
    hproduct_text.set_text("$\sigma_x\sigma_p$= %s" %round(hproduct[2*i], 3))
    
    return line, time_text, x_text, delx_text, p_text, delp_text, hproduct_text,

start_time=time.time()
if(np.max(potential(xgrid)).real==0):
    plt.plot(xgrid, (-0.4/np.min(potential(xgrid).real))*np.array(potential(xgrid).real),color="black")
else:
    plt.plot(xgrid, (0.4/np.max(potential(xgrid).real))*np.array(potential(xgrid).real),color="black")
anchored_text=AnchoredText("Initial energy = %s"%ee, loc=2)
ax.add_artist(anchored_text)

time_text=ax.text(xf-10-20,0.43,'',bbox=dict(facecolor='none',edgecolor='black',pad=5.0))
x_text=ax.text(xi+10+3,-0.37,'',bbox=dict(facecolor='none',edgecolor='black',pad=3.0))
delx_text=ax.text(xi+10+3,-0.46,'',bbox=dict(facecolor='none',edgecolor='black',pad=3.0))
p_text=ax.text(xi+10+3,-0.19,'',bbox=dict(facecolor='none',edgecolor='black',pad=3.0))
delp_text=ax.text(xi+10+3,-0.28,'',bbox=dict(facecolor='none',edgecolor='black',pad=3.0))
hproduct_text=ax.text(xi+10+40,-0.46,'',bbox=dict(facecolor='none',edgecolor='black',pad=3.0))

anim= FuncAnimation(fig, animate, init_func=init, frames=490, interval=10, blit=True)

anim.save('TDSE_1d_well3.gif', writer='-')

print("--- %s seconds ---" % (time.time() - start_time))
