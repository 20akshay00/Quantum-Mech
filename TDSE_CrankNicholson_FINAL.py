import numpy as np
import scipy.sparse.linalg
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.offsetbox import AnchoredText
import matplotlib.animation as anim
import time

fig = plt.figure()
xi=-90
xf=90

ax = plt.axes(xlim=(xi+10, xf-10), ylim=(-0.5, 0.5))
line, = ax.plot([], [], lw=1.5)
ax.set_title("TDSE; 1D potential barrier")
ax.set_ylabel("Potential")
ax.set_xlabel("Position")


dx=0.005
dt=0.05
ma=1

xgrid=np.arange(xi,xf,dx)
M=1000
N=len(xgrid)
ee=3.5

diag=np.array([-1,1])
A=np.ones((N-1))
A[0]=0
B=np.ones((N-1))
B[N-2]=0
C=-2*np.ones((N-1))
C[0]=0
C[N-2]=0

xmatrix=scipy.sparse.spdiags(np.array([-A,B]),diag,N,N)
xmatrix=xmatrix.tocsc()
x2matrix=scipy.sparse.spdiags(np.array([A,C,B]),np.array([-1,0,1]),N,N)
x2matrix=x2matrix.tocsc()


def probability(wavef):
    return dx*np.sum(wavef)

def p_expect(wavefunc):
    vec1=np.conj(wavefunc)
    mean=-0.5j*np.sum(np.dot(vec1,xmatrix.dot(wavefunc)))
    mean2=-(1/dx)*np.sum(np.dot(vec1,x2matrix.dot(wavefunc)))
    return [mean.real, (mean2.real-(mean.real)**2)**0.5]

def x_expect(wavefunc):

    av=np.sum(np.multiply(abs(wavefunc)**2,xgrid))*dx
    av2=np.sum(np.multiply(abs(wavefunc)**2,xgrid**2))*dx
    
    var=av2-av**2
    return [av,var**0.5]

    
def potential(position):
    v=[] 
    for x in position:
        if x<xi+10:
            v.append(-(x-xi)*ee/20*1j)
        elif x>xf-10:
            v.append(-(x-xf+10)*ee/20*1j)
        elif x>-1 and x<1:
            v.append(3.7)
        else:
            v.append(0)
            
    return np.array(v)

def final_config(wavef):
    boundaryval=[]
    start=0
    V=potential(xgrid).real
    
    for i in range(len(V)-1):
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

def psi_initial(mu=-60,sigma=4):
    return (1/(2*np.pi*sigma**2)**0.25)*(np.exp((2*ma*ee)**0.5*1j*xgrid-(xgrid-mu)**2/(4*sigma**2)))

def TDSE():
    V=potential(xgrid)
    
    i = np.ones((N),complex)
    alpha = ((1j)*dt/(4*ma*dx**2))*i
    Xi = i + (1j*dt/2)*((1/(ma*dx**2))*i + V)
    gamma = i - (1j*dt/2)*((1/(ma*dx**2))*i + V)
    
    diags=np.array([-1,0,1]) #indices of the diagonal relative to central diagonal
    
    vecs1=np.array([-alpha,Xi,-alpha]) #tridiagonal values
    U1=scipy.sparse.spdiags(vecs1,diags,N,N) #filling a sparse matrix using diagonal values
    U1=U1.tocsc() #converting to compressed sparse column format, for LU decomposition
    
    vecs2=np.array([alpha,gamma,alpha])
    U2=scipy.sparse.spdiags(vecs2,diags,N,N)
    U2=U2.tocsc()
    
    PSI = np.zeros((N,M),complex) # N- discrete space, M- discrete time
    PSI[:,0] = psi_initial()
    LU=scipy.sparse.linalg.splu(U1) # LU decomposition of U1 <- returns object with 'solve' method
    
    for m in range(0,M-1):
        b=U2.dot(PSI[:,m]) 
        PSI[:,m+1] = LU.solve(b)
        
    prob=abs(PSI)**2
    print(probability(prob[:,0]))
    print(final_config(prob[:,M-1]))
    expectation_x= [x_expect(PSI[:,m]) for m in range(M)]
    expectation_p= [p_expect(PSI[:,m]) for m in range(M)]
    
    return prob,expectation_x,expectation_p

start_time=time.time()
y,y1,y2=TDSE()
print("--- %s seconds ---" % (time.time() - start_time))
hproduct=[y1[i][1]*y2[i][1] for i in range(M)]

def init():
    line.set_data([],[])
    return line, 

timeclock=dt*np.arange(M)
def animate(i):
    line.set_data(xgrid, y[:,2*i])
    time_text.set_text("t= %s" % round(timeclock[2*i],3))
    x_text.set_text(r"$\langle x \rangle = %s" % round(y1[2*i][0],1))
    delx_text.set_text(r"$\sigma_x$= %s" % round(y1[2*i][1],1))
    p_text.set_text(r"$\langle p \rangle$= %s" % round(y2[2*i][0],2))
    delp_text.set_text(r"$\sigma_p$= %s" % round(y2[2*i][1],3))
    hproduct_text.set_text(r"$\sigma_x\sigma_p$= %s" %round(hproduct[2*i],3))
    
    return line, time_text, x_text, delx_text, p_text, delp_text, hproduct_text

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

anim.save('TDSE_1d_well3.gif', fps=30)

print("--- %s seconds ---" % (time.time() - start_time))