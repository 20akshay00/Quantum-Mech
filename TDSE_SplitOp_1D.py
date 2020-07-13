import numpy as np
import scipy.sparse.linalg
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.offsetbox import AnchoredText
import time

#strongly based on implementation at https://www.algorithm-archive.org/contents/split-operator_method/split-operator_method.html

def potential_config(position, type=1):
    v=[]
    if type==1: #harmonic potential
        v = 0.5 * position**2
    
    elif type==2: #1D barrier
        for x in position:
            if x<-90:
                v.append(-(x+100)*0.1j)
            elif x>90:
                v.append(-(x-90)*0.1j)
            elif x>-1.0 and x<1.0:
                v.append(3.7)
            else:
                v.append(0)
    
    return np.array(v)

    
def initial_config(xgrid, type=1, mu=-60, sigma=4, mass=1, energy= 3.5):
    if type==1: #stationary gaussian packet
        return (1/(2*np.pi*sigma**2)**0.25) * np.exp(-(xgrid-mu)**2/(4*sigma**2))
    if type==2: #moving gaussian packet
        return (1/(2*np.pi*sigma**2)**0.25)*(np.exp((2*mass*energy)**0.5*1j*xgrid-(xgrid-mu)**2/(4*sigma**2)))


class Parameters:
    """Simulation parameters"""
    
    def __init__(self, xlim: float, dx: float, dt: float, timesteps: int, mass: float, im_time: bool) -> None:
        self.xlim = xlim #limits of grid
        self.dt = dt 
        self.dx = dx
        self.timesteps = timesteps 
        self.mass = mass
        self.im_time = im_time #whether imaginary time or not
        
        self.res = int(2 * xlim/dx) #length of grid
        self.x = np.arange(-xlim + self.dx/2, xlim, self.dx) #grid
        self.dk = np.pi/xlim 
        self.k = np.concatenate((np.arange(0,self.res/2),np.arange(-self.res/2,0))) * self.dk
        
        
class Operators:
    """Holds operators"""
    
    def __init__(self, res: int) -> None:
        
        self.V = np.empty(res, dtype=complex) #Potential
        self.R = np.empty(res, dtype=complex) #Position component of Hamiltonian
        self.K = np.empty(res, dtype=complex) #Momentum component of Hamiltonian
        self.wfc = np.empty(res, dtype=complex) #Wave function
        
        
def init(par: Parameters, type1, type2) -> Operators:
    """Initialize wavefunction and potential"""
    opr = Operators(len(par.x))
    opr.V = potential_config(par.x, type1)
    opr.wfc = initial_config(par.x, type2, mass = par.mass)
    
    if par.im_time: #imaginary time to determine stationary state
        opr.K = np.exp(-0.5 * (1/par.mass) *(par.k ** 2) * par.dt)
        opr.R = np.exp(-0.5 * opr.V * par.dt)
    else:
        opr.K = np.exp(-0.5 * (1/par.mass) * (par.k ** 2) * par.dt * 1j)
        opr.R = np.exp(-0.5 * opr.V * par.dt * 1j)
        
    return opr

def split_operator(par: Parameters, opr: Operators):
    
    PSI = np.zeros((par.res, par.timesteps),complex)
    
    for i in range(par.timesteps):
        
        PSI[:,i] = opr.wfc[:]
        
        #Half step in position space
        opr.wfc = opr.wfc * opr.R
        
        #Fourier transform to momentum space
        opr.wfc = np.fft.fft(opr.wfc)
        
        #Full step in momentum space
        opr.wfc = opr.wfc * opr.K
        
        #Inverse fourier transform
        opr.wfc = np.fft.ifft(opr.wfc)
        
        #Half step in position space
        opr.wfc = opr.wfc * opr.R
        
        if par.im_time: #renormalization for imaginary time
            opr.wfc = opr.wfc / (par.dx * np.sum((opr.wfc)**2))**0.5
      
    return PSI
    
def data_visual(par, psi, frameskip=2, nframe=500, type=1):
    fig, axes = plt.subplots()
    
    axes.set_xlim(min(par.x),max(par.x))
    axes.set_ylim(-0.5,0.5)
    
    axes.set_title("TDSE; Split Operator method")
    axes.set_xlabel("Position")
    axes.set_ylabel(r"$|\psi|^2$")
    
    
    line, = axes.plot([],[], lw=0.75)
    clocktime = par.dt*np.arange(par.timesteps)
    
    def anim_init():
        line.set_data([],[])
        return line,

    def animate(i):
        line.set_data(par.x , psi[:,frameskip*i])
        time_text.set_text("t= %s" % round(clocktime[frameskip*i],3))
        return line, time_text, 

    start_time=time.time()
    
    P = potential_config(par.x, type)
    
    if(np.max(P).real==0):
        axes.plot(par.x, (-0.4/np.min(P.real))*np.array(P.real),color="black")
    
    else:
        axes.plot(par.x, (0.4/np.max(P.real))*np.array(P.real),color="black")

    time_text=axes.text(par.xlim-20 ,0.4,'',bbox=dict(facecolor='none',edgecolor='black',pad=5.0))
    anim= FuncAnimation(fig, animate, init_func=anim_init, frames=nframe, interval=35, blit=True)
    anim.save('split_op_1.gif', writer='-')
    
    print("--- %s seconds ---" % (time.time() - start_time))
    
def driver():
    p=Parameters(100 ,0.005 ,0.05 ,1500 ,1 ,False)
    o=init(p,2,2)
    
    start_time=time.time()
    wavefunction = split_operator(p,o)
    print("--- %s seconds ---" % (time.time() - start_time))
    
    data_visual(p, abs(wavefunction)**2, type=2, frameskip=5, nframe=299)
    
#check if psi**2 is plotted

driver()