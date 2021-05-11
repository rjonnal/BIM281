import numpy as np
from fig2gif import GIF
from matplotlib import pyplot as plt
import sys,os
from matplotlib import animation

def pad(ax=None,frac=0.1):
    """Add some vertical padding to a plot."""
    if ax is None:
        ax = plt.gca()
    ymin = np.inf
    ymax = -np.inf
    for line in ax.lines:
        yd = line.get_ydata()
        if yd.min()<ymin:
            ymin = yd.min()
        if yd.max()>ymax:
            ymax = yd.max()
            
    yr = ymax-ymin
    ylim = ((ymin-yr*frac,ymax+yr*frac))
    ax.set_ylim(ylim)


def despine(ax=None):
    """Remove the spines from a plot. (These are the lines drawn
    around the edge of the plot.)"""
    if ax is None:
        ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)

def fourier_basis(t,n,T0=1.0):
    """Return the nth Fourier basis function, evaluated at points
    specified by t."""
    return np.exp(1j*2*np.pi*n*t/T0)

def fourier_synthesis(t,coefs,T0=1.0):
    """Synthesize a signal from Fourier basis coefficients, evaluated
    at points specified by t."""
    superposition = np.zeros(t.shape)
    for idx,c in enumerate(coefs):
        superposition = superposition + c*fourier_basis(t,idx,T0)
    return superposition

def fourier_analysis(t,signal,N_arr,T0_start=None,T0_end=None):
    dt = t[1]-t[0]
    out = np.zeros(N_arr.shape,dtype=np.complex)
    period_indices = np.where(np.logical_and(t>=T0_start,t<T0_end))[0]
    T0 = T0_end-T0_start
    for idx,n in enumerate(N_arr):
        out[idx] = np.sum(signal[period_indices]*np.exp(-1j*2*np.pi*n*t[period_indices]/T0)*dt)/T0
    return out

def analyze(signal,t,n,T0):
    dt = np.diff(t)[0]
    T0_idx = int(round(T0/dt))
    return np.sum(np.exp(-1j*2*np.pi*n*t[:T0_idx]/T0)*signal[:T0_idx]*dt)

def neaten(yticks=[0,1]):
    xt_arr = plt.gca().get_xticks()
    xtl_arr = []
    for xt in xt_arr:
        if xt==0:
            xtl_arr.append('$0$')
        elif xt%1:
            xtl_arr.append('$%0.1f\cdot T$'%xt)
        else:
            xtl_arr.append('$%dT$'%xt)
    plt.gca().set_xticklabels(xtl_arr)
    plt.xlabel('$t$')
    plt.yticks([0,1])
    plt.legend()


def u(t):
    out = np.zeros(len(t))
    out[np.where(t>0)]=1.0
    return out

def low_pass_irf(t,T=1.0,A=1.0):
    return A/T*u(t)*np.exp(-t/T)
    
def sawtooth(t,t0=0.0,T=1.0,A=1.0,invert=False):
    out = np.zeros(len(t))
    start = np.argmin(np.abs(t-t0))
    end = np.argmin(np.abs((t-(t0+T))))
    ramp = np.linspace(0,A,end-start)
    if invert:
        ramp = ramp[::-1]
    out[start:end]=ramp
    return out

def delta(t,t0=0.0,T=1.0,A=1.0):
    dt = np.mean(np.diff(t))
    out = np.zeros(len(t))
    out[np.argmin(np.abs(t-t0))]=1.0
    return out

def rect_old(t,t0=0.0,T=1.0,A=1.0):
    out = np.zeros(len(t))
    out[np.where(np.logical_and(t>t0,t<=t0+T))]=A
    return out


def rect(t,t0=0.0,T=1.0,A=1.0):
    out = np.zeros(len(t))
    out[np.abs(t-t0)<T/2.0] = A
    return out

def tri(t,t0=0.0,T=1.0,A=1.0):
    out = np.zeros(len(t))
    out[np.abs(t-t0)<T] = (1-np.abs(t-t0)/T)[np.abs(t-t0)<T]
    out = out*A
    return out

def tri_old(t,t0=0.0,T=1.0,A=1.0):
    out = np.zeros(len(t))
    start = np.argmin(np.abs(t-t0+T))
    end = np.argmin(np.abs((t-(t0+T))))
    mid = int(round((start+end)/2.0))
    ramp1 = np.linspace(0,A,mid-start)
    ramp2 = np.linspace(A,0,end-mid)
    out[start:mid]=ramp1
    out[mid:end] = ramp2
    return out

def sine(t,t0=0.0,T=1.0,A=1.0):
    return np.sin((t-t0)*2*np.pi/T)*A

def square(t,t0=0.0,T=1.0,A=1.0,D=0.5):
    out = np.zeros(t.shape)
    out[np.where((t-t0)%T<(D*T))]=A
    return out


def comb(t,t0=0.0,T=1.0):
    dt = t[1]-t[0]
    out = np.zeros(t.shape)
    out[np.where((t-t0)%T<dt)]=1.0
    return out

def plot_samples(x,y,comb,ax=None,**kwargs):
    assert len(x)==len(y)==len(comb)
    if ax is None:
        ax = plt.gca()

    for idx in np.where(comb>0.5)[0]:
        sample_x = x[idx]
        plt.axvline(sample_x,**kwargs)

    plt.grid(False)

def get_sampled_function(x,y,comb):
    assert len(x)==len(y)==len(comb)
    idx = np.where(comb>0.5)[0]
    sample_x = x[idx]
    sample_y = y[idx]
    return sample_x,sample_y
    
def convolve(signal_1,signal_2,tau_arr,t_arr,signal_1_label='',signal_2_label='',ylim=None,stretch=True):
    signal_2_rev = signal_2[::-1]
    dt = np.mean(np.diff(tau_arr))
    roll_indices = np.round(t_arr/dt).astype(np.int)
    conv_mat = []
    for ri in roll_indices:
        print(ri)
        conv_mat.append(np.roll(signal_2_rev,ri))

    conv_mat = np.array(conv_mat)
    smax = max(signal_1.max(),signal_2.max())
    smin = min(signal_1.min(),signal_2.min())
    srange = smax-smin
    padding = srange*.1
    if ylim is None:
        ylim = (smin-padding,smax+padding)


    fig,axs = plt.subplots(2,4)
    fig.set_size_inches((6,3))
    axs[0,0].plot(tau_arr,signal_1,'r-')
    axs[0,0].set_xlabel(r'$\tau$')
    axs[0,0].set_title(r'$x_1(\tau)$')
    axs[0,0].set_ylim(ylim)
    
    axs[0,1].plot(tau_arr,signal_2,'b-')
    axs[0,1].set_xlabel(r'$\tau$')
    axs[0,1].set_title(r'$x_2(\tau)$')
    axs[0,1].set_ylim(ylim)

    axs[0,2].plot(tau_arr,signal_2_rev,'b-')
    axs[0,2].set_xlabel(r'$\tau$')
    axs[0,2].set_title(r'$x_2(-\tau)$')
    axs[0,2].set_ylim(ylim)

    axs[0,3].plot(tau_arr,signal_2_rev,'b-')
    axs[0,3].set_xlabel(r'$\tau$')
    axs[0,3].set_title(r'$x_2(t-\tau)$')
    axs[0,3].set_ylim(ylim)
    
    xtl = []
    for tt in axs[0,3].get_xticks():
        if tt<0:
            xtl.append('t-%d'%abs(tt))
        elif tt==0:
            xtl.append('t')
        else:
            xtl.append('t+%d'%tt)
    axs[0,3].set_xticklabels(xtl)

    axs[1,0].imshow(conv_mat,aspect='auto',interpolation='none',clim=ylim)
    axs[1,0].set_xlabel(r'$\tau$')
    axs[1,0].set_ylabel(r'$t$')
    axs[1,0].set_yticks([])
    axs[1,0].set_xticks([])
    axs[1,0].text(len(tau_arr),0,r'$x_2(t-\tau)$, %s'%signal_2_label,color='w',ha='right',va='top')

    prod_mat = conv_mat*signal_1

    axs[1,1].imshow(prod_mat,aspect='auto',interpolation='none',clim=ylim)
    axs[1,1].set_xlabel(r'$\tau$')
    axs[1,1].set_ylabel(r'$t$')
    axs[1,1].set_yticks([])
    axs[1,1].set_xticks([])
    axs[1,1].text(len(tau_arr),0,r'$x_1(\tau)\times x_2(t-\tau)$, %s'%signal_2_label,color='w',ha='right',va='top')

    axs[1,2].imshow(prod_mat.T,aspect='auto',interpolation='none',clim=ylim)
    axs[1,2].set_ylabel(r'$\tau$')
    axs[1,2].set_xlabel(r'$t$')
    axs[1,2].set_yticks([])
    axs[1,2].set_xticks([])
    axs[1,2].text(len(t_arr),0,r'$x_1(\tau)\times x_2(t-\tau)$, %s'%signal_2_label,color='w',ha='right',va='top')

    conv = np.sum(prod_mat,axis=1)*dt
    
    axs[1,3].plot(t_arr,conv,'k-')
    axs[1,3].set_xlabel(r'$t$')
    axs[1,3].set_title(r'$(x_1*x_2)(t)$')
    axs[1,3].set_ylim(ylim)
    if stretch:
        axs[1,3].set_xlim([-0.5,t_arr.max()+0.5])
        
    
    outfn = 'convolution_%s_%s.png'%(signal_1_label,signal_2_label)
    plt.savefig(outfn,dpi=300)
    
    fig, (ax1,ax2,ax3) = plt.subplots(1,3)
    fig.set_size_inches((6,3))

    xlim = (tau_arr.min(),tau_arr.max())
    
    ax1.set_ylim(ylim)
    ax1.set_yticks([0,1])
    ax1.set_xlabel(r'$\tau$')
    ax1.set_xlim(xlim)
    ax1.set_title(r'$x_1(\tau)$,$x_2(t-\tau)$')
    
    ax2.set_ylim(ylim)
    ax2.set_yticks([0,1])
    ax2.set_xlabel(r'$\tau$')
    ax2.set_xlim(xlim)
    ax2.set_title(r'$x_1(\tau)\times x_2(t-\tau)$')
    
    ax3.set_ylim(ylim)
    ax3.set_yticks([0,1])
    ax3.set_title('convolution')
    if stretch:
        ax3.set_xlim([-0.5,t_arr.max()+.5])
    else:
        ax3.set_xlim(xlim)
        
    ax3.set_xlabel(r'$t$')

    ax3.plot(t_arr,conv,'ks',alpha=0.5)
    
    x1_line, = ax1.plot([], [], lw=1, color='r')
    x2_line, = ax1.plot([], [], lw=1, color='b')
    x3_line, = ax2.plot([], [], lw=1, color='k')
    c_line, = ax3.plot([],[],linestyle='none',marker='s',color='k',markersize=5,alpha=1.0)

    # initialization function: plot the background of each frame
    def init():
        x1_line.set_data([], [])
        x2_line.set_data([], [])
        x3_line.set_data([], [])
        #title1.set_text(ax1_title)
        #title2.set_text(ax2_title)
        c_line.set_data([],[])
        return (x1_line,x2_line,x3_line,c_line)

    def animate(k):
        if k<0 or k>=len(t_arr):
            return

        x1_line.set_data(tau_arr, signal_1)
        x2_line.set_data(tau_arr, conv_mat[k,:])
        x3_line.set_data(tau_arr, prod_mat[k,:])

        #title1.set_text(ax1_title%t_arr[k])
        #title2.set_text(ax2_title%t_arr[k])
        
        c_line.set_data([t_arr[k]],[conv[k]])
        return (x1_line,x2_line,x3_line,c_line)

    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=len(t_arr), interval=20, blit=True)

    outfn = 'convolution_%s_%s.gif'%(signal_1_label,signal_2_label)
    anim.save(outfn, writer='imagemagick', fps=10,dpi=100)
    
    plt.show()

# the mesh on which we'll simulate our functions is tau_arr:
def get_tau(s,dtau=0.001):
    return np.arange(-s,s,dtau)

# the mesh on which we'll simulate convolution is t_arr:
# t_arr can be coarser than tau_arr
def get_t(s,dt=0.1):
    return np.arange(-s,s,dt)

if __name__=='__main__':

    dt = 0.001
    t_arr = np.arange(-3,3,dt)
    r = rect(t_arr,t0=-0.5,T=1.0)

    plt.figure()
    plt.plot(t_arr,r)
    plt.figure()

    freq = np.arange(n_terms)*1.0/T
    plt.plot(np.abs(fourier_analysis(t_arr,r,100,T0_start=-1,T0_end=1.0)))
    plt.plot(np.abs(fourier_analysis(t_arr,r,100,T0_start=-2,T0_end=2.0)))
    plt.show()

    sys.exit()
    
    tau_arr = get_tau(5)
    t_arr = get_t(4)
    widerect = rect(tau_arr,T=2.0)
    tallrect = rect(tau_arr,T=1.0,A=2.0)
    saw = sawtooth(tau_arr)
    lp = low_pass_irf(tau_arr)
    deltafunc = delta(tau_arr)
    rect = rect(tau_arr)
    fast_sine = sine(tau_arr,T=0.5)
    vfast_sine = sine(tau_arr,T=0.1)
    slow_sine = sine(tau_arr,T=5.0)
    isaw = sawtooth(tau_arr,invert=True)


    #convolve(widerect,saw,tau_arr,t_arr,signal_1_label='widerect',signal_2_label='saw')
    #convolve(deltafunc,rect,tau_arr,t_arr,signal_1_label='delta',signal_2_label='rect',ylim=(-0.1,1.1))
    #convolve(rect,deltafunc,tau_arr,t_arr,signal_1_label='rect',signal_2_label='delta',ylim=(-0.1,1.1))

    #convolve(rect,rect,tau_arr,t_arr,signal_1_label='rect',signal_2_label='rect')
    #convolve(widerect,tallrect,tau_arr,t_arr,signal_1_label='widerect',signal_2_label='tallrect')
    #convolve(rect,lp,tau_arr,t_arr,signal_1_label='rect',signal_2_label='lowpass')

    #convolve(slow_sine,lp,tau_arr,t_arr,signal_1_label='slowsine',signal_2_label='lowpass',stretch=False)
    #convolve(fast_sine,lp,tau_arr,t_arr,signal_1_label='fastsine',signal_2_label='lowpass',stretch=False)
    #convolve(vfast_sine,lp,tau_arr,t_arr,signal_1_label='vfastsine',signal_2_label='lowpass',stretch=False)
    #convolve(rect,isaw,tau_arr,t_arr,signal_1_label='rect',signal_2_label='isaw')
    convolve(deltafunc,lp,tau_arr,t_arr,signal_1_label='delta',signal_2_label='lowpass',ylim=(-0.1,1.1))

