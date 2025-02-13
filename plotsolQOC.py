import os
import subprocess
import argparse
import numpy as np

parser = argparse.ArgumentParser(prog='plotsolQOC',
                                 description='Plot quantum optimal control solution')

parser.add_argument('--mol', required=True, help='name of molecule')
parser.add_argument('--basis', required=True, help='name of basis set')
parser.add_argument('--outpath', required=True, help='output path')
parser.add_argument('--postfix', required=True, help='string to append to saved files')
parser.add_argument('--nsteps', type=int, required=True, help='number of time steps to take from 0 to T')
parser.add_argument('--dt', type=float, required=True, help='deltat (size of time step) to use')
parser.add_argument('--gpu', required=False, type=int, help='which GPU to use')

# actually parse command-line arguments
args = parser.parse_args()

# set gpu
if args.gpu:
    mygpu = args.gpu
else:
    mygpu = 0

os.environ["CUDA_VISIBLE_DEVICES"]=str(mygpu)
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import jit, grad, jacobian, lax, vmap

# matplotlib, with Agg to save to disk
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# set mol
mol = args.mol
if mol!='h2' and mol!='heh+':
    print("Molecule must either be H2 or HeH+!")
    sys.exit(1)

# set basis
basis = args.basis
if basis=='sto-3g':
    prefix = 'casscf22_s2_'
elif basis=='6-31g':
    prefix = 'casscf24_s15_'
else:
    print("Basis set must either be sto-3g or 6-31g!")
    sys.exit(1)
    
# load Hamiltonian
h0 = jnp.array(np.load('./data/'+prefix+mol+'_'+basis+'_hamiltonian.npz'))
n = h0.shape[0]

# load dipole moment matrix
m = jnp.array(np.load('./data/'+prefix+mol+'_'+basis+'_CI_dimat.npz'))

# load initial and final states
P0T = np.load('./data/'+mol+'_'+basis+'_P0T.npz')
thisalpha = jnp.array(P0T['alpha'])
thisbeta = jnp.array(P0T['beta'])

print("alpha = " + str(thisalpha))
print("beta = " + str(thisbeta))
# set outpath
path = args.outpath

# set postfix
postfix = args.postfix

# set nsteps
numsteps = args.nsteps
print("numsteps = " + str(numsteps))

# set dt
dt = args.dt
print("dt = " + str(dt))

def onematexp(evecs,expevals):
    return evecs @ jnp.diag(expevals) @ evecs.conj().T

manyeigh = vmap(jnp.linalg.eigh)
vonematexp = vmap(onematexp)

# given initial condition and forcing f, return trajectory a
def propSchro(f, alpha):
    manyhams = jnp.expand_dims(h0,0) + jnp.expand_dims(f,(1,2))*jnp.expand_dims(m,0)
    allevals, allevecs = manyeigh(manyhams)
    expevals = jnp.exp(-1j*dt*allevals)
    matexp = vonematexp(allevecs,expevals)
    
    a = jnp.concatenate([jnp.expand_dims(alpha,0), jnp.zeros((numsteps, n), dtype=jnp.complex128)])
    def amatbody(k, am):
        return am.at[k+1].set( matexp[k] @ am[k] )
    
    # forward trajectory
    a = lax.fori_loop(0, numsteps, amatbody, a)
    return a

# given forcing f, IC alpha, FC beta, return cost
def cost(f, alpha, beta):
    a = propSchro(f, alpha)
    resid = a[-1] - beta
    pen = jnp.real(jnp.sum(resid * resid.conj()))
    return 0.5*jnp.sum(f**2) + 0.5*rho*pen

# given forcing f, IC alpha, FC beta, return stats
def stats(f, alpha, beta):
    a = propSchro(f, alpha)
    resid = a[-1] - beta
    pen = jnp.real(jnp.sum(resid * resid.conj()))
    controlcost = 0.5*jnp.sum(f**2)
    return controlcost, pen

# load saved solution from disk
fstar = np.load(path+'final_'+mol+'_'+basis+postfix+'.npy')

# extract some data from log file
logfname = path+'livestats_'+mol+'_'+basis+postfix+'.txt'
print(logfname)
result3 = subprocess.run(['awk', '-v', 'FS= ', '{print $3}',logfname],check=True, capture_output=True, text=True)
normresid2 = np.array(list(map(float,result3.stdout.splitlines())))
result6 = subprocess.run(['awk', '-v', 'FS= ', '{print $6}',logfname],check=True, capture_output=True, text=True)
normf2 = np.array(list(map(float,result6.stdout.splitlines())))
result9 = subprocess.run(['awk', '-v', 'FS= ', '{print $9}',logfname],check=True, capture_output=True, text=True)
totalcost = np.array(list(map(float,result9.stdout.splitlines())))
result12 = subprocess.run(['awk', '-v', 'FS= ', '{print $12}',logfname],check=True, capture_output=True, text=True)
normgrad = np.array(list(map(float,result12.stdout.splitlines())))

# extract info
numiters = normresid2.shape[0]
itervec = np.arange(numiters)

# set plot font+size
font = {'weight' : 'bold', 'size' : 16}
matplotlib.rc('font', **font)

plt.rcParams['pdf.fonttype'] = 42

# make and save plots
# plot norm of residual versus iterations
plt.figure(figsize=(8,6))
plt.loglog(itervec,normresid2,color='blue')
plt.xlabel('iteration')
plt.ylabel(r'$\| a(T) - \beta \|$')
plt.grid(visible=True)
plt.savefig(path+'resid_'+mol+'_'+basis+postfix+'.pdf',bbox_inches = "tight")
plt.close()

# plot control cost versus iterations
plt.figure(figsize=(8,6))
plt.loglog(itervec,normf2,color='blue')
plt.xlabel('iteration')
plt.ylabel(r'$\| f \|$')
plt.grid(visible=True)
plt.savefig(path+'controlcost_'+mol+'_'+basis+postfix+'.pdf',bbox_inches = "tight")
plt.close()

# plot total cost versus iterations
plt.figure(figsize=(8,6))
plt.loglog(itervec,totalcost,color='blue')
plt.xlabel('iteration')
plt.ylabel('total cost')
plt.grid(visible=True)
plt.savefig(path+'totalcost_'+mol+'_'+basis+postfix+'.pdf',bbox_inches = "tight")
plt.close()

# plot norm of gradient versus iterations
plt.figure(figsize=(8,6))
plt.loglog(itervec,normgrad,color='blue')
plt.xlabel('iteration')
plt.ylabel('norm of gradient')
plt.grid(visible=True)
plt.savefig(path+'normgrad_'+mol+'_'+basis+postfix+'.pdf',bbox_inches = "tight")
plt.close()

# compute trajectory
a = propSchro(jnp.array(fstar), thisalpha)
a = np.array(a)

# autogenerate labels
labels=[]
for i in range(n):
    labels.append(r'| $a_{'+str(i+1)+'}(t)$ |')

# time vector
tvec = np.arange(numsteps+1)*dt

# this only works because we've hard-coded i==0 and i==11 for the case of 2x2 and 6x6 matrices
labeled = False
plt.figure(figsize=(9,6))
for i in range(n):
    if i==0:
        plt.plot(tvec, np.abs(a[:,i]), label=labels[i], color='#d01c8b', zorder=10, linewidth=2)
    elif i==(n-1):
        plt.plot(tvec, np.abs(a[:,i]), label=labels[i], color='#4dac26', zorder=10, linewidth=2)
    else:
        plt.plot(tvec, np.abs(a[:,i]), color='silver')

plt.legend(loc='upper center', bbox_to_anchor=(.5, 1.12), ncol=3, fancybox=False, shadow=False, frameon=False)

plt.xlabel('time')
plt.savefig(path+'controltraj_'+mol+'_'+basis+postfix+'.pdf',bbox_inches = "tight")
plt.close()

plt.figure(figsize=(8,6))
plt.plot(tvec[:-1], fstar)
plt.xlabel('time (a.u.)')
plt.ylabel('control amplitude f(t)')
plt.savefig(path+'controlsig_'+mol+'_'+basis+postfix+'.pdf',bbox_inches = "tight")
plt.close()