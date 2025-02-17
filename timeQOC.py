import os
import argparse
import numpy as np
import time

parser = argparse.ArgumentParser(prog='timeQOC',
                                 description='Time the adjgrad and adjhess QOC functions')

parser.add_argument('--mol', required=True, help='name of molecule')
parser.add_argument('--basis', required=True, help='name of basis set')
parser.add_argument('--outpath', required=True, help='output path')
parser.add_argument('--postfix', required=True, help='string to append to saved files')
parser.add_argument('--nsteps', type=int, required=True, help='number of time steps to take from 0 to T')
parser.add_argument('--numruns', required=True, type=int, help='number of runs')
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
h0 = np.load('./data/'+prefix+mol+'_'+basis+'_hamiltonian.npz')
n = h0.shape[0]

# load dipole moment matrix
m = np.load('./data/'+prefix+mol+'_'+basis+'_CI_dimat.npz')

# load initial and final states
P0T = np.load('./data/'+mol+'_'+basis+'_P0T.npz')
thisalpha = jnp.array(P0T['alpha'])
thisbeta = jnp.array(P0T['beta'])

print("alpha = " + str(thisalpha))
print("beta = " + str(thisbeta))

# set outpath
outpath = args.outpath

# set postfix
postfix = args.postfix

# set nsteps
numsteps = args.nsteps
print("numsteps = " + str(numsteps))

# set dt
dt = args.dt
print("dt = " + str(dt))

rho = 1e6

# (d/dx) \exp(-1j*dt*(h0 + x m))
# where you pass in the eigenvectors and eigenvalues of (h0 + x m)
def firstderiv(evecs, evals):
    amat = evecs.conj().T @ (-1j*dt*m) @ evecs
    dvec = -1j*dt*evals
    dvec1, dvec2 = jnp.meshgrid(dvec, dvec)
    mask = jnp.ones((n,n)) - jnp.eye(n)
    numer = jnp.exp(dvec1) - jnp.exp(dvec2)
    denom = (dvec1 - dvec2)*mask + jnp.eye(n)
    derivmat = mask*numer/denom + jnp.diag(jnp.exp(dvec))
    qmat = evecs @ (amat * derivmat) @ evecs.conj().T
    return qmat

mask = jnp.ones((n,n)) - jnp.eye(n)

# (d^2/dx^2) \exp(-1j*dt*(h0 + x m))
# where you pass in the eigenvectors and eigenvalues of (h0 + x m)
def secondderiv(evecs, rawevals):
    evals = (-1j*dt)*rawevals
    a = (evecs.conj().T @ m @ evecs) * (-1j*dt)
    expevals = jnp.exp(evals)
    evals1, evals2 = jnp.meshgrid(evals, evals, indexing='ij')
    expevals1, expevals2 = jnp.meshgrid(expevals, expevals, indexing='ij')
    # first D_{ii}=D_{kk} term
    diagterm1 = expevals1*jnp.diag(jnp.diag(a*a))
    # second D_{ii}=D_{kk} term
    numer1 = -expevals1 + evals1*expevals1 - evals2*expevals1 + expevals2
    denom1 = (evals1-evals2)**2 + jnp.eye(n)
    maska = mask * a
    diagterm2 = jnp.eye(n) * 2*((numer1/denom1 * maska) @ maska)
    # first D_{ii}!=D_{kk} term
    frac1 = numer1/denom1 * mask
    term1 = frac1*2*(jnp.diag(a)*a).conj().T
    # second D_{ii}!=D_{kk} term
    numer2 = -expevals1 + evals1*expevals2 - evals2*expevals2 + expevals2
    denom2 = (evals1-evals2)**2 + jnp.eye(n)
    frac2 = numer2/denom2 * mask
    term2 = frac2*2*(a*jnp.diag(a))
    # third D_{ii}!=D_{kk} term
    matij = mask*(1.0/((evals1-evals2) + jnp.eye(n)))
    matind1a = (expevals1 * matij) * a
    matind2a = (expevals2 * matij) * a
    term3 = 2*mask*((matind2a) @ (matij*a))
    term3 -= 2*matij*( matind1a @ maska )
    term3 -= 2*matij*( maska @ matind2a ) 
    # put it all together
    # udagru stands for "U^{\dagger} R U"
    udagru = term1 - term2 - term3 + diagterm1 + diagterm2
    return evecs @ udagru @ evecs.conj().T

def gradal(l,expderiv,a,matexp):
    ea = expderiv @ a
    gradvecs = [(l==0)*ea + (l>0)*jnp.zeros(n, dtype=jnp.complex128)]
    for k in range(1,numsteps):
        thisvec = (k<l)*jnp.zeros(n, dtype=jnp.complex128)
        thisvec += (k==l)*ea
        thisvec += (k>l)*(matexp[k] @ gradvecs[k-1])
        gradvecs.append( thisvec )
    
    # zeroblock = jnp.zeros((l, n), dtype=jnp.complex128)
    return jnp.stack(gradvecs, axis=0)

def onehessrow(lamb,mu,ed,a,ga):
    part1 = jnp.real(mu @ ed @ a)
    part2 = jnp.real(lamb.T @ ed @ ga.T)
    return part1 + part2

def onematexp(evecs,expevals):
    return evecs @ jnp.diag(expevals) @ evecs.conj().T

manyeigh = vmap(jnp.linalg.eigh)
vfd = vmap(firstderiv, in_axes=(0,0))
vsd = vmap(secondderiv, in_axes=(0,0))
vgradal = vmap(gradal, in_axes=(0,0,0,None))
vohr = vmap(onehessrow)
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
    controlcost = jnp.sum(f**2)
    return controlcost, pen

# first-order adjoint method
def adjgrad(f, alpha, beta):
    manyhams = jnp.expand_dims(h0,0) + jnp.expand_dims(f,(1,2))*jnp.expand_dims(m,0)
    allevals, allevecs = manyeigh(manyhams)
    expevals = jnp.exp(-1j*dt*allevals)
    matexp = vonematexp(allevecs,expevals)
    
    a = jnp.concatenate([jnp.expand_dims(alpha,0), jnp.zeros((numsteps, n), dtype=jnp.complex128)])
    def amatbody(k, am):
        return am.at[k+1].set( matexp[k] @ am[k] )
    
    # forward trajectory
    a = lax.fori_loop(0, numsteps, amatbody, a)
    
    # initialize lambda
    resid = a[-1] - beta
    
    # we are storing "lambda conjugate" throughout this calculation
    alllamb = jnp.concatenate([jnp.expand_dims(rho*resid.conj(),0), jnp.zeros((numsteps, n), dtype=jnp.complex128)])
    def lambbody(i, al):
        k = (numsteps-1) - i
        return al.at[i+1].set( al[i] @ matexp[k] )
    
    # backward trajectory
    alllamb = lax.fori_loop(0, numsteps, lambbody, alllamb)
    alllamb = jnp.flipud(alllamb)
    
    # first critical calculation
    allexpderivs = vfd(allevecs, allevals)
    
    # output gradient we want
    ourgrad = jnp.einsum('ai,aij,aj->a',alllamb[1:],allexpderivs,a[:-1])
    thegrad = f + jnp.real(ourgrad)
    
    return thegrad

# second-order adjoint method
def adjhess(f, alpha, beta):
    manyhams = jnp.expand_dims(h0,0) + jnp.expand_dims(f,(1,2))*jnp.expand_dims(m,0)
    allevals, allevecs = manyeigh(manyhams)
    expevals = jnp.exp(-1j*dt*allevals)
    matexp = vonematexp(allevecs,expevals)

    a = jnp.concatenate([jnp.expand_dims(alpha,0), jnp.zeros((numsteps, n), dtype=jnp.complex128)])
    def amatbody(k, am):
        return am.at[k+1].set( matexp[k] @ am[k] )
    
    # forward trajectory
    a = lax.fori_loop(0, numsteps, amatbody, a)
    
    # initialize lambda
    resid = a[-1] - beta
    
    # we are storing "lambda conjugate" throughout this calculation
    alllamb = jnp.concatenate([jnp.expand_dims(rho*resid.conj(),0), jnp.zeros((numsteps, n), dtype=jnp.complex128)])
    def lambbody(i, al):
        k = (numsteps-1) - i
        return al.at[i+1].set( al[i] @ matexp[k] )
    
    # backward trajectory
    alllamb = lax.fori_loop(0, numsteps, lambbody, alllamb)
    alllamb = jnp.flipud(alllamb)
    
    # first critical calculation
    allexpderivs = vfd(allevecs, allevals)
    
    # compute gradient of a w.r.t. f
    # grada tensor stores the gradient of the n-dimensional vector a[k] with respect to f[l]
    lvec = jnp.arange(numsteps,dtype=jnp.int16)
    grada = vgradal(lvec, allexpderivs, a[:-1], matexp)
    grada = jnp.transpose(grada,(1,0,2))
     
    # create and propagate mu
    # as before, let us store and propagate "mu conjugate"
    allmu0 = rho*grada[numsteps-1,:,:].conj()
    allmu = jnp.concatenate([jnp.expand_dims(allmu0,0),
                             jnp.zeros((numsteps, numsteps, n), dtype=jnp.complex128)])
    allprevmu2 = jnp.flipud(jnp.outer(jnp.ones(n),jnp.eye(numsteps)).T.reshape((numsteps,numsteps,n)))
    def mubody(kk, amu):
        k = (numsteps-1) - kk
        prevmu1 = amu[kk] @ matexp[k]
        prevmu2 = allprevmu2[kk] * jnp.expand_dims(alllamb[k+1].T @ allexpderivs[k],0)
        return amu.at[kk+1].set( prevmu1+prevmu2 )

    # backward trajectory
    allmu = lax.fori_loop(0, numsteps, mubody, allmu)
    allmu = jnp.flipud(allmu)
        
    # second critical calculation
    allexpderivs2 = vsd(allevecs, allevals)
    
    # compute Hessian
    gradapad = jnp.concatenate([jnp.zeros((1,numsteps,n),dtype=jnp.complex128), grada[:-1,:,:]])
    parts12 = vohr(alllamb[1:],allmu[1:],allexpderivs,a[:-1],gradapad)
    part3 = jnp.diag(jnp.real(jnp.einsum('ai,aij,aj->a',alllamb[1:],allexpderivs2,a[:-1])))
    thehess = jnp.eye(numsteps) + parts12 + part3
    
    return thehess

jcost = jit(cost)
jadjgrad = jit(adjgrad)
jadjhess = jit(adjhess)
jstats = jit(stats)

# force JIT compilation
finit = jnp.array(np.random.normal(size=numsteps))
mycost = jcost(finit, thisalpha, thisbeta)
mygrad = jadjgrad(finit, thisalpha, thisbeta)
myhess = jadjhess(finit, thisalpha, thisbeta)

# number of runs
numruns = args.numruns

# time adjgrad
gradtimes = np.zeros(numruns)
for j in range(numruns):
    finit = jnp.array(np.random.normal(size=numsteps))
    start = time.time()
    mygrad = jadjgrad(finit, thisalpha, thisbeta)
    end = time.time()
    gradtimes[j] = end-start

# time adjgrad
hesstimes = np.zeros(numruns)
for j in range(numruns):
    finit = jnp.array(np.random.normal(size=numsteps))
    start = time.time()
    myhess = jadjhess(finit, thisalpha, thisbeta)
    end = time.time()
    hesstimes[j] = end-start

outfname = outpath + 'times_' + str(n) + '_' + str(numsteps) + postfix + '.npz'
np.savez(outfname, gt=gradtimes, ht=hesstimes)
