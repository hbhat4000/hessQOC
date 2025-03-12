import os
import argparse
import numpy as np
import pickle
import scipy.optimize as so

parser = argparse.ArgumentParser(prog='compareQOC',
                                 description='Compare QOC with and without Hessians')

parser.add_argument('--mol', required=True, help='name of molecule')
parser.add_argument('--basis', required=True, help='name of basis set')
parser.add_argument('--outpath', required=True, help='output path')
parser.add_argument('--nsteps', type=int, required=True, help='number of time steps to take from 0 to T')
parser.add_argument('--dt', type=float, required=True, help='deltat (size of time step) to use')
parser.add_argument('--numruns', type=int, required=True, help='number of runs')
parser.add_argument('--postfix', required=False, help='string to append to saved files')
parser.add_argument('--gpu', required=False, type=int, help='which GPU to use')
parser.add_argument('--opt', required=False, help='pickled dictionary with all optimization parameters')
parser.add_argument('--arch', required=False, help='pickled dictionary with neural network architecture')

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
if args.postfix:
    postfix = args.postfix
else:
    postfix = ""

# set nsteps
numsteps = args.nsteps
print("numsteps = " + str(numsteps))

# set dt
dt = args.dt
print("dt = " + str(dt))

# set tvec
tvec = dt*jnp.arange(numsteps)

# set optimization parameters
if args.opt:
    opfile = open(args.opt, 'rb')
    optparams = pickle.load(opfile)
    opfile.close()
    print("")
    print("Loading optimization parameters from " + args.opt)
    print("")
    maxiter = optparams['maxiter']
    xtol = optparams['xtol']
    gtol = optparams['gtol']
    rho = optparams['rho']
else:
    maxiter = 10000
    xtol = 1e-10
    gtol = 1e-10
    rho = 1e6
    
# set neural network architecture
if args.arch:
    archfile = open(args.arch, 'rb')
    archparams = pickle.load(archfile)
    archfile.close()
    print("")
    print("Loading neural network architecture from " + args.arch)
    print("")
    layerwidths = archparams['layerwidths']
    activations = archparams['activations']
else:
    layerwidths = [1, 16, 16, 1]
    activations = [jax.nn.softplus, jnp.sin]

nlayers = len(layerwidths)-1
numparams = 0
numweights = 0
for j in range(nlayers):
    numparams += layerwidths[j]*layerwidths[j+1] + layerwidths[j+1]
    numweights += layerwidths[j]*layerwidths[j+1]

print("number of neural network parameters = " + str(numparams))

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
    
    return jnp.stack(gradvecs, axis=0)

def onematexp(evecs,expevals):
    return evecs @ jnp.diag(expevals) @ evecs.conj().T

manyeigh = vmap(jnp.linalg.eigh)
vfd = vmap(firstderiv, in_axes=(0,0))
vsd = vmap(secondderiv, in_axes=(0,0))
vgradal = vmap(gradal, in_axes=(0,0,0,None))
vonematexp = vmap(onematexp)

def fmodelraw(theta, t):
    filt = []
    for j in range(nlayers):
        if j==0:
            si = 0
            ei = layerwidths[0]*layerwidths[1]
        else:
            si += layerwidths[j-1]*layerwidths[j]
            ei += layerwidths[j]*layerwidths[j+1]
        filt.append( theta[si:ei].reshape((layerwidths[j],layerwidths[j+1])) )

    bias = []
    for j in range(nlayers):
        if j==0:
            si += layerwidths[nlayers-1]*layerwidths[nlayers]
            ei += layerwidths[1]
        else:
            si += layerwidths[j]
            ei += layerwidths[j+1]
        bias.append( theta[si:ei] )
    
    f = activations[0]( t * filt[0] + bias[0] )
    for j in range(nlayers-2):
        f = activations[j+1]( f @ filt[j+1] + bias[j+1] )
    
    f = f @ filt[nlayers-1] + bias[nlayers-1]
    return f[0,0]

fmodel = vmap(fmodelraw, in_axes=(None,0))

gradfraw = jacobian(fmodelraw)
gradf = vmap(gradfraw, in_axes=(None,0))

hessfraw = jacobian(gradfraw)
hessf = vmap(hessfraw, in_axes=(None,0))

# XAVIER WEIGHT INITIALIZATION
def xavier():
    params = []
    for i in range(nlayers):
        a = 1.0/np.sqrt(layerwidths[i])
        params.append( np.random.uniform(size=layerwidths[i]*layerwidths[i+1], low=-a, high=a) )
    params.append( np.zeros(numparams-numweights) )
    return np.concatenate(params)

# given initial condition and forcing f, return trajectory a
def propSchro(theta, a0):
    manyhams = jnp.expand_dims(h0,0) + jnp.expand_dims(fmodel(theta, tvec),(1,2))*jnp.expand_dims(m,0)
    allevals, allevecs = manyeigh(manyhams)
    expevals = jnp.exp(-1j*dt*allevals)
    matexp = vonematexp(allevecs,expevals)
    
    a = jnp.concatenate([jnp.expand_dims(a0,0), jnp.zeros((numsteps, n), dtype=jnp.complex128)])
    def amatbody(k, am):
        return am.at[k+1].set( matexp[k] @ am[k] )
    
    # forward trajectory
    a = lax.fori_loop(0, numsteps, amatbody, a)
    return a

# given forcing f, IC a0, FC alpha, return cost
def cost(theta, a0, alpha):
    a = propSchro(theta, a0)
    resid = a[-1] - alpha
    pen = jnp.real(jnp.sum(resid * resid.conj()))
    return 0.5*jnp.sum(fmodel(theta, tvec)**2) + 0.5*rho*pen

# adjoint method
def adjgrad(theta, a0, alpha):
    f = fmodel(theta, tvec)
    gf = gradf(theta, tvec)
    manyhams = jnp.expand_dims(h0,0) + jnp.expand_dims(f,(1,2))*jnp.expand_dims(m,0)
    allevals, allevecs = manyeigh(manyhams)
    expevals = jnp.exp(-1j*dt*allevals)
    matexp = vonematexp(allevecs,expevals)
    
    a = jnp.concatenate([jnp.expand_dims(a0,0), jnp.zeros((numsteps, n), dtype=jnp.complex128)])
    def amatbody(k, am):
        return am.at[k+1].set( matexp[k] @ am[k] )
    
    # forward trajectory
    a = lax.fori_loop(0, numsteps, amatbody, a)
    
    # initialize lambda
    resid = a[-1] - alpha
    
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
    ourgrad = jnp.einsum('ai,aij,al,aj->l',alllamb[1:],allexpderivs,gf,a[:-1])
    thegrad = f @ gf + jnp.real(ourgrad)
    
    return thegrad

# second-order adjoint method
def adjhess(theta, a0, alpha):
    f = fmodel(theta, tvec)
    gf = gradf(theta, tvec)
    hf = hessf(theta, tvec)
    manyhams = jnp.expand_dims(h0,0) + jnp.expand_dims(f,(1,2))*jnp.expand_dims(m,0)
    allevals, allevecs = manyeigh(manyhams)
    expevals = jnp.exp(-1j*dt*allevals)
    matexp = vonematexp(allevecs,expevals)

    a = jnp.concatenate([jnp.expand_dims(a0,0), jnp.zeros((numsteps, n), dtype=jnp.complex128)])
    def amatbody(k, am):
        return am.at[k+1].set( matexp[k] @ am[k] )
    
    # forward trajectory
    a = lax.fori_loop(0, numsteps, amatbody, a)
    
    # initialize lambda
    resid = a[-1] - alpha
    
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
    grada = jnp.einsum('ijk,jl->ilk',grada,gf)
    
    # create and propagate mu
    # as before, let us store and propagate "mu conjugate"
    allmu0 = rho*grada[numsteps-1,:,:].conj()
    allmu = jnp.concatenate([jnp.expand_dims(allmu0,0),
                             jnp.zeros((numsteps, numparams, n), dtype=jnp.complex128)])
    # allprevmu2 = jnp.flipud(jnp.outer(jnp.ones(n),jnp.eye(numsteps)).T.reshape((numsteps,numsteps,n)))
    def mubody(kk, amu):
        k = (numsteps-1) - kk
        prevmu1 = amu[kk] @ matexp[k]
        prevmu2 = jnp.outer(gf[k],alllamb[k+1].T @ allexpderivs[k])
        return amu.at[kk+1].set( prevmu1+prevmu2 )

    # backward trajectory
    allmu = lax.fori_loop(0, numsteps, mubody, allmu)
    allmu = jnp.flipud(allmu)
        
    # second critical calculation
    allexpderivs2 = vsd(allevecs, allevals)
    
    # compute Hessian
    gradapad = jnp.concatenate([jnp.zeros((1,numparams,n),dtype=jnp.complex128), grada[:-1,:,:]])
    # j -> numsteps
    # l -> numparams
    # k -> n
    term1 = jnp.einsum('jlk,jka,jm,ja->lm',allmu[1:],allexpderivs,gf,a[:-1])
    term2a = jnp.einsum('jk,jka,jlm,ja->lm',alllamb[1:],allexpderivs,hf,a[:-1])
    term2b = jnp.einsum('jk,jka,jl,jm,ja->lm',alllamb[1:],allexpderivs2,gf,gf,a[:-1])
    term3 = jnp.einsum('jk,jka,jm,jla->lm',alllamb[1:],allexpderivs,gf,gradapad)
    pcc = term1 + term2a + term2b + term3
    hcc = jnp.einsum('ai,aj->ij',gf,gf) + jnp.einsum('a,aij->ij',f,hf)
    thehess = hcc + jnp.real(pcc)
    
    return thehess

jcost = jit(cost)
jadjgrad = jit(adjgrad)
jadjhess = jit(adjhess)

# force JIT compilation
finit = jnp.array(xavier())
mycost = jcost(finit, thisalpha, thisbeta)
mygrad = jadjgrad(finit, thisalpha, thisbeta)
myhess = jadjhess(finit, thisalpha, thisbeta)

# wrappers
def obj(x):
    jx = jnp.array(x)
    return jcost(jx,thisalpha,thisbeta).item()

def gradobj(x):
    jx = jnp.array(x)
    return np.array(jadjgrad(jx,thisalpha,thisbeta))

def hessobj(x):
    jx = jnp.array(x)
    return np.array(jadjhess(jx,thisalpha,thisbeta))

# run both methods many times and save results
numruns = args.numruns
gradstats = np.zeros((numruns,6))
hessstats = np.zeros((numruns,6))

def stats(xstar, finit):
    out = np.zeros(6)
    out[0] = xstar.nit
    out[1] = xstar.execution_time
    out[2] = obj(xstar.x)
    out[3] = np.linalg.norm(gradobj(xstar.x))
    a = propSchro(jnp.array(xstar.x), thisalpha)
    out[4] = np.linalg.norm(np.array(a[-1] - thisbeta))
    out[5] = obj(xstar.x)/obj(finit)
    return out

bestGobj = np.inf
bestGres = np.zeros(numsteps)
bestHobj = np.inf
bestHres = np.zeros(numsteps)

for run in range(numruns):
    print("Run " + str(run))
    finit = xavier()
    xstargrad = so.minimize(obj, x0=finit, method='trust-constr', jac=gradobj,
                            options={'gtol':gtol,'xtol':xtol,'maxiter':maxiter,'verbose':1})
    if xstargrad.fun < bestGobj:
        bestGobj = xstargrad.fun
        bestGres = xstargrad.x
    
    gradstats[run, :] = stats(xstargrad, finit)
    
    xstarhess = so.minimize(obj, x0=finit, method='trust-constr', jac=gradobj, hess=hessobj,
                            options={'gtol':gtol,'xtol':xtol,'maxiter':maxiter,'verbose':1})
    if xstarhess.fun < bestHobj:
        bestHobj = xstarhess.fun
        bestHres = xstarhess.x
    
    hessstats[run, :] = stats(xstarhess, finit)

fname = outpath + "compareneural_" + mol + "_" + basis + postfix + ".npz"
np.savez(fname, gradstats=gradstats, hessstats=hessstats, bestGres=bestGres, bestHres=bestHres)
