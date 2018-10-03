import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

import autograd.numpy as np
import autograd.numpy.random as npr
import autograd.scipy.stats.multivariate_normal as mvn
import autograd.scipy.stats.norm as norm
from autograd import grad
from autograd.misc.optimizers import adam

from scipy.stats import multivariate_normal

def black_box_variational_inference(logprob, D, s2, l, X, y, num_samples, batch_size):
    """Implements http://arxiv.org/abs/1401.0118, and uses the
    local reparameterization trick from http://arxiv.org/abs/1506.02557"""

    def unpack_params(params):
        # Variational dist is a diagonal Gaussian.
        mean, log_std = params[:D], params[D:]
        return mean, log_std

    def gaussian_entropy(log_std):
        return 0.5 * D * (1.0 + np.log(2*np.pi)) + np.sum(log_std)

    rs = npr.RandomState(0)
    def variational_objective(params, t):
        """Provides a stochastic estimate of the variational lower bound."""
        mean, log_std = unpack_params(params)
        samples = rs.randn(num_samples, D) * np.exp(log_std) + mean
        datafit = 0.
        for sample in samples:
            datafit += logprob(sample, s2, l, X, y, batch_size, t)
        datafit /= num_samples
        regularizer = gaussian_entropy(log_std)
        lower_bound = regularizer + datafit
        return -lower_bound

    gradient = grad(variational_objective)

    return variational_objective, gradient, unpack_params

# Function.
def f(w, X):
    return X.dot(w)
        
if __name__ == '__main__':
    
    np.random.seed(1)
    
    # Dimension of the parameter vector.
    D = 2
    assert D > 1
    # Number of observations.
    N = 50
    # True slope and intercept.
    a = -1
    b = 0.3
    # Observation noise.
    s2 = 1e-1
    # Parameter of the Gaussian prior.
    l = 1e-2
    # Noisy observations
    Xtrain = np.c_[ np.ones(N), np.random.uniform(low=-1.0, high=1.0, size=(N,D-1)) ]
    ytrain = f( [a,b], Xtrain) + np.sqrt(s2)*np.random.randn(Xtrain.shape[0])
    # exact log evidence
    # brute force:
    # loge = multivariate_normal.logpdf(ytrain.ravel(), cov=s2*np.eye(N)+1./l*np.matmul(Xtrain,Xtrain.T))
    # but for large datasets, we use the Woodbury formula
    S = l*s2*np.eye(D)+np.matmul(Xtrain.T,Xtrain)
    L = np.linalg.cholesky(S)
    C = np.linalg.solve(L,Xtrain.T)
    inv_cov = 1./s2*(np.eye(N) - np.matmul(C.T,C))    
    logdet_cov = -D*np.log(l)+(N-D)*np.log(s2)+2.*np.sum(np.log(np.diag(L)))
    loge = -N/2.*np.log(2.*np.pi)-0.5*logdet_cov-0.5*np.dot(ytrain,np.matmul(inv_cov,ytrain))
    print("log evidence = {}".format(loge))
    # Joint probabilities.
    def logprob(w, s2, l, X, y, batch_size, t):
        
        N = X.shape[0]
        D = w.shape[0]
        b = float(batch_size)
        indices = np.random.choice(N, batch_size, replace = False)
        Xbatch = X[indices]
        ybatch = y[indices]
        
        def logprior():
            # Prior variance is scaled-down by l.
           return -D/2.*np.log(2*np.pi/l)-D*l/2.*np.sum(np.square(w))
            
        def loglik():
            y_mean = f(w, Xbatch)
            return -b/2.*np.log(2*np.pi*s2)-1./(2.*s2)*np.sum( np.square(ybatch-y_mean) )
        
        return N/b*loglik() + logprior()

    # Build variational objective.
    objective, gradient, unpack_params = \
        black_box_variational_inference(logprob, D, s2, l, Xtrain, ytrain, num_samples=1, batch_size=20)
    
    # Set up figure.
    fig = plt.figure(figsize=(16,8), facecolor='white')
    density_ax = fig.add_subplot(131)
    curve_ax   = fig.add_subplot(132, frameon=True)
    log_ax     = fig.add_subplot(133, frameon=True)
    plt.ion()
    plt.show(block=False)
    
    files = []
    
    def callback(params, t, g):
        lb = -objective(params, t)
        print("Iteration {:05d} lower bound {:.3e}, parameters = {}".format(t, lb, params))
        
        mean, log_std = params[:D], params[D:]
        
        log_ax.scatter(t, lb, color = "red")
        log_ax.hlines(loge, xmin=1, xmax=t, color = "black")
        log_ax.set_title("Lower bounds")
        log_ax.set_xlabel("Time-step")
        density_ax.cla()
        density_ax.scatter( mean[0], mean[1], s = 50, marker = "+", color = "red")
        density_ax.scatter( a, b, s = 50, marker = "o", color = "black")
        e = Ellipse(xy=[mean[0], mean[1]], width=2*np.exp(log_std[0]), height=2*np.exp(log_std[1]), angle=0.)
        density_ax.add_artist(e)
        e.set_alpha(0.3)
        e.set_facecolor("red")
        density_ax.set_xlim([ -3, 3 ])
        density_ax.set_ylim([ -3, 3 ])
        density_ax.set_title("Density plot")
        density_ax.set_xlabel("Intercept")
        density_ax.set_ylabel("Slope")
        curve_ax.cla()
        curve_ax.scatter(Xtrain[:,1], ytrain, s = 30, color = "black")
        X_test = np.array([-3, 3])
        curve_ax.plot(X_test, a+b*X_test, color = "black")
        curve_ax.plot(X_test, mean[0]+mean[1]*X_test, color = "red")
        curve_ax.set_xlim([ -1, 1 ])
        curve_ax.set_ylim([ -2.5, 0.5 ])
        curve_ax.set_title("Functions")
        curve_ax.set_xlabel("x")
        curve_ax.set_ylabel("y")
        plt.tight_layout()
        plt.draw()
        
        # fname = './movie/bbvi_tmp%04d.png' % t
        # plt.savefig(fname)
        plt.pause(1.0/30.0)
    
    print("Optimizing variational parameters...")
    init_mean    = -1 * np.ones(D)
    init_log_std =  1 * np.ones(D)
    init_var_params = np.concatenate([init_mean, init_log_std])
    variational_params = adam(gradient, init_var_params, step_size=0.1, num_iters=300, callback=callback)
    
