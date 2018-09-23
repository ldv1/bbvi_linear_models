import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

import autograd.numpy as np
import autograd.numpy.random as npr
import autograd.scipy.stats.multivariate_normal as mvn
import autograd.scipy.stats.norm as norm
from autograd import grad
from autograd.misc.optimizers import adam

from scipy.stats import multivariate_normal

def black_box_variational_inference(logprob, D, s2, l, X, y, num_samples):
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
            datafit += logprob(sample, s2, l, X, y, t)
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
    # Number of observations.
    N = 20
    # True slope and intercept.
    a = -1
    b = 0.3
    # Observation noise.
    s2 = 1e-1
    # Parameter of the Gaussian prior.
    l = 1e-6
    # Noisy observations
    X = np.random.uniform(low=-1.0, high=1.0, size=(N,1))
    X = np.c_[ np.ones(N), X ]
    y = f( [a,b], X) + np.sqrt(s2)*np.random.randn(X.shape[0])
    # exact log evidence
    loge = multivariate_normal.logpdf(y.ravel(), cov=s2*np.eye(N)+1./l*np.matmul(X,X.T))
    print("log evidence = {}".format(loge))
    
    # Joint probabilities.
    def logprob(w, s2, l, X, y, t):
        
        N = X.shape[0]
        D = w.shape[0]
        
        def logprior():
            # Prior variance is scaled-down by l.
           return -D/2.*np.log(2*np.pi/l)-D*l/2.*np.sum(np.square(w))
            
        def loglik():
            y_mean = f(w, X)
            return -N/2.*np.log(2*np.pi*s2)-1./(2.*s2)*np.sum( np.square(y-y_mean) )
        
        return loglik() + logprior()

    # Build variational objective.
    objective, gradient, unpack_params = \
        black_box_variational_inference(logprob, D, s2, l, X, y, num_samples=1)
    
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
        curve_ax.scatter(X[:,1], y, s = 30, color = "black")
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
    
