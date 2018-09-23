# Black-box variational inference, example with linear models

## Motivation

Duvenaud showed in [Black-Box Stochastic Variational Inference in Five Lines of Python](https://www.cs.toronto.edu/~duvenaud/papers/blackbox.pdf) how to make use of the Python module [autograd](https://github.com/HIPS/autograd) to easily code black-box variational inference introduced in [Black Box Variational Inference](http://www.cs.columbia.edu/~blei/papers/RanganathGerrishBlei2014.pdf) by Ranganath et al.

I adapted his code to linear models.

## Dependencies
You will need python 3 with [autograd](https://github.com/HIPS/autograd), [matplotlib](https://matplotlib.org/), and [scipy](https://www.scipy.org/).

## See also
* [Vprop: Variational Inference using RMSprop](http://bayesiandeeplearning.org/2017/papers/50.pdf) by Khan et al.:
As in this paper, I deliberately chose a prior of the form <img src="/tex/b1b094e0f0d44c4c89e75a07526b2776.svg?invert_in_darkmode&sanitize=true" align=middle width=140.00208089999998pt height=24.65753399999998pt/> so that results can be compared to those obtained using the algorithm Vprop.
* [Automatic Variational Inference in Stan](http://www.stat.columbia.edu/~gelman/research/unpublished/bbvb.pdf) by Kucukelbir:
They automated black-box variational inference. What is great is that you can constraint the support of a random variable.  

## Authors
Laurent de Vito

## License
All third-party libraries are subject to their own license.

<a rel="license" href="http://creativecommons.org/licenses/by-nc-nd/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-nd/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-nd/4.0/">Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License</a>.
