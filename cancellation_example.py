import numpy as np
# import matplotlib.pyplot as plt

import tensorflow as tf

from lestratnet.functions import gaussian_gen
from lestratnet.functions_tf import x_gen, divide_merge_train, \
    integrate_fromnnfun
from lestratnet.loss_experiment import loss_setup

# Notes: The phase space generator outputs the upper and lower
# limits of the space if no number of points is requested

# Cancellation function setup ----------------------
ndim = 7
side = 10


def data_transform(x):
    return tf.convert_to_tensor(x/side/2)


# mu1_rand = 2*np.random.rand(ndim)
# Seven random values used for figure in paper
mu1_rand = np.array([
    1.42872452, 0.83847923, 1.58055441,
    0.30422123, 1.7291153, 1.27053504,
    1.54311618
])
mu1_rand = mu1_rand[:ndim]
sig1 = [0.3]*ndim
# mu2_rand = -2*np.random.rand(ndim)
# Seven random values used for figure in paper
mu2_rand = np.array([
    -1.99715329, -1.46426389, -1.77558595,
    -0.76466156, -1.09619859, -0.62166405,
    -1.69391713
])
mu2_rand = mu2_rand[:ndim]
sig2 = [0.3]*ndim

g1 = gaussian_gen(ndim, mu1_rand, sig1)
g2 = gaussian_gen(ndim, mu2_rand, sig2)
bg = gaussian_gen(ndim, [0.0]*ndim, [side/10]*ndim)
my_x_gen = x_gen(ndim, -side/2, side/2)


# The big cancellation function
def my_cancel(x):
    if not isinstance(x, np.ndarray):
        x = x.numpy()
    return 100*g1(x) - 100*g2(x) + 0.1*bg(x)
# --------------------------------------------------


# Integrand callable
integrand = my_cancel
# Callable for random generator of the domain, generates the points passed to
# the integrand
xgenerator = my_x_gen
# Maximum number of steps to create more divisions (one step != one division)
maxruns = 12
# Approximate maximum of regions created
maxregions = 30
# Due to the stochastic nature of region creation, the actual number may be
# slightly larger

loss = loss_setup(loss='squared_hinge')
activation_out = 'tanh'
model_restart = True
retrains = 4
npts_trainregion = 5000
epochs = 4000
batch_size = int(1e6)
# Size of the first sample of data, used to train the first model
init_size = int(2e5)
# Add points for retraining of wrong predictions
ptsadd_retrains = 1000

# Run the process of dividing/merging -> training -> predicting -> ...
nnfun_tst, mdl_tst, xpool_tst, fpool_tst = divide_merge_train(
    integrand, xgenerator, maxruns, maxregions,
    loss, activation_out, model_restart, retrains, npts_trainregion, epochs,
    init_size=init_size, nntest_size=int(1e7), batch_size=batch_size,
    ptsadd_retrains=ptsadd_retrains, data_transform=data_transform,
    merging=True
)

# Create an auxiliar sample from the points obtained in the previous step
xpool_all = np.concatenate(xpool_tst, axis=0)
fpool_all = np.concatenate(fpool_tst)
# Get an actual integrate including precise estimation of region volumes using
# the last trained network
intres = integrate_fromnnfun(
    integrand, xgenerator, nnfun_tst[-1], 0.5, ntgtmin=1000,
    batch_size=int(1e6), preestpts=int(2e5),
    sample_seed=(xpool_all, fpool_all, None)
)

print("Integration estimate:", intres[0])
print("Integration error:", intres[1])
np.savetxt("integration_sample.csv", np.concatenate(intres[2][0]))

print("Model configuration:")
mdl_tst[-1].summary()

mdl_tst[-1].save_weights('regions_model')
mdl_json = mdl_tst[-1].to_json()
with open("regions_model.json", "w") as json_file:
    json_file.write(mdl_json)

print("Weights for the model used for integration are saved in 'regions_model'")
print("Model configuration saved to 'regions_model.json'")
