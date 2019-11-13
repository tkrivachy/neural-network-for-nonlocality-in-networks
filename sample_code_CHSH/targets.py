import numpy as np
from itertools import product

from scipy.io import loadmat

def target_distribution_gen_all(name, param_range, which_param, other_param):
    """ Generate a set of target distributions by varying one parameter. which_param sets whether distr. param or noise param."""
    if which_param == 1:
        p_target_shapeholder = target_distribution_gen(name, param_range[0], other_param);
    elif which_param == 2:
        p_target_shapeholder = target_distribution_gen(name, other_param, param_range[0]);
    target_distributions = np.ones(param_range.shape + p_target_shapeholder.shape) / (p_target_shapeholder.shape[0])
    for i in range(len(param_range)):
        if which_param == 1:
            p_target = target_distribution_gen(name, param_range[i], other_param);
        elif which_param == 2:
            p_target = target_distribution_gen(name, other_param, param_range[i]);
        target_distributions[i,:] = p_target
    return target_distributions

def target_distribution_gen(name, parameter1, parameter2):
    """ parameter1 is usually a parameter of distribution (not always relevant). parameter2 is usually noise."""
    if name=="CHSH":
        v = parameter2
        p = np.array([
        (-2 + np.sqrt(2) + (-1 + np.sqrt(2))*v)/(16.*(-2 + np.sqrt(2))),(2 + np.sqrt(2) + v + np.sqrt(2)*v)/(32 + 16*np.sqrt(2)),(-2 + np.sqrt(2) + v - np.sqrt(2)*v)/(16.*(-2 + np.sqrt(2))),
        (2 + np.sqrt(2) - (1 + np.sqrt(2))*v)/(16.*(2 + np.sqrt(2))),(2 + np.sqrt(2) + v + np.sqrt(2)*v)/(32 + 16*np.sqrt(2)),(-2 + np.sqrt(2) + (-1 + np.sqrt(2))*v)/(16.*(-2 + np.sqrt(2))),(2 + np.sqrt(2) - (1 + np.sqrt(2))*v)/(16.*(2 + np.sqrt(2))),
        (-2 + np.sqrt(2) + v - np.sqrt(2)*v)/(16.*(-2 + np.sqrt(2))),(-2 + np.sqrt(2) + (-1 + np.sqrt(2))*v)/(16.*(-2 + np.sqrt(2))),(2 + np.sqrt(2) + v + np.sqrt(2)*v)/(32 + 16*np.sqrt(2)),(-2 + np.sqrt(2) + v - np.sqrt(2)*v)/(16.*(-2 + np.sqrt(2))),
        (2 + np.sqrt(2) - (1 + np.sqrt(2))*v)/(16.*(2 + np.sqrt(2))),(2 + np.sqrt(2) - (1 + np.sqrt(2))*v)/(16.*(2 + np.sqrt(2))),(-2 + np.sqrt(2) + v - np.sqrt(2)*v)/(16.*(-2 + np.sqrt(2))),(2 + np.sqrt(2) + v + np.sqrt(2)*v)/(32 + 16*np.sqrt(2)),
        (-2 + np.sqrt(2) + (-1 + np.sqrt(2))*v)/(16.*(-2 + np.sqrt(2)))
        ])

    assert (np.abs(np.sum(p)-1.0) < (1E-6)),"Improperly normalized p!"
    return p
