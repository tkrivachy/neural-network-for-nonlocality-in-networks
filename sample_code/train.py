import numpy as np
import pickle
import os
import matplotlib.pyplot as plt

import config as cf
from targets import target_distribution_gen_all
from utils_nn import np_distance, np_euclidean_distance
from run_single_gen import single_run, single_evaluation

# Create directories for saving stuff
if __name__ == '__main__':
    for dir in ['saved_models', 'saved_results', 'saved_configs', 'figs_distributions', 'figs_training_loop']:
        if not os.path.exists(dir):
            os.makedirs(dir)
    # Set up the Parameters of the Neural Network (i.e. the config object)
    cf.initialize()

# Loop through parameters. Convention is to start from right, since that is the least noisy distribution if I do a noise scan.
for i in range(cf.pnn.target_distributions.shape[0]-1,-1,-1):

    # Set up new distribution
    cf.pnn.change_p_target(i)
    print('\n\n At round {} of {} (decreasing!), with distribution {} of id {}\n{}'.format(i+1,cf.pnn.target_distributions.shape[0],cf.pnn.target_distr_name, cf.pnn.target_ids[i],cf.pnn.p_target))

    # Run and evaluate model
    model = single_run()
    result = single_evaluation(model)

    # Update results
    cf.pnn.distributions[i,:] = result
    cf.pnn.distances[i] = np_distance(result, cf.pnn.p_target)
    cf.pnn.rms_distances[i] = np_euclidean_distance(result, cf.pnn.p_target)
    print("This distance:", cf.pnn.rms_distances[i])

    # Plot distances
    plt.clf()
    plt.plot(cf.pnn.target_ids,cf.pnn.rms_distances, 'ro')
    if i!=0:
        plt.ylim(bottom=0,top = np.sort(np.unique(cf.pnn.rms_distances))[-2]*1.2)
    else:
        plt.ylim(bottom=0,top = np.sort(np.unique(cf.pnn.rms_distances))[-1]*1.2)
    plt.savefig("./figs_training_loop/loop.png")

    # Plot distributions
    plt.clf()
    plt.plot(cf.pnn.p_target,'ro',markersize=5)
    plt.plot(result,'gs',alpha = 0.85,markersize=5)
    plt.title("Target distr.: {} {:.3f}".format(cf.pnn.target_distr_name, cf.pnn.target_ids[i]))
    plt.ylim(bottom=0,top=max(cf.pnn.p_target)*1.2)
    plt.savefig("./figs_distributions/target_"+str(i).zfill(int(np.ceil(np.log10(cf.pnn.target_distributions.shape[0]))))+".png")
