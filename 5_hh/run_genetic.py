import bluepyopt as bpopt
import numpy as np

from bluepyopt.parameters import Parameter

class hh_evaluator(bpopt.evaluators.Evaluator):
    def __init__(self, model, summary, obs_stats, labels_sum_stats, stats_std, params):
        self.m = model
        self.s = summary
        self.obs_stats = obs_stats
        self.stats_std = stats_std

        super(hh_evaluator, self).__init__(objectives=labels_sum_stats,params=params)

    def evaluate_with_lists(self, param_list):
        A = param_list

        # simulation
        params = np.asarray(A)
        states = self.m.gen_single(params)

        # summary statistics
        sum_stats = self.s.calc([states])

        # output with z-transformed stats
        diff_stats = np.ndarray.tolist(np.abs((sum_stats - self.obs_stats)/self.stats_std)[0])
        
        # replace NaNs by large number
        return [100 if np.isnan(x) else x for x in diff_stats]


def run_deap(model, bounds, labels_params, summary, obs_stats, labels_sum_stats,
            stats_std, algo='ibea', offspring_size=10, max_ngen=10, seed=None):
    """Runs genetic algorithm

    Parameters
    ----------
    model :
        Model
    bounds :
        Bounds
    labels_params : list of str
        Labels of parameters
    summary :
        Function to compute summary statistics
    obs_stats :
        Observed summary statistics
    labels_sum_stats :
        Labels of summary statistics
    stats_std : array
        Standard deviations of summary statistics on pilot run
    algo: str
        Determines which genetic algorithm is run.
        So far only ibea ('ibea') and deap default ('deap') are implemented
    offspring_size : int
        Offspring size
    max_ngen : int
        Maximum number of generations
    seed : int or None
        If set, randomness in sampling is disabled
    """


    n_params = len(bounds[:,0])

    params = []
    for i in range(n_params):
        params.append(Parameter(labels_params[i], bounds=[bounds[i,0], bounds[i,1]]))

    # choose and run genetic algorithm
    evaluator = hh_evaluator(model, summary, obs_stats, labels_sum_stats, stats_std, params)

    if algo=='ibea':
        opt = bpopt.deapext.optimisations.IBEADEAPOptimisation(evaluator,offspring_size=offspring_size,seed=seed)
    else:
        opt = bpopt.deapext.optimisations.DEAPOptimisation(evaluator,offspring_size=offspring_size,seed=seed)

    final_pop, halloffame, log, hist = opt.run(max_ngen=max_ngen)

    return final_pop, halloffame, log, hist
