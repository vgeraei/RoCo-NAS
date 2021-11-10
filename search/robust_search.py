import sys, os
# update your projecty root path before running
sys.path.insert(0, 'C:\\Users\\Sima\\codes\\NSGA\\nsga-net')


import time
import logging
import argparse
import pickle
import errno
from misc import utils

import numpy as np
from search import train_search
from search import micro_encoding
from search import macro_encoding
from search import nsganet as engine

from pymop.problem import Problem
from pymoo.optimize import minimize

save_location = ''
gen_number = 1
adversarial_examples = 0


# ---------------------------------------------------------------------------------------------------------
# Save function for results
# ---------------------------------------------------------------------------------------------------------
def save_results(payload, folder_name, file_name):
    file_location = save_location + '/' + folder_name + '/' + file_name + '.pckl'

    if not os.path.exists(os.path.dirname(file_location)):
        try:
            os.makedirs(os.path.dirname(file_location))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

    f = open(file_location, 'wb')
    pickle.dump(payload, f)
    f.close()


# ---------------------------------------------------------------------------------------------------------
# Define your NAS Problem
# ---------------------------------------------------------------------------------------------------------
class NAS(Problem):
    # first define the NAS problem (inherit from pymop)
    def __init__(self, search_space='micro', n_var=20, n_obj=1, n_constr=0, lb=None, ub=None,
                 init_channels=24, layers=8, epochs=25, save_dir=None):
        super().__init__(n_var=n_var, n_obj=n_obj, n_constr=n_constr, type_var=np.int)
        self.xl = lb
        self.xu = ub
        self._search_space = search_space
        self._init_channels = init_channels
        self._layers = layers
        self._epochs = epochs
        self._save_dir = save_dir
        self._n_evaluated = 0  # keep track of how many architectures are sampled

    def _evaluate(self, x, out, *args, **kwargs):

        objs = np.full((x.shape[0], self.n_obj), np.nan)

        # Save genomes for each generation
        global gen_number
        file_name = 'gen_' + str(gen_number)
        save_results(payload=x, folder_name='genomes', file_name=file_name)
        gen_number = gen_number + 1

        # Evaluation process
        for i in range(x.shape[0]):
            arch_id = self._n_evaluated + 1
            print('\n')
            logging.info('Network id = {}'.format(arch_id))

            # call back-propagation training
            if self._search_space == 'micro':
                genome = micro_encoding.convert(x[i, :])
            elif self._search_space == 'macro':
                genome = macro_encoding.convert(x[i, :])
            performance = train_search.main(genome=genome,
                                            search_space=self._search_space,
                                            init_channels=self._init_channels,
                                            layers=self._layers, cutout=False,
                                            epochs=self._epochs,
                                            save='arch_{}'.format(arch_id),
                                            expr_root=self._save_dir)


            # all objectives assume to be MINIMIZED !!!!!
            objs[i, 0] = 200 - performance['valid_acc'] - performance['adversarial_acc']
            objs[i, 1] = performance['flops']
            # objs[i, 2] = 100 - performance['adversarial_acc']

            self._n_evaluated += 1

        out["F"] = objs
        # if your NAS problem has constraints, use the following line to set constraints
        # out["G"] = np.column_stack([g1, g2, g3, g4, g5, g6]) in case 6 constraints


# ---------------------------------------------------------------------------------------------------------
# Define what statistics to print or save for each generation
# ---------------------------------------------------------------------------------------------------------
def do_every_generations(algorithm):

    # this function will be call every generation
    # it has access to the whole algorithm class
    gen = algorithm.n_gen
    pop_var = algorithm.pop.get("X")
    pop_obj = algorithm.pop.get("F")

    # Saving the population object
    file_name = 'gen_' + str(gen)
    save_results(payload=algorithm.pop, folder_name='pops', file_name=file_name)


    # report generation info to files
    logging.info("generation = {}".format(gen))
    logging.info("population error: best = {}, mean = {}, "
                 "median = {}, worst = {}".format(np.min(pop_obj[:, 0]), np.mean(pop_obj[:, 0]),
                                                  np.median(pop_obj[:, 0]), np.max(pop_obj[:, 0])))
    logging.info("population complexity: best = {}, mean = {}, "
                 "median = {}, worst = {}".format(np.min(pop_obj[:, 1]), np.mean(pop_obj[:, 1]),
                                                  np.median(pop_obj[:, 1]), np.max(pop_obj[:, 1])))


def main():
    parser = argparse.ArgumentParser("Multi-objetive Genetic Algorithm for NAS")
    parser.add_argument('--save', type=str, default='Robustness', help='experiment name')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--search_space', type=str, default='micro', help='macro or micro search space')
    # arguments for micro search space
    parser.add_argument('--n_blocks', type=int, default=5, help='number of blocks in a cell')
    parser.add_argument('--n_ops', type=int, default=9, help='number of operations considered')
    parser.add_argument('--n_cells', type=int, default=2, help='number of cells to search')
    # arguments for macro search space
    parser.add_argument('--n_nodes', type=int, default=6, help='number of nodes per phases')
    # hyper-parameters for algorithm
    parser.add_argument('--pop_size', type=int, default=40, help='population size of networks')
    parser.add_argument('--n_gens', type=int, default=20, help='number of generations')
    parser.add_argument('--n_offspring', type=int, default=40, help='number of offspring created per generation')
    # arguments for back-propagation training during search
    parser.add_argument('--init_channels', type=int, default=16, help='# of filters for first cell')
    parser.add_argument('--layers', type=int, default=11, help='equivalent with N = 3')
    parser.add_argument('--epochs', type=int, default=25, help='# of epochs to train during architecture search')

    args = parser.parse_args()

    args.save = 'search-{}-{}-{}'.format(args.save, args.search_space, time.strftime("%Y%m%d-%H%M%S"))

    global save_location
    save_location = args.save
    utils.create_exp_dir(args.save)

    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

    pop_hist = []  # keep track of every evaluated architecture



    np.random.seed(args.seed)
    logging.info("args = %s", args)

    # setup NAS search problem
    if args.search_space == 'micro':  # NASNet search space
        n_var = int(4 * args.n_blocks * 2)
        lb = np.zeros(n_var)
        ub = np.ones(n_var)
        h = 1
        for b in range(0, n_var//2, 4):
            ub[b] = args.n_ops - 1
            ub[b + 1] = h
            ub[b + 2] = args.n_ops - 1
            ub[b + 3] = h
            h += 1
        ub[n_var//2:] = ub[:n_var//2]
    elif args.search_space == 'macro':  # modified GeneticCNN search space
        n_var = int(((args.n_nodes-1)*args.n_nodes/2 + 1)*3)
        lb = np.zeros(n_var)
        ub = np.ones(n_var)
    else:
        raise NameError('Unknown search space type')

    problem = NAS(n_var=n_var, search_space=args.search_space,
                  n_obj=2, n_constr=0, lb=lb, ub=ub,
                  init_channels=args.init_channels, layers=args.layers,
                  epochs=args.epochs, save_dir=args.save)

    # configure the nsga-net method
    method = engine.nsganet(pop_size=args.pop_size,
                            n_offsprings=args.n_offspring,
                            eliminatsampe_duplicates=True)

    res = minimize(problem,
                   method,
                   callback=do_every_generations,
                   termination=('n_gen', args.n_gens))

    return


if __name__ == "__main__":
    main()