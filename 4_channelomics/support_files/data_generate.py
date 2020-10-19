import argparse
import delfi.distribution as dd
import logging
import dill as pickle
import numpy as np
import os
import time
import shutil

from model.ChannelOmni import ChannelOmni
from model.ChannelOmniStats import ChannelOmniStats as ChannelStats
from delfi.generator import Default, MPGenerator
from delfi.summarystats import Identity
from tqdm import tqdm


def args_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--disable_debug",
        dest="disable_debug",
        action="store_true",
        help="Disable debug mode",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=1000,
        help="Number of samples (theta, stats) to generate from the prior.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="default",
        help="Sets the name of the data folder for storage.",
    )
    parser.add_argument(
        "--overwrite",
        dest="overwrite",
        action="store_true",
        help="Overwrite files that exist.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed, if set to zero a seed will be generated randomly.",
    )
    parser.add_argument(
        "--verbose",
        dest="verbose",
        action="store_false",
        help="Verbose output, defaults to True.",
    )
    args = parser.parse_args()

    # seed
    if args.seed == 0:
        args.seed = np.random.randint(2 ** 32 - 1)

    # directory setup
    args.bp = "data/{}".format(args.name)
    if os.path.exists(args.bp) and os.path.isdir(args.bp):
        if args.overwrite:
            shutil.rmtree(args.bp)
            os.makedirs(args.bp)
    else:
        os.makedirs(args.bp)

    return args


def prep_log(args):
    logger = logging.getLogger("generate.py")
    logger.setLevel(logging.DEBUG)

    # create file handler which logs even debug messages
    fh = logging.FileHandler("{}/run.log".format(args.bp))
    fh.setLevel(logging.DEBUG)

    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # create formatter and add it to the handlers
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s]: %(message)s", "%m-%d %H:%M:%S"
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


def main(args):
    log = prep_log(args)

    log.info("Running {}.".format(args.name))
    if args.verbose:
        log.info(args)

    prior_lims = np.array([
        [0, 1],
        [-10., 10.],
        [-120., 120.],
        [0., 2000],
        [0., 0.5],
        [0, 0.05],
        [0., 0.5],
        [0, 0.05]
    ])

    if args.verbose:
        log.info('prior')
        log.info(prior_lims)

    m = ChannelOmni(third_exp_model=False, seed=args.seed)
    p = dd.Uniform(lower=prior_lims[:,0], upper=prior_lims[:,1])
    s = ChannelStats()
    g = Default(model=m, prior=p, summary=s)

    tic = time.time()
    dats = g.gen(args.n_samples)
    toc = time.time()
    log.info('Generation took {}s.'.format(toc-tic))

    np.save('{}/theta.npy'.format(args.bp), dats[0])
    np.save('{}/stats.npy'.format(args.bp), dats[1])


if __name__ == "__main__":
    try:
        args = args_parse()
        main(args)
    except:
        if args.disable_debug:
            pass
        else:
            import traceback, pdb, sys

            traceback.print_exc()
            pdb.post_mortem()
            sys.exit(1)
