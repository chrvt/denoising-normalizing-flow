#! /usr/bin/env python

""" Top-level script for evaluating models """

import numpy as np
import logging
import sys
import torch
import configargparse
import copy
import tempfile
import os
import random

sys.path.append("/storage/homefs/ch19g182/anaconda3/lib/python3.8/site-packages/pytorch-fid-master/src/pytorch_fid")
sys.path.append("../")

from evaluation import mcmc, sq_maximum_mean_discrepancy
from datasets import load_simulator, SIMULATORS, IntractableLikelihoodError, DatasetNotAvailableError
from utils import create_filename, create_modelname, sum_except_batch, array_to_image_folder
from architectures import create_model
from architectures.create_model import ALGORITHMS

logger = logging.getLogger(__name__)

try:
    from fid_score import calculate_fid_given_paths
except:
    logger.warning("Could not import fid_score, make sure that pytorch-fid is in the Python path")
    calculate_fid_given_paths = None


def parse_args():
    """ Parses command line arguments for the evaluation """

    parser = configargparse.ArgumentParser()

    # What what what
    parser.add_argument("--truth", action="store_true", help="Evaluate ground truth rather than learned model")
    parser.add_argument("--modelname", type=str, default=None, help="Model name. Algorithm, latent dimension, dataset, and run are prefixed automatically.")
    parser.add_argument("--algorithm", type=str, default="flow", choices=ALGORITHMS, help="Model: flow (AF), mf (FOM, M-flow), emf (Me-flow), pie (PIE), gamf (M-flow-OT)...")
    parser.add_argument("--dataset", type=str, default="spherical_gaussian", choices=SIMULATORS, help="Dataset: spherical_gaussian, power, lhc, lhc40d, lhc2d, and some others")
    parser.add_argument("-i", type=int, default=0, help="Run number")

    # Dataset details
    parser.add_argument("--truelatentdim", type=int, default=2, help="True manifold dimensionality (for datasets where that is variable)")
    parser.add_argument("--datadim", type=int, default=3, help="True data dimensionality (for datasets where that is variable)")
    parser.add_argument("--epsilon", type=float, default=0.01, help="Noise term (for datasets where that is variable)")

    # Model details
    parser.add_argument("--modellatentdim", type=int, default=2, help="Model manifold dimensionality")
    parser.add_argument("--specified", action="store_true", help="Prescribe manifold chart: FOM instead of M-flow")
    parser.add_argument("--outertransform", type=str, default="rq-coupling", help="Scalar base trf. for f: {affine | quadratic | rq}-{coupling | autoregressive}")
    parser.add_argument("--innertransform", type=str, default="rq-coupling", help="Scalar base trf. for h: {affine | quadratic | rq}-{coupling | autoregressive}")
    parser.add_argument("--lineartransform", type=str, default="permutation", help="Scalar linear trf: linear | permutation")
    parser.add_argument("--outerlayers", type=int, default=5, help="Number of transformations in f (not counting linear transformations)")
    parser.add_argument("--innerlayers", type=int, default=5, help="Number of transformations in h (not counting linear transformations)")
    parser.add_argument("--conditionalouter", action="store_true", help="If dataset is conditional, use this to make f conditional (otherwise only h is conditional)")
    parser.add_argument("--dropout", type=float, default=0.0, help="Use dropout")
    parser.add_argument("--pieepsilon", type=float, default=0.01, help="PIE epsilon term")
    parser.add_argument("--pieclip", type=float, default=None, help="Clip v in p(v), in multiples of epsilon")
    parser.add_argument("--encoderblocks", type=int, default=5, help="Number of blocks in Me-flow / PAE encoder")
    parser.add_argument("--encoderhidden", type=int, default=100, help="Number of hidden units in Me-flow / PAE encoder")
    parser.add_argument("--splinerange", default=3.0, type=float, help="Spline boundaries")
    parser.add_argument("--splinebins", default=8, type=int, help="Number of spline bins")
    parser.add_argument("--levels", type=int, default=3, help="Number of levels in multi-scale architectures for image data (for outer transformation f)")
    parser.add_argument("--actnorm", action="store_true", help="Use actnorm in convolutional architecture")
    parser.add_argument("--batchnorm", action="store_true", help="Use batchnorm in ResNets")
    parser.add_argument("--linlayers", type=int, default=2, help="Number of linear layers before the projection for M-flow and PIE on image data")
    parser.add_argument("--linchannelfactor", type=int, default=2, help="Determines number of channels in linear trfs before the projection for M-flow and PIE on image data")
    parser.add_argument("--intermediatensf", action="store_true", help="Use NSF rather than linear layers before projecting (for M-flows and PIE on image data)")
    parser.add_argument("--decoderblocks", type=int, default=5, help="Number of blocks in PAE encoder")
    parser.add_argument("--decoderhidden", type=int, default=100, help="Number of hidden units in PAE encoder")

    # Evaluation settings
    parser.add_argument("--evaluate", type=int, default=1000, help="Number of test samples to be evaluated")
    parser.add_argument("--generate", type=int, default=10000, help="Number of samples to be generated from model")
    parser.add_argument("--gridresolution", type=int, default=11, help="Grid ressolution (per axis) for likelihood eval")
    parser.add_argument("--observedsamples", type=int, default=20, help="Number of iid samples in synthetic 'observed' set for inference tasks")
    parser.add_argument("--slicesampler", action="store_true", help="Use slice sampler for MCMC")
    parser.add_argument("--mcmcstep", type=float, default=0.15, help="MCMC step size")
    parser.add_argument("--thin", type=int, default=1, help="MCMC thinning")
    parser.add_argument("--mcmcsamples", type=int, default=5000, help="Length of MCMC chain")
    parser.add_argument("--burnin", type=int, default=100, help="MCMC burn in")
    parser.add_argument("--evalbatchsize", type=int, default=100, help="Likelihood eval batch size")
    parser.add_argument("--chain", type=int, default=0, help="MCMC chain")
    parser.add_argument("--trueparam", type=int, default=None, help="Index of true parameter point for inference tasks")

    # Other settings
    parser.add_argument("-c", is_config_file=True, type=str, help="Config file path")
    parser.add_argument("--dir", type=str, default="/scratch/jb6504/manifold-flow", help="Base directory of repo")
    parser.add_argument("--debug", action="store_true", help="Debug mode (more log output, additional callbacks)")
    parser.add_argument("--skipgeneration", action="store_true", help="Skip generative mode eval")
    parser.add_argument("--skiplikelihood", action="store_true", help="Skip likelihood eval")
    parser.add_argument("--skipood", action="store_true", help="Skip OOD likelihood eval")
    parser.add_argument("--skipinference", action="store_true", help="Skip all inference tasks (likelihood eval and MCMC)")
    parser.add_argument("--skipmcmc", action="store_true", help="Skip MCMC")

    return parser.parse_args()


def sample_from_model(args, model, simulator, batchsize=200):
    """ Generate samples from model and store """

    logger.info("Sampling from model")

    x_gen_all = []
    while len(x_gen_all) < args.generate:
        n = min(batchsize, args.generate - len(x_gen_all))

        if simulator.parameter_dim() is None:
            x_gen = model.sample(n=n).detach().numpy()

        elif args.trueparam is None:  # Sample from prior
            params = simulator.sample_from_prior(n)
            params = torch.tensor(params, dtype=torch.float)
            x_gen = model.sample(n=n, context=params).detach().numpy()

        else:
            params = simulator.default_parameters(true_param_id=args.trueparam)
            params = np.asarray([params for _ in range(n)])
            params = torch.tensor(params, dtype=torch.float)
            x_gen = model.sample(n=n, context=params).detach().numpy()

        x_gen_all += list(x_gen)
        
    #generate grid
    x = torch.linspace(-2, 2, 7)
    xx, yy = torch.meshgrid((x, x))
    grid= torch.stack((xx.flatten(), yy.flatten()), dim=1).float()
    images = []
    for k in range(7):
        for j in range(7):
            i = 7*k
            #u_ij = u[0:1,:]
            img_ij = model.sample(u = grid[i+j,:].reshape([1,2]))
            #img_ij = model.module.decode(u_ij.to(device).double())
            images += [img_ij.detach().numpy()]
    np.save(create_filename("results", "grid", args), images)  

    x_gen_all = np.array(x_gen_all)
    return x_gen_all


def evaluate_model_samples(args, simulator, x_gen):
    """ Evaluate model samples and save results """


    if simulator.is_image():
        if calculate_fid_given_paths is None:
            logger.warning("Cannot compute FID score, did not find FID implementation")
            return

        # The FID script needs an image folder
        with tempfile.TemporaryDirectory() as gen_dir:
            logger.debug(f"Storing generated images in temporary folder {gen_dir}")
            array_to_image_folder(x_gen, gen_dir)

            true_dir = create_filename("dataset", None, args) + "/test"
            os.makedirs(os.path.dirname(true_dir), exist_ok=True)
            if not os.path.exists(f"{true_dir}/0.jpg"):
                array_to_image_folder(
                    simulator.load_dataset(train=False, numpy=True, dataset_dir=create_filename("dataset", None, args), true_param_id=args.trueparam)[0], true_dir
                )
            logger.debug("Beginning FID calculation with batchsize 50")
            fid = calculate_fid_given_paths([gen_dir, true_dir], 50, "cuda", 2048)
            logger.info(f"FID = {fid}")
            return fid


def evaluate_test_samples(args, simulator, filename, model=None, ood=False, n_save_reco=100):
    """ Likelihood evaluation """

    logger.info(
        "Evaluating %s samples according to %s, %s likelihood evaluation, saving in %s",
        "the ground truth" if model is None else "a trained model",
        "ood" if ood else "test",
        "with" if not args.skiplikelihood else "without",
        filename,
    )

    # Prepare
    x, _ = simulator.load_dataset(
        train=False, numpy=True, ood=ood, dataset_dir=create_filename("dataset", None, args), true_param_id=args.trueparam, joint_score=False, limit_samplesize=args.evaluate,
    )
    parameter_grid = [None] if simulator.parameter_dim() is None else simulator.eval_parameter_grid(resolution=args.gridresolution)

    log_probs = []
    x_recos = None
    reco_error = None

    # Evaluate
    for i, params in enumerate(parameter_grid):
        logger.debug("Evaluating grid point %s / %s", i + 1, len(parameter_grid))
        if model is None:
            params_ = None if params is None else np.asarray([params for _ in x])
            log_prob = simulator.log_density(x, parameters=params_)

        else:
            log_prob = []
            reco_error_ = []
            x_recos_ = []
            n_batches = (args.evaluate - 1) // args.evalbatchsize + 1
            for j in range(n_batches):
                x_ = torch.tensor(x[j * args.evalbatchsize : (j + 1) * args.evalbatchsize], dtype=torch.float)
                if params is None:
                    params_ = None
                else:
                    params_ = np.asarray([params for _ in x_])
                    params_ = torch.tensor(params_, dtype=torch.float)

                if args.algorithm == "flow":
                    x_reco, log_prob_, _ = model(x_, context=params_)
                elif args.algorithm in ["pie", "slice"]:
                    x_reco, log_prob_, _ = model(x_, context=params_, mode=args.algorithm if not args.skiplikelihood else "projection")
                else:
                    x_reco, log_prob_, _ = model(x_, context=params_, mode="mf" if not args.skiplikelihood else "projection")

                if not args.skiplikelihood:
                    log_prob.append(log_prob_.detach().numpy())
                reco_error_.append((sum_except_batch((x_ - x_reco) ** 2) ** 0.5).detach().numpy())
                x_recos_.append(x_reco.detach().numpy())

            if not args.skiplikelihood:
                log_prob = np.concatenate(log_prob, axis=0)
            if reco_error is None:
                reco_error = np.concatenate(reco_error_, axis=0)
            if x_recos is None:
                x_recos = np.concatenate(x_recos_, axis=0)

        if not args.skiplikelihood:
            log_probs.append(log_prob)
    
    
    # Save results
    if len(log_probs) > 0:
        if simulator.parameter_dim() is None:
            log_probs = log_probs[0]

        np.save(create_filename("results", filename.format("log_likelihood"), args), log_probs)

    if len(x_recos) > 0:
        np.save(create_filename("results", filename.format("x_reco"), args), x_recos[:n_save_reco])

    if reco_error is not None:
        np.save(create_filename("results", filename.format("reco_error"), args), reco_error)

    if parameter_grid is not None:
        np.save(create_filename("results", "parameter_grid_test", args), parameter_grid)
    
    return reco_error

if __name__ == "__main__":
    # Parse args
    args = parse_args()
    logging.basicConfig(format="%(asctime)-5.5s %(name)-20.20s %(levelname)-7.7s %(message)s", datefmt="%H:%M", level=logging.DEBUG if args.debug else logging.INFO)
    # Silence PIL
    for key in logging.Logger.manager.loggerDict:
        if "PIL" in key:
            logging.getLogger(key).setLevel(logging.WARNING)

    logger.info("Hi!")
    logger.debug("Starting evaluate.py with arguments %s", args)

    """
    seed = 1237
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    """
    #########################################
    ##Hyperparameter Grid:
    K = 1 if args.algorithm == 'mf' else 1
    fid_score = np.zeros(K)
    reco = np.zeros(K)
    original_name = args.modelname
    for idx in range(K):
        args.modelname = original_name #+ str(idx) #+'_run' + str(idx)
    
        # Model name
        if args.truth:
            create_modelname(args)
            logger.info("Evaluating simulator truth")
        else:
            create_modelname(args)
            logger.info("Evaluating model %s", args.modelname)
    
        # Bug fix related to some num_workers > 1 and CUDA. Bad things happen otherwise!
        torch.multiprocessing.set_start_method("spawn", force=True)
    
        # Data set
        simulator = load_simulator(args)
    
        # Load model
        if not args.truth:
            model = create_model(args, simulator=simulator)
            model.load_state_dict(torch.load(create_filename("model", None, args), map_location=torch.device("cpu")))
            model.eval()
        else:
            model = None
    
        # Evaluate generative performance
        if args.skipgeneration:
            logger.info("Skipping generative evaluation")
        elif not args.truth:
            x_gen = sample_from_model(args, model, simulator)
            fid = evaluate_model_samples(args, simulator, x_gen)
            fid_score[idx] = fid
            reco_error = evaluate_test_samples(args, simulator, model=model, filename="model_{}_test")
            reco[idx] = reco_error.mean()
            logger.info('Mean Reconstruction error %s',reco[idx])
    np.save(create_filename("results", "samples_fid", args), fid_score)
    ##For DNF  
    #remove best
    idx_best = np.argmin(fid_score)
    idx_best2 = np.argmin(reco)
    idx_worst = np.argmax(fid_score)
    idx_worst2 = np.argmin(reco)
    
    if idx_best == idx_best2 and idx_worst == idx_worst2:
        logger.info('FID and mean recostruction consistent')
    else: logger.info('FID and mean recostruction not consistent')
    
    indices = np.arange(fid_score.size)
    indices = np.delete(indices, idx_best)
    indices = np.delete(indices, idx_worst)
    #remove worst
    logger.info('Mean fid-score %s with sd %s',np.mean(fid_score[indices]),np.std(fid_score[indices]))
    logger.info('Mean Reconstr. error %s with sd %s',np.mean(reco[indices]),np.std(reco[indices]))
    
    if args.algorithm  == 'dnf':
        #x = torch.linspace(111.1,112,10)
        #y = torch.linspace(0.191,0.2,10)
        x = torch.tensor([1.0,100.0,500.0,1000.0,2000.0,5000.0])   
        y = torch.tensor([0.05,0.1,0.15,0.2,0.3])
        xx, yy = torch.meshgrid((x, y))
        zz = torch.stack((xx.flatten(), yy.flatten()), dim=1).numpy()
        idx_star = np.argmin(fid_score)
        logger.info('Optimal combination of hyperparameters: lambda = %s and sig2 = %s with FID-score %s',zz[idx_star][0],zz[idx_star][1],np.min(fid_score))
        #np.save(create_filename("results", filename.format("x_reco"), args), fid_score)
    else: 
   ##For NF
        x = torch.linspace(101,200,100)
        idx_star = np.argmin(fid_score)
        logger.info('Optimal combination of hyperparameter: lambda = %s with FID-score %s ',x[idx_star],np.min(fid_score))
        #np.save(create_filename("results", filename.format("x_reco"), args), fid_score)

