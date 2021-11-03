#! /usr/bin/env python

""" Top-level script for evaluating models """

import numpy as np
import logging
import sys
import torch
import torch.distributions as D
import configargparse
import copy
import tempfile
import os
import time
import torch.nn as nn
from matplotlib import pyplot as plt

# append pytorch_fid to system variables
sys.path.append("/storage/homefs/ch19g182/anaconda3/lib/python3.8/site-packages/pytorch-fid-master/src/pytorch_fid")

sys.path.append("../")

from evaluation import mcmc, sq_maximum_mean_discrepancy
from datasets import load_simulator, SIMULATORS, IntractableLikelihoodError, DatasetNotAvailableError
from utils import create_filename, create_modelname, sum_except_batch, array_to_image_folder
from architectures import create_model
from architectures.create_model import ALGORITHMS
from torch.utils.data import DataLoader
import torch.distributions as D

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
    parser.add_argument("--datadim", type=int, default=2, help="True data dimensionality (for datasets where that is variable)")
    parser.add_argument("--epsilon", type=float, default=0.3, help="Noise term (for datasets where that is variable)")

    # Model details
    parser.add_argument("--modellatentdim", type=int, default=2, help="Model manifold dimensionality")
    parser.add_argument("--specified", action="store_true", help="Prescribe manifold chart: FOM instead of M-flow")
    parser.add_argument("--outertransform", type=str, default="rq-coupling", help="Scalar base trf. for f: {affine | quadratic | rq}-{coupling | autoregressive}")
    parser.add_argument("--innertransform", type=str, default="affine-autoregressive", help="Scalar base trf. for h: {affine | quadratic | rq}-{coupling | autoregressive}")
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
    # DNF additions
    parser.add_argument("--sig2", type=float, default=0., help="noise magnitude sigma^2")
    parser.add_argument("--v_threshold", type=float, default=3., help="threshold for v component for setting p(x)=0")

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
    # DNF additions
    parser.add_argument("--MAP_steps", type=int, default=0, help="Number of MAP steps to infere z (relevent for PAE on vector data only)")
    parser.add_argument("--only_fid", action="store_true", help="Only evaluating fid")
    parser.add_argument("--only_KS", action="store_true", help="Only calculating KS-stats (latent probs for thin_spiral)")
    
    # Other settings
    parser.add_argument("-c", is_config_file=True, type=str, help="Config file path")
    parser.add_argument("--dir", type=str, default="D:\manifold-flow-public", help="Base directory of repo")
    parser.add_argument("--debug", action="store_true", help="Debug mode (more log output, additional callbacks)")
    parser.add_argument("--skipgeneration", action="store_true", help="Skip generative mode eval")
    parser.add_argument("--skiplikelihood", action="store_true", help="Skip likelihood eval")
    parser.add_argument("--skipood", action="store_true", help="Skip OOD likelihood eval")
    parser.add_argument("--skipinference", action="store_true", help="Skip all inference tasks (likelihood eval and MCMC)")
    parser.add_argument("--skipmcmc", action="store_true", help="Skip MCMC")

    return parser.parse_args()

device = torch.device("cpu")
dtype = torch.float

def integrand_circle(z, model, data_dim, sig2):
    x1 = 3*torch.cos(z)
    x2 = 3*torch.sin(z)
   #  print('x1 shape', x1.shape)
    data = torch.stack([x1,x2],dim=1)
    for i in range(data_dim-2):
        dz =   torch.zeros([z.shape[0],1])
        data = torch.cat([data,dz], dim=1)
    model.eval()
    x_reco, log_prob_, _ = model(data.float(), context=None, mode="dnf")
    #gauss
    log_prob = log_prob_ + (data_dim-1)*np.log(np.sqrt((2*np.pi*sig2)))
    
    density = torch.exp(log_prob) * 3  
    return density

def calculate_KS_stats(args,model,sumulator):
    logger.info("Start calculating KS statistics")
    if args.dataset == 'von_Mises_circle':
        prec = 100 #precision for integrals
        CDF_original = torch.zeros(prec)
        CDF_model = torch.zeros(prec) 
        for k in range(prec):
            b = -np.pi*(prec-1-k)/(prec-1) + np.pi*k/(prec-1)
            z = torch.linspace(-np.pi,b,1000)
            dens = integrand_circle(z,model,args.datadim, args.pieepsilon**2) 
            CDF_model[k] = torch.trapz(dens, z)
            log_prob = torch.tensor(simulator._log_density(z.cpu().numpy()))
            CDF_original[k] = torch.trapz(torch.exp(log_prob),z)
        CDF_diff = torch.abs(CDF_model-CDF_original)
        KS_test = torch.max(CDF_diff).cpu().detach().cpu().numpy() 
        logger.info("KS statistics: %s", KS_test )
        np.save(create_filename("results", "KS_stats", args), KS_test)
        np.save(create_filename("results", "KS_density", args), 
        dens.detach().cpu().numpy())
    elif args.dataset == 'sphere_mixture':     
        n_pts = 60
        
        #theta = torch.linspace(0, np.pi, n_pts+2)
        #dx = theta[1]-theta[0]
        #phi = torch.linspace(0, 2*np.pi, n_pts+1) #[-1]
        #dy = phi[1]-phi[0]
        
        data = simulator.generate_grid(n_pts,mode='latent')
        probs = simulator._density(data)
        
        theta, phi = data[0], data[1]
        dx = np.pi / (n_pts+1)   #for integration
        dy = 2 * np.pi / n_pts      #for integration
        
        grid = np.stack((theta.flatten(), phi.flatten()), axis=1)
        
        c, a = 0, 1
        jacobians = a*(c+a*np.sin(theta))
        
        zz = simulator._transform_z_to_x(grid[:,0],grid[:,1],mode='test')
        zz = torch.tensor(zz)
        zz= zz.to(torch.float) 
        zzk, logprobs = [], []
        with torch.no_grad():
            model.eval()
            for zz_i in zz.split(args.evalbatchsize, dim=0):
                #print('zz_i shape',zz_i.shape)
                zzk_i, logprobs_i, _ = model(zz_i, context=None, mode="dnf")	
                zzk += [zzk_i]
                logprobs += [logprobs_i]
        zzk, logprobs = torch.cat(zzk, 0), torch.cat(logprobs, 0) 
        probs_flow = torch.exp(logprobs+0.5*np.log(2*np.pi*args.sig2) ) 
        probs_flow = probs_flow.view(n_pts,n_pts)       

        density_flow = probs_flow * torch.abs(torch.tensor(jacobians)) 
        
        diff = density_flow - torch.tensor(probs)
        CDF_diff = torch.zeros([n_pts,n_pts])
        for i in range(n_pts):
            for j in range(n_pts):
                summe = torch.sum(diff[0:i,0:j])
                CDF_diff[i,j] = torch.abs(summe)
        
        KS_test = torch.max(CDF_diff)*(dx*dy)
        logger.info("KS statistics: %s", KS_test)   
    elif args.dataset == 'thin_spiral':
        latent_test = simulator.load_latent(train=False,dataset_dir=create_filename("dataset", None, args))
        #np.load(r'D:\manifold-flow-public\experiments\data\samples\thin_spiral\x_test_latent.npy')
        order = np.argsort(latent_test)
        latent_test = latent_test[order] #sort: lowest to highest
        z = np.sqrt(latent_test) * 540 * (2 * np.pi) / 360 
        d1x = - np.cos(z) * z     #d/dz = -cos(z) +sin(z)z    --> ||grad||^2 = cos^2 - cos sin z + sin^2 z^2 +
        d1y =   np.sin(z) * z     #d/dz =  sin(z) +cos(z)z   --->             sin^2 + cos sin z + cos^2 z^2
        x = np.stack([ d1x,  d1y], axis=1) / 3  #        
        x = torch.tensor(x).to(torch.float) 
        logprobs =  []
        # with torch.no_grad():
        model.eval()
        params_ = None
        step = 0
        for x_ in x.split(args.evalbatchsize, dim=0):
            step += 1
            print('step ',step)
            if args.algorithm == "flow":
                x_reco, logprobs_, _ = model(x_, context=params_)
            elif args.algorithm in ["pie", "slice"]:
                x_reco, logprobs_, _ = model(x_, context=params_, mode=args.algorithm if not args.skiplikelihood else "projection")
            elif args.algorithm in ["dnf"]:
                x_reco, logprobs_, _, v_ = model(x_,return_hidden=True, context=params_, mode=args.algorithm if not args.skiplikelihood else "projection")
                criterion = torch.sum(v_**2,dim=1)> args.sig2/args.v_threshold #lorenz_setting_nn: 10
                logprobs_[criterion] = -2**23
            elif args.algorithm in ["pae"]:     
                #########get SIGMA for p(x|z) in PAE, see equation (15) in https://arxiv.org/pdf/2006.05479.pdf#########
                """ #For PAE only """
                x_train = simulator.load_dataset(train=True,numpy=True, dataset_dir=create_filename("dataset", None, args), limit_samplesize=None, joint_score=None)
                val_idx = simulator.load_idx(dataset_dir=create_filename("dataset", None, args))
                x_val = torch.tensor(x_train[0][val_idx,:], dtype=torch.float)
                #First get sigmas
                x_hat_val, _, _, z0 = model(x_val, context=None, mode="logl", return_hidden=True)
                sig = torch.mean(torch.abs(x_hat_val-x_val),dim=0)
                Sigma = torch.diag(sig).clone().detach()
                p_x_z = D.MultivariateNormal(torch.zeros(2), Sigma)
                #"""                   
                #MAP ESTIMATE for z
                x_hat, _, _, z0 = model(x_, context=None, mode="logl", return_hidden=True)
                zz = z0.clone().detach() #.requires_grad_(True)
                #define dummy model
                MAP_ = MAP_model(zz,model)
                optim = torch.optim.Adam(MAP_.parameters(), lr=0.001)
                MAP_.train()
                for kk in range(args.MAP_steps):
                    x_hat = MAP_() 
                    diff = x_hat.clone()-x_.clone()  
                    loss =    - p_x_z.log_prob(diff).mean() -MAP_.log_prob().clone().mean() 
                    loss.backward(retain_graph=True)
                    optim.step()
                    optim.zero_grad()
                    del x_hat, diff
                MAP_.eval() 
                x_hat = MAP_()
                z_star = MAP_.get_MAP().detach()
                x_reco = model.decoder(z_star)
                logprobs_ =  p_x_z.log_prob(x_hat-x_) + MAP_.log_prob()
                del z0, MAP_
            
            else:
                x_reco, logprobs_, _, v_ = model(x_,return_hidden=True, context=params_, mode="mf" if not args.skiplikelihood else "projection")
	
            logprobs += [logprobs_]
        logprobs = torch.cat(logprobs, 0)  
        logger.info("Calculated latent probs, KS not implemented")  
        np.save(create_filename("results", "latent_probs", args), logprobs.detach().cpu().numpy())
    else:
        logger.info("KS not implemented for %s dataset", args.dataset)  

def sample_from_model(args, model, simulator, batchsize=200):
    """ Generate samples from model and store """

    logger.info("Sampling from model")

    x_gen_all = []
    while len(x_gen_all) < args.generate:
        n = min(batchsize, args.generate - len(x_gen_all))

        if simulator.parameter_dim() is None:
            x_gen = model.sample(n=n).detach().cpu().numpy()

        elif args.trueparam is None:  # Sample from prior
            params = simulator.sample_from_prior(n)
            params = torch.tensor(params, dtype=torch.float)
            x_gen = model.sample(n=n, context=params).detach().cpu().numpy()

        else:
            params = simulator.default_parameters(true_param_id=args.trueparam)
            params = np.asarray([params for _ in range(n)])
            params = torch.tensor(params, dtype=torch.float)
            x_gen = model.sample(n=n, context=params).detach().cpu().numpy()

        x_gen_all += list(x_gen)

    x_gen_all = np.array(x_gen_all)
    np.save(create_filename("results", "samples", args), x_gen_all)
    
    logger.info("Creating image grid")
    #generate grid
    if simulator.is_image() and simulator.latent_dim()==2:
        x = torch.linspace(-2, 2, 7)
        xx, yy = torch.meshgrid((x, x))
        grid= torch.stack((xx.flatten(), yy.flatten()), dim=1).float()
        images = []
        t = time.time()    
        for k in range(7):
            for j in range(7):
                i = 7*k
                img_ij = model.sample(u = grid[i+j,:].reshape([1,2]))
                images += [img_ij.detach().cpu().numpy()]
        elapsed = time.time() - t
        logger.info('Time needed to evaluate model samples: %s sec',elapsed)
        np.save(create_filename("results", "grid", args), images)        
    return x_gen_all


def evaluate_model_samples(args, simulator, x_gen):
    """ Evaluate model samples and save results """

    logger.info("Calculating likelihood of generated samples")
    try:
        if simulator.parameter_dim() is None:
            log_likelihood_gen = simulator.log_density(x_gen)
        else:
            params = simulator.default_parameters(true_param_id=args.trueparam)
            params = np.asarray([params for _ in range(args.generate)])
            log_likelihood_gen = simulator.log_density(x_gen, parameters=params)
        log_likelihood_gen[np.isnan(log_likelihood_gen)] = -1.0e-12
        np.save(create_filename("results", "samples_likelihood", args), log_likelihood_gen)
    except IntractableLikelihoodError:
        logger.info("True simulator likelihood is intractable for dataset %s", args.dataset)
    except NotImplementedError:
        logger.info("True simulator likelihood is not implemented for dataset %s", args.dataset)

    logger.info("Calculating distance from manifold of generated samples")
    try:
        distances_gen = simulator.distance_from_manifold(x_gen)
        np.save(create_filename("results", "samples_manifold_distance", args), distances_gen)
    except NotImplementedError:
        logger.info("Cannot calculate distance from manifold for dataset %s", args.dataset)

    if simulator.is_image():
        if calculate_fid_given_paths is None:
            logger.warning("Cannot compute FID score, did not find FID implementation")
            return

        logger.info("Calculating FID score of generated samples")
        # The FID script needs an image folder
        with tempfile.TemporaryDirectory() as gen_dir:
            logger.debug(f"Storing generated images in temporary folder {gen_dir}")
            array_to_image_folder(x_gen, gen_dir)

            true_dir = create_filename("dataset", None, args) + "/test"
            os.makedirs(true_dir, exist_ok=True)
            if not os.path.exists(f"{true_dir}/0.jpg"):
                array_to_image_folder(
                    simulator.load_dataset(train=False, numpy=True, dataset_dir=create_filename("dataset", None, args), true_param_id=args.trueparam)[0], true_dir
                )

            logger.debug("Beginning FID calculation with batchsize 50")
            fid = calculate_fid_given_paths([gen_dir, true_dir], 50, "cuda", 2048)
            logger.info(f"FID = {fid}")

            np.save(create_filename("results", "samples_fid", args), [fid])

def calculate_fid(x_gen,simulator):
        with tempfile.TemporaryDirectory() as gen_dir:
            logger.debug(f"Storing generated images in temporary folder {gen_dir}")
            array_to_image_folder(x_gen, gen_dir)

            true_dir = create_filename("dataset", None, args) + "/test"
            os.makedirs(true_dir, exist_ok=True)
            if not os.path.exists(f"{true_dir}/0.jpg"):
                array_to_image_folder(
                    simulator.load_dataset(train=False, numpy=True, dataset_dir=create_filename("dataset", None, args), true_param_id=args.trueparam)[0], true_dir
                )

            logger.debug("Beginning FID calculation with batchsize 50")
            fid = calculate_fid_given_paths([gen_dir, true_dir], 50, "cuda", 2048)
            logger.info(f"FID = {fid}")

            np.save(create_filename("results", "samples_fid", args), [fid])
    

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
                    x_reco, log_prob_, u_, v_ = model(x_, context=params_, mode=args.algorithm if not args.skiplikelihood else "projection")
                 
                elif args.algorithm in ["dnf"]:
                    x_reco, log_prob_, _, v_ = model(x_,return_hidden=True, context=params_, mode=args.algorithm if not args.skiplikelihood else "projection")
                    criterion = torch.sum(v_**2,dim=1)> args.sig2/args.v_threshold #lorenz_setting_nn: 10
                    log_prob_[criterion] = -2**23   
                    
                elif args.algorithm in ["pae"]:        
                    # get SIGMA for p(x|z) in PAE, see equation (15) in https://arxiv.org/pdf/2006.05479.pdf
                    x_train = simulator.load_dataset(train=True,numpy=True, dataset_dir=create_filename("dataset", None, args), limit_samplesize=None, joint_score=None)
                    try:
                        val_idx = simulator.load_idx(dataset_dir=create_filename("dataset", None, args))
                    except:
                        logger.warning("Could not import validation indices.")                
                    
                    x_val = torch.tensor(x_train[0][val_idx,:], dtype=torch.float)
                    # Get Sigma
                    x_hat_val, _, _, z0 = model(x_val, context=None, mode="logl", return_hidden=True)
                    sig = torch.mean(torch.abs(x_hat_val-x_val),dim=0)
                    Sigma = torch.diag(sig).clone().detach()
                    p_x_z = D.MultivariateNormal(torch.zeros(2), Sigma)
    
                    #MAP ESTIMATE for z
                    x_hat, _, _, z0 = model(x_, context=None, mode="logl", return_hidden=True)
                    zz = z0.clone().detach()
    
                    #define dummy model
                    MAP_ = MAP_model(zz,model)
                    optim = torch.optim.Adam(MAP_.parameters(), lr=0.001)
                    
                    MAP_.train()
                    for kk in range(args.MAP_steps):
                        z = MAP_()
                        x_hat = model.decoder(z)
                        diff = x_hat.clone()-x_.clone()  
                        
                        u, log_det_inner = model.inner_transform(z, full_jacobian=False, context=None)
                        log_prob_z = model.manifold_latent_distribution._log_prob(u, context=None) + log_det_inner
                        
                        loss =    - p_x_z.log_prob(diff).mean() -log_prob_z.mean() #.clone()
                        loss.backward(retain_graph=True)
                        optim.step()
                        optim.zero_grad()
                        del x_hat, diff
                    MAP_.eval() 
                    x_hat = MAP_()
                    z_star = MAP_.get_MAP().detach()
                    x_reco = model.decoder(z_star)
                    log_prob_ =  p_x_z.log_prob(x_hat-x_) + MAP_.log_prob()
                    del z0, MAP_
                else:
                    x_reco, log_prob_, _ = model(x_, context=params_, mode="mf" if not args.skiplikelihood else "projection")

                if not args.skiplikelihood:
                    log_prob.append(log_prob_.detach().cpu().numpy())
                reco_error_.append((sum_except_batch((x_ - x_reco) ** 2) ** 0.5).detach().cpu().numpy())
                x_recos_.append(x_reco.detach().cpu().numpy())

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


class MAP_model(nn.Module):
    """Custom Pytorch model for gradient optimization. Inspired by https://towardsdatascience.com/how-to-use-pytorch-as-a-general-optimizer-a91cbf72a7fb
    """
    def __init__(self,z0,fwd_model):        
        super().__init__()
        self.z = nn.Parameter(z0) 
        self.model = fwd_model
        
    def forward(self):
        return self.model.decoder(self.z)
    
    def get_MAP(self):
        return self.z
    
    def log_prob(self):
        u, log_det_inner = self.model.inner_transform(self.z, full_jacobian=False, context=None)
        log_prob = self.model.manifold_latent_distribution._log_prob(u, context=None) + log_det_inner
        return log_prob
    
        
def evaluate_grid_samples(args, simulator):
    """ Likelihood evaluation """

    logger.info(
        "Evaluating grid for density compariison."
    )
    # Prepare
    x = simulator.generate_grid(100)
    x = x.clone().detach().to(device,dtype)
    grid_loader = DataLoader(
                x,
                batch_size=args.evalbatchsize,
                num_workers=0
            )    
            
    # Evaluate
    log_prob = []
    with torch.autograd.set_detect_anomaly(True):
        count=0
        for x_ in grid_loader:
            count += 1
            t_ = time.time() 
            if args.algorithm == "flow":
                x_reco, log_prob_, _ = model(x_, context=None)
            elif args.algorithm in ["pie", "slice","dnf"]:
                x_reco, log_prob_, _ = model(x_, context=None, mode=args.algorithm if not args.skiplikelihood else "projection")
                
                #criterion = torch.sum(h_orthogonal**2,dim=1)> 0.01/10 #lorenz_setting_nn: 10
                #log_prob[criterion] = -2**23
            
            elif args.algorithm in ["pae"]:        
                # get SIGMA for p(x|z) in PAE, see equation (15) in https://arxiv.org/pdf/2006.05479.pdf
                x_train = simulator.load_dataset(train=True,numpy=True, dataset_dir=create_filename("dataset", None, args), limit_samplesize=None, joint_score=None)
                try:
                    val_idx = simulator.load_idx(dataset_dir=create_filename("dataset", None, args))
                except:
                    logger.warning("Could not import validation indices.")                
                
                x_val = torch.tensor(x_train[0][val_idx,:], dtype=torch.float)
                
                # Get Sigma
                x_hat_val, _, _, z0 = model(x_val, context=None, mode="logl", return_hidden=True)
                sig = torch.mean(torch.abs(x_hat_val-x_val),dim=0)
                Sigma = torch.diag(sig).clone().detach()
                p_x_z = D.MultivariateNormal(torch.zeros(2), Sigma)

                #MAP ESTIMATE for z
                x_hat, _, _, z0 = model(x_, context=None, mode="logl", return_hidden=True)
                zz = z0.clone().detach()

                #define dummy model
                MAP_ = MAP_model(zz,model)
                optim = torch.optim.Adam(MAP_.parameters(), lr=0.001)
                
                MAP_.train()
                for kk in range(args.MAP_steps):
                    z = MAP_()
                    x_hat = model.decoder(z)
                    diff = x_hat.clone()-x_.clone()  
                    
                    u, log_det_inner = model.inner_transform(z, full_jacobian=False, context=None)
                    log_prob_z = model.manifold_latent_distribution._log_prob(u, context=None) + log_det_inner
                    
                    loss =    - p_x_z.log_prob(diff).mean() -log_prob_z.mean() #.clone()
                    loss.backward(retain_graph=True)
                    optim.step()
                    optim.zero_grad()
                    del x_hat, diff
                MAP_.eval() 
                x_hat = MAP_()
                z_star = MAP_.get_MAP().detach()
                x_reco = model.decoder(z_star)
                log_prob_ =  p_x_z.log_prob(x_hat-x_) + MAP_.log_prob()
                del z0, MAP_    
                # MAP_.eval()
                # z_star = MAP_()
                # x_hat = model.decoder(z_star)
                # u, log_det_inner = model.inner_transform(z_star, full_jacobian=False, context=None)
                # log_prob_star = model.manifold_latent_distribution._log_prob(u, context=None) + log_det_inner
                    
                # log_prob_ =  p_x_z.log_prob(x_hat-x_) + log_prob_star
                # del z0, MAP_
            else:
                x_reco, log_prob_, _ = model(x_, context=None, mode="mf" if not args.skiplikelihood else "projection")
       
            if not args.skiplikelihood:
                log_prob.append(log_prob_.detach().cpu().numpy())
                
            elapsed_ = time.time() - t_
            logger.info('Time needed for batch %s: %s sec',count, elapsed_)
            
        if not args.skiplikelihood:
            log_prob = np.concatenate(log_prob, axis=0)
        np.save(create_filename("results", "log_grid_likelihood", args), log_prob)

def interpolation(args, simulator, imgs):
    """ Interpolate between two images """
    logger.info(
        "Creating path in latent space and generating corresponding path in image space."
    )
    # Step1: transform img1, img2 into latent space    
    if args.algorithm == "flow":
        x_reco, log_prob_, u = model(imgs, context=None)
    elif args.algorithm in ["pie", "slice","dnf"]:
        x_reco, log_prob_, u = model(imgs, context=None, mode=args.algorithm if not args.skiplikelihood else "projection")
    elif args.algorithm in ["pae"]:
        x_reco, log_prob_, u = model(imgs, context=None, mode="logl")
    else:
        x_reco, log_prob_, u = model(imgs, context=None, mode="mf" if not args.skiplikelihood else "projection")
    # Step2: create latent path, map into image space, and save the images
    N = 5  
    dt = torch.linspace(0,1,N)   
    u_t = torch.zeros([N,args.modellatentdim])
    for k in range(N):
        u_t[k] = u[0,:]*dt[k] + (1-dt[k])*u[1,:]
    x_reco = model.decode(u=u_t, context=None)    
    np.save(create_filename("results", "image_path", args), x_reco.detach().cpu().numpy())

def run_mcmc(args, simulator, model=None):
    """ MCMC """

    logger.info(
        "Starting MCMC based on %s after %s observed samples, generating %s posterior samples with %s for parameter point number %s",
        "true simulator likelihood" if model is None else "neural likelihood estimate",
        args.observedsamples,
        args.mcmcsamples,
        "slice sampler" if args.slicesampler else "Metropolis-Hastings sampler (step = {})".format(args.mcmcstep),
        args.trueparam,
    )

    # Data
    true_parameters = simulator.default_parameters(true_param_id=args.trueparam)
    x_obs, _ = simulator.load_dataset(
        train=False, numpy=True, dataset_dir=create_filename("dataset", None, args), true_param_id=args.trueparam, joint_score=False, limit_samplesize=args.observedsamples
    )
    x_obs_ = torch.tensor(x_obs, dtype=torch.float)

    if model is None:
        # MCMC based on ground truth likelihood
        def log_posterior(params):
            log_prob = np.sum(simulator.log_density(x_obs, parameters=params))
            log_prob += simulator.evaluate_log_prior(params)
            return float(log_prob)

    else:
        # MCMC based on neural likelihood estimator
        def log_posterior(params):
            params_ = np.broadcast_to(params.reshape((-1, params.shape[-1])), (x_obs.shape[0], params.shape[-1]))
            params_ = torch.tensor(params_, dtype=torch.float)

            if args.algorithm == "flow":
                log_prob = np.sum(model.log_prob(x_obs_, context=params_).detach().cpu().numpy())
            elif args.algorithm in ["pie", "slice"]:
                log_prob = np.sum(model.log_prob(x_obs_, context=params_, mode=args.algorithm).detach().cpu().numpy())
            elif not args.conditionalouter:
                # Slow part of Jacobian drops out in LLR / MCMC acceptance ratio
                log_prob = np.sum(model.log_prob(x_obs_, context=params_, mode="mf-fixed-manifold").detach().cpu().numpy())
            else:
                log_prob = np.sum(model.log_prob(x_obs_, context=params_, mode="mf").detach().cpu().numpy())

            log_prob += simulator.evaluate_log_prior(params)
            return float(log_prob)

    if args.slicesampler:
        logger.debug("Initializing slice sampler")
        sampler = mcmc.SliceSampler(true_parameters, log_posterior, thin=args.thin)
    else:
        logger.debug("Initializing Gaussian Metropolis-Hastings sampler")
        sampler = mcmc.GaussianMetropolis(true_parameters, log_posterior, step=args.mcmcstep, thin=args.thin)

    if args.burnin > 0:
        logger.info("Starting burn in")
        sampler.gen(args.burnin)
    logger.info("Burn in done, starting main chain")
    posterior_samples = sampler.gen(args.mcmcsamples)
    logger.info("MCMC done")

    return posterior_samples


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
    
    # match latent component v(x) with noise magnitude sig2
    if args.sig2 >0:
        args.pieepsilon = np.sqrt(args.sig2)

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
    
    if args.only_fid:
        try:
            model_smpls_path = r'/storage/homefs/ch19g182/Python/dmanifold-flow-public/experiments/data/results/PAE_model_samples.npy'
            x_gen = np.load(model_smpls_path)
        except:
            logger.warning("Could not load model samples, make sure absolute path %s is right.", model_smpls_path)     
        
        calculate_fid(x_gen,simulator)
        logger.info("All done. Have a nice day <3")
        exit()
        
    # Load model
    if not args.truth:
        model = create_model(args, simulator=simulator)
        model.load_state_dict(torch.load(create_filename("model", None, args), map_location=torch.device("cpu")))
        model.eval()
    else:
        model = None
        
    if torch.cuda.is_available() and not args.truth: 
        device = torch.device("cuda")
        dtype = torch.float
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
        model = model.to(device, dtype)

    if args.only_KS:
        calculate_KS_stats(args,model,simulator)    
        logger.info("Have a nice day!")
        exit()

    # Evaluate generative performance
    if args.skipgeneration:
        logger.info("Skipping generative evaluation")
    elif not args.truth:
        #evaluate grid
        if not simulator.is_image():
            t = time.time()  
            evaluate_grid_samples(args, simulator)
            elapsed = time.time() - t
            logger.info('Time needed to evaluate model samples: %s sec',elapsed)
            calculate_KS_stats(args,model,simulator) #calculate KS if possible
        
        if simulator.is_image():
            #image interpolation
            dataset = simulator.load_dataset(train=True, numpy=True, dataset_dir=create_filename("dataset", None, args))
            imgs = torch.tensor(dataset[0][2:4,:],dtype=torch.float)
            interpolation(args, simulator,imgs)
        
        #generate and then evaluate model samples
        x_gen = sample_from_model(args, model, simulator) #generates image grid for images
        evaluate_model_samples(args, simulator, x_gen)  #calculates FID score for images
        
    if args.skipinference:
        logger.info("Skipping all inference tasks. Have a nice day!")
        exit()

    # Evaluate test and ood samples
    if args.truth:
        evaluate_test_samples(args, simulator, model=None, filename="true_{}_test")
        if args.skipood:
            logger.info("Skipping OOD evaluation")
        else:
            try:
                evaluate_test_samples(args, simulator, ood=True, model=None, filename="true_{}_ood")
            except DatasetNotAvailableError:
                logger.info("OOD evaluation not available")
    else:
        evaluate_test_samples(args, simulator, model=model, filename="model_{}_test")
        if args.skipood:
            logger.info("Skipping OOD evaluation")
        else:
            try:
                evaluate_test_samples(args, simulator, model=model, ood=True, filename="model_{}_ood")
            except DatasetNotAvailableError:
                logger.info("OOD evaluation not available")

    # Inference on model parameters
    if args.skipmcmc:
        logger.info("Skipping MCMC")
    elif simulator.parameter_dim() is not None and args.truth:  # Truth MCMC
        try:
            true_posterior_samples = run_mcmc(args, simulator)
            np.save(create_filename("mcmcresults", "posterior_samples", args), true_posterior_samples)
        except IntractableLikelihoodError:
            logger.info("Ground truth likelihood not tractable, skipping MCMC based on true likelihood")
    elif simulator.parameter_dim() is not None and not args.truth:  # Model-based MCMC
        model_posterior_samples = run_mcmc(args, simulator, model)
        np.save(create_filename("mcmcresults", "posterior_samples", args), model_posterior_samples)

        # MMD calculation (only accurate if there is only one chain)
        args_ = copy.deepcopy(args)
        args_.truth = True
        args_.modelname = None
        create_modelname(args_)
        try:
            true_posterior_samples = np.load(create_filename("mcmcresults", "posterior_samples", args_))

            mmd = sq_maximum_mean_discrepancy(model_posterior_samples, true_posterior_samples, scale="ys")
            np.save(create_filename("results", "mmd", args), mmd)
            logger.info("MMD between model and true posterior samples: %s", mmd)
        except FileNotFoundError:
            logger.info("No true posterior data, skipping MMD calculation!")

    logger.info("All done! Have a nice day!")
