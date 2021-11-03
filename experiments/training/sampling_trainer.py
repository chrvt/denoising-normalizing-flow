""" Trainer for datasets where sampling is possible """

import logging
import numpy as np
import torch
from torch import optim, nn
from torch.autograd import grad
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch.nn.utils import clip_grad_norm_

import random

from .trainer import BaseTrainer, logger, NanException, EarlyStoppingException

logger = logging.getLogger(__name__)


class SamplingTrainer(BaseTrainer):
    """ Base trainer class. Any subclass has to implement the forward_pass() function. """

    def train(
        self,
        dataset,
        loss_functions,
        loss_weights=None,
        loss_labels=None,
        epochs=50,
        batch_size=100,
        optimizer=optim.AdamW,
        optimizer_kwargs=None,
        initial_lr=1.0e-3,
        scheduler=optim.lr_scheduler.CosineAnnealingLR,
        scheduler_kwargs=None,
        restart_scheduler=None,
        validation_split=0.25,
        early_stopping=True,
        early_stopping_patience=None,
        clip_gradient=1.0,
        verbose="all",
        parameters=None,
        callbacks=None,
        forward_kwargs=None,
        custom_kwargs=None,
        compute_loss_variance=False,
        seed=None,
        initial_epoch=None,
        sig2 = 0.0,
        noise_type = None,
        simulator = None
    ):

        if initial_epoch is not None and initial_epoch >= epochs:
            logging.info("Initial epoch is larger than epochs, nothing to do in this training phase!")
        elif initial_epoch is not None and initial_epoch <= 0:
            initial_epoch = None

        if loss_labels is None:
            loss_labels = [fn.__name__ for fn in loss_functions]
            
        
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
            
        #logger.debug("Initialising training data")
        #train_loader, val_loader = self.make_dataloader(dataset, validation_split, batch_size)
        ##############################################################
        logger.debug("Setting up optimizer")
        optimizer_kwargs = {} if optimizer_kwargs is None else optimizer_kwargs
        if parameters is None:
            parameters = list(self.model.parameters())
        opt = optimizer(parameters, lr=initial_lr, **optimizer_kwargs)
                
        logger.debug("Setting up LR scheduler")
        if epochs < 2:
            scheduler = None
            logger.info("Deactivating scheduler for only %s epoch", epochs)
        scheduler_kwargs = {} if scheduler_kwargs is None else scheduler_kwargs
        sched = None
        epochs_per_scheduler = restart_scheduler if restart_scheduler is not None else epochs
        if scheduler is not None:
            try:
                sched = scheduler(optimizer=opt, T_max=epochs_per_scheduler, **scheduler_kwargs)
            except:
                sched = scheduler(optimizer=opt, **scheduler_kwargs)
        ##############################################################
         
        early_stopping = early_stopping and (validation_split is not None) and (epochs > 1)
        best_loss, best_model, best_epoch = None, None, None
        if early_stopping and early_stopping_patience is None:
            logger.debug("Using early stopping with infinite patience")
        elif early_stopping:
            logger.debug("Using early stopping with patience %s", early_stopping_patience)
        else:
            logger.debug("No early stopping")

        n_losses = len(loss_labels)
        loss_weights = [1.0] * n_losses if loss_weights is None else loss_weights

        n_epochs_verbose = self._set_verbosity(epochs, verbose)

        logger.debug("Beginning main training loop")
        losses_train, losses_val = [], []

        # Resuming training
        if initial_epoch is None:
            initial_epoch = 0
        else:
            logger.info("Resuming with epoch %s", initial_epoch + 1)
            for _ in range(initial_epoch):
                sched.step()  # Hacky, but last_epoch doesn't work when not saving the optimizer state

        # Initial callbacks
        if callbacks is not None:
            for callback in callbacks:
                callback(-1, self.model, 0.0, 0.0, last_batch=self.last_batch)

        # Loop over epochs
        for i_epoch in range(initial_epoch, epochs):
            logger.debug("Training epoch %s / %s", i_epoch + 1, epochs)

            # LR schedule
            if sched is not None:
                logger.debug("Learning rate: %s", sched.get_last_lr())

            try:
                #logger.info("Sampling new data for epoch.")
                dataset_i = simulator.sample(len(dataset)) 
                #logger.info("dataset shape is %s",dataset_i.shape)
                if i_epoch == 0:
                    train_loader, val_loader = self.make_dataloader(dataset_i, validation_split, batch_size)
                else: 
                    del train_loader
                    train_loader, _ = self.make_dataloader(dataset_i, validation_split, batch_size)
                
                loss_train, loss_val, loss_contributions_train, loss_contributions_val = self.epoch(
                    i_epoch,
                    train_loader,
                    val_loader,
                    opt,
                    loss_functions,
                    loss_weights,
                    clip_gradient,
                    parameters,
                    sig2,
                    noise_type,
                    forward_kwargs=forward_kwargs,
                    custom_kwargs=custom_kwargs,
                    compute_loss_variance=compute_loss_variance,
                )
                losses_train.append(loss_train)
                losses_val.append(loss_val)
            except NanException:
                logger.info("Ending training during epoch %s because NaNs appeared", i_epoch + 1)
                raise

            if early_stopping:
                try:
                    best_loss, best_model, best_epoch = self.check_early_stopping(best_loss, best_model, best_epoch, loss_val, i_epoch, early_stopping_patience)
                except EarlyStoppingException:
                    logger.info("Early stopping: ending training after %s epochs", i_epoch + 1)
                    break

            verbose_epoch = (i_epoch + 1) % n_epochs_verbose == 0
            self.report_epoch(i_epoch, loss_labels, loss_train, loss_val, loss_contributions_train, loss_contributions_val, verbose=verbose_epoch)

            # Callbacks
            if callbacks is not None:
                for callback in callbacks:
                    callback(i_epoch, self.model, loss_train, loss_val, last_batch=self.last_batch)

            # LR scheduler
            if sched is not None:
                sched.step()
                if restart_scheduler is not None and (i_epoch + 1) % restart_scheduler == 0:
                    try:
                        sched = scheduler(optimizer=opt, T_max=epochs_per_scheduler, **scheduler_kwargs)
                    except:
                        sched = scheduler(optimizer=opt, **scheduler_kwargs)

        if early_stopping and len(losses_val) > 0:
            self.wrap_up_early_stopping(best_model, losses_val[-1], best_loss, best_epoch)

        logger.debug("Training finished")

        return np.array(losses_train), np.array(losses_val)

    def epoch(
        self,
        i_epoch,
        train_loader,
        val_loader,
        optimizer,
        loss_functions,
        loss_weights,
        clip_gradient,
        parameters,
        sig2,
        noise_type,
        forward_kwargs=None,
        custom_kwargs=None,
        compute_loss_variance=False,
    ):
        n_losses = len(loss_weights)
        
        self.model.train()
        loss_contributions_train = np.zeros(n_losses)
        loss_train = [] if compute_loss_variance else 0.0
        
        for i_batch, batch_data in enumerate(train_loader):
            if i_batch == 0 and i_epoch == 0:
                self.first_batch(batch_data)
            batch_loss, batch_loss_contributions = self.batch_train(
                batch_data, loss_functions, loss_weights, optimizer, clip_gradient, parameters,sig2,noise_type,i_epoch, forward_kwargs=forward_kwargs, custom_kwargs=custom_kwargs
            )
            if compute_loss_variance:
                loss_train.append(batch_loss)
            else:
                loss_train += batch_loss
            for i, batch_loss_contribution in enumerate(batch_loss_contributions[:n_losses]):
                loss_contributions_train[i] += batch_loss_contribution

            self.report_batch(i_epoch, i_batch, True, batch_data, batch_loss)

        loss_contributions_train /= len(train_loader)
        if compute_loss_variance:
            loss_train = np.array([np.mean(loss_train), np.std(loss_train)])
        else:
            loss_train /= len(train_loader)

        if val_loader is not None:
            self.model.eval()
            loss_contributions_val = np.zeros(n_losses)
            loss_val = [] if compute_loss_variance else 0.0

            for i_batch, batch_data in enumerate(val_loader):
                batch_loss, batch_loss_contributions = self.batch_val(batch_data, loss_functions, loss_weights,sig2,noise_type,i_epoch, forward_kwargs=forward_kwargs, custom_kwargs=custom_kwargs)
                if compute_loss_variance:
                    loss_val.append(batch_loss)
                else:
                    loss_val += batch_loss
                for i, batch_loss_contribution in enumerate(batch_loss_contributions[:n_losses]):
                    loss_contributions_val[i] += batch_loss_contribution

                self.report_batch(i_epoch, i_batch, False, batch_data, batch_loss)

            loss_contributions_val /= len(val_loader)
            if compute_loss_variance:
                loss_val = np.array([np.mean(loss_val), np.std(loss_val)])
            else:
                loss_val /= len(val_loader)

        else:
            loss_contributions_val = None
            loss_val = None

        return loss_train, loss_val, loss_contributions_train, loss_contributions_val

    def partial_epoch(
        self,
        i_epoch,
        train_loader,
        val_loader,
        optimizer,
        loss_functions,
        loss_weights,
        parameters,
        sig2 = None,
        noise_type = None,
        clip_gradient=None,
        i_batch_start_train=0,
        i_batch_start_val=0,
        forward_kwargs=None,
        custom_kwargs=None,
        compute_loss_variance=False,
    ):
        if compute_loss_variance:
            raise NotImplementedError

        n_losses = len(loss_weights)
        assert len(loss_functions) == n_losses, "{} loss functions, but {} weights".format(len(loss_functions), n_losses)

        self.model.train()
        loss_contributions_train = np.zeros(n_losses)
        loss_train = [] if compute_loss_variance else 0.0

        i_batch = i_batch_start_train

        for batch_data in train_loader:
            if i_batch == 0 and i_epoch == 0:
                self.first_batch(batch_data)
            batch_loss, batch_loss_contributions = self.batch_train(
                batch_data, loss_functions, loss_weights, optimizer, clip_gradient, parameters,sig2,noise_type,i_epoch,forward_kwargs=forward_kwargs, custom_kwargs=custom_kwargs
            )
            if compute_loss_variance:
                loss_train.append(batch_loss)
            else:
                loss_train += batch_loss
            for i, batch_loss_contribution in enumerate(batch_loss_contributions[:n_losses]):
                loss_contributions_train[i] += batch_loss_contribution

            self.report_batch(i_epoch, i_batch, True, batch_data, batch_loss)

        i_batch += 1

        loss_contributions_train /= len(train_loader)
        if compute_loss_variance:
            loss_train = np.array([np.mean(loss_train), np.std(loss_train)])
        else:
            loss_train /= len(train_loader)

        if val_loader is not None:
            self.model.eval()
            loss_contributions_val = np.zeros(n_losses)
            loss_val = [] if compute_loss_variance else 0.0

            i_batch = i_batch_start_val

            for batch_data in val_loader:
                batch_loss, batch_loss_contributions = self.batch_val(batch_data, loss_functions, loss_weights, sig2,noise_type, i_epoch,forward_kwargs=forward_kwargs, custom_kwargs=custom_kwargs)
                if compute_loss_variance:
                    loss_val.append(batch_loss)
                else:
                    loss_val += batch_loss
                for i, batch_loss_contribution in enumerate(batch_loss_contributions[:n_losses]):
                    loss_contributions_val[i] += batch_loss_contribution

                self.report_batch(i_epoch, i_batch, False, batch_data, batch_loss)

            i_batch += 1

            loss_contributions_val /= len(val_loader)
            if compute_loss_variance:
                loss_val = np.array([np.mean(loss_val), np.std(loss_val)])
            else:
                loss_val /= len(val_loader)

        else:
            loss_contributions_val = None
            loss_val = None

        return loss_train, loss_val, loss_contributions_train, loss_contributions_val

    def first_batch(self, batch_data):
        pass

    def batch_train(self, batch_data, loss_functions, loss_weights, optimizer, clip_gradient, parameters,sig2,noise_type,i_epoch , forward_kwargs=None, custom_kwargs=None):
        loss_contributions = self.forward_pass(batch_data, loss_functions,sig2,noise_type,i_epoch, forward_kwargs=forward_kwargs, custom_kwargs=custom_kwargs)
        loss = self.sum_losses(loss_contributions, loss_weights)
        self.optimizer_step(optimizer, loss, clip_gradient, parameters)

        loss = loss.item()
        loss_contributions = [contrib.item() for contrib in loss_contributions]
        return loss, loss_contributions

    def batch_val(self, batch_data, loss_functions, loss_weights,sig2,noise_type, i_epoch, forward_kwargs=None, custom_kwargs=None):
        loss_contributions = self.forward_pass(batch_data, loss_functions, None, noise_type, i_epoch, forward_kwargs=forward_kwargs, custom_kwargs=custom_kwargs)
        loss = self.sum_losses(loss_contributions, loss_weights)

        loss = loss.item()
        loss_contributions = [contrib.item() for contrib in loss_contributions]
        return loss, loss_contributions

    def forward_pass(self, batch_data, loss_functions,sig2,noise_type, forward_kwargs=None, custom_kwargs=None):
        """
        Forward pass of the model. Needs to be implemented by any subclass.

        Parameters
        ----------
        batch_data : OrderedDict with str keys and Tensor values
            The data of the minibatch.

        loss_functions : list of function
            Loss functions.

        Returns
        -------
        losses : list of Tensor
            Losses as scalar pyTorch tensors.

        """
        raise NotImplementedError

    def report_batch(self, i_epoch, i_batch, train, batch_data, batch_loss):
        pass


class SamplingForwardTrainer(SamplingTrainer):
    """ Trainer for likelihood-based flow training when the model is not conditional. """

    def first_batch(self, batch_data):
        if self.multi_gpu:
            x, y = batch_data
            if len(x.size()) < 2:
                x = x.view(x.size(0), -1)
            x = x.to(self.device, self.dtype)
            self.model(x[: x.shape[0] // torch.cuda.device_count(), ...])
    
    def add_noise(self,dataset,noise_type,x,sig2):
        if noise_type == 'gaussian':            
            noise = np.sqrt(sig2) * torch.randn(x.shape,device=self.device,requires_grad = False).to(self.device)
        
        elif noise_type == 'true_normal':   
            if dataset == 'thin_spiral':
                norm = torch.norm(x,dim=1).reshape([x.shape[0],1])
                z = 3 * norm
                e_r = x / norm
                R = torch.tensor([[0,-1],[1,0]]).float()
                e_phi = +1*torch.matmul(e_r,R)
                x_norm = (e_r + z * e_phi)/3 
                scale = np.sqrt(sig2) * torch.randn([x.shape[0]])
                noise_ = scale.reshape([x.shape[0],1]) * x_norm  / torch.norm(x_norm,dim=1) 
                noise = torch.matmul(noise_,R)
            elif dataset == 'circle':
                sample_norm = sample.clone()/3
                noise = x
                
        elif noise_type == 'model_nn':
            x_normal = self.model.normal_sampling(x).detach().clone().to(self.device, self.dtype) - x
            norm = torch.norm(x_normal,dim=1).reshape([x.shape[0],1])
            x_normal_norm = (x_normal / norm)
            scale = np.sqrt(sig2) * torch.randn([x_normal.shape[0]])
            noise = scale.reshape([x_normal.shape[0],1]) * x_normal_norm            
         
        elif noise_type == 'R3_nn':
            noise = self.model.normal_sampling(x).detach().clone().to(self.device, self.dtype) - x
            
        return noise


    def forward_pass(self, batch_data, loss_functions, sig2, noise_type, i_epoch, forward_kwargs=None, custom_kwargs=None):
        if forward_kwargs is None:
            forward_kwargs = {}
        #print('batch_data',batch_data.shape)
        x = batch_data
        self._check_for_nans("Training data", x)
        
        if len(x.size()) < 2:
            #logger.info('x size is <2')
            x = x.view(x.size(0), -1)
        x = x.to(self.device, self.dtype)
        #logger.info('First batch coordinate %s',x[0,0,0,0])
        if self.multi_gpu:
            results = nn.parallel.data_parallel(self.model, x, module_kwargs=forward_kwargs)
        else:
            if sig2 is not None:
                noise = self.add_noise('thin_spiral',noise_type,x,sig2)
                x_tilde =  x + noise

            else: x_tilde = x
            results = self.model(x_tilde, **forward_kwargs)
        if len(results) == 4:
            x_reco, log_prob, u, hidden = results
        else:
            x_reco, log_prob, u = results
            hidden = None
        #logger.info('First x_reco %s',x_reco[0,0,0,0])
        
        self._check_for_nans("Reconstructed data", x_reco, fix_until=5)
        if log_prob is not None:
            self._check_for_nans("Log likelihood", log_prob, fix_until=5)
        if x.size(0) >= 15:
            self.last_batch = {
                "x": x.detach().cpu().numpy(),
                "x_reco": x_reco.detach().cpu().numpy(),
                "log_prob": None if log_prob is None else log_prob.detach().cpu().numpy(),
                "u": u.detach().cpu().numpy(),
            }

        losses = [loss_fn(x_reco, x, log_prob, hidden=hidden) for loss_fn in loss_functions]
        self._check_for_nans("Loss", *losses)

        return losses


