import logging
from collections import OrderedDict

import torch
import torch.nn as nn
import os
import model.networks as networks
from model.sr3_modules.diffusion import GLoss
from .base_model import BaseModel
logger = logging.getLogger('base')


class DDPM(BaseModel):
    def __init__(self, opt):
        super(DDPM, self).__init__(opt)
        # define network and load pretrained models
        self.netG = self.set_device(networks.define_G(opt))
        self.schedule_phase = None
        self.ddim_sampling = opt['model'].get('ddim_sampling', False)  # Load from config
        self.ddim_timesteps = opt['model'].get('ddim_timesteps', 200)  # Load from config

        # set loss and load resume state
        self.set_loss()
        self.set_new_noise_schedule(
            opt['model']['beta_schedule']['train'], schedule_phase='train')
        if self.opt['phase'] == 'train':
            self.netG.train()
            # find the parameters to optimize
            if opt['model']['finetune_norm']:
                optim_params = []
                for k, v in self.netG.named_parameters():
                    v.requires_grad = False
                    if k.find('transformer') >= 0:
                        v.requires_grad = True
                        v.data.zero_()
                        optim_params.append(v)
                        logger.info(
                            'Params [{:s}] initialized to 0 and will optimize.'.format(k))
            else:
                optim_params = list(self.netG.parameters())

            self.optG = torch.optim.Adam(
                optim_params, lr=opt['train']["optimizer"]["lr"])
            self.log_dict = OrderedDict()
        self.load_network()
        self.print_network()

    def feed_data(self, data):
        self.data = self.set_device(data)

    def optimize_parameters(self):
        self.optG.zero_grad()
    
        # Forward pass through the network
        output = self.netG(self.data)
        
        # Handle different loss types
        if isinstance(self.netG.module.loss_func, GLoss):
            # For GLoss, the output is already the loss value
            loss = output
        else:
            # For L1/L2 losses, we need to compute the loss
            b, c, h, w = self.data['HR'].shape
            if isinstance(output, tuple):
                # If the network returns a tuple, use the first element as prediction
                pred = output[0]
                loss = output[0].sum() / int(b*c*h*w)
            else:
                # Single output case
                loss = output.sum() / int(b*c*h*w)
        
        # Backward pass
        loss.backward()
        self.optG.step()

        # Log the training loss
        self.log_dict['training/training_loss'] = loss.item()

    def test(self, continous=False, ddim=None, timesteps=None):
        """
        Modified to accept DDIM parameters
        If ddim/timesteps are None, use the instance defaults
        """
        self.netG.eval()
        with torch.no_grad():
            # Use provided parameters or fall back to instance defaults
            use_ddim = self.ddim_sampling if ddim is None else ddim
            use_timesteps = self.ddim_timesteps if timesteps is None else timesteps
            
            if isinstance(self.netG, nn.DataParallel):
                self.SR = self.netG.module.super_resolution(
                    self.data['SR'], 
                    continous=continous,
                    ddim=use_ddim,
                    timesteps=use_timesteps
                )
            else:
                self.SR = self.netG.super_resolution(
                    self.data['SR'], 
                    continous=continous,
                    ddim=use_ddim,
                    timesteps=use_timesteps
                )
        self.netG.train()

    def sample(self, batch_size=1, continous=False, ddim=None, timesteps=None):
        """
        Modified to accept DDIM parameters
        If ddim/timesteps are None, use the instance defaults
        """
        self.netG.eval()
        with torch.no_grad():
            # Use provided parameters or fall back to instance defaults
            use_ddim = self.ddim_sampling if ddim is None else ddim
            use_timesteps = self.ddim_timesteps if timesteps is None else timesteps
            
            if isinstance(self.netG, nn.DataParallel):
                self.SR = self.netG.module.sample(
                    batch_size, 
                    continous=continous,
                    ddim=use_ddim,
                    timesteps=use_timesteps
                )
            else:
                self.SR = self.netG.sample(
                    batch_size, 
                    continous=continous,
                    ddim=use_ddim,
                    timesteps=use_timesteps
                )
        self.netG.train()

    def set_loss(self):
        if isinstance(self.netG, nn.DataParallel):
            self.netG.module.set_loss(self.device)
        else:
            self.netG.set_loss(self.device)

    def set_new_noise_schedule(self, schedule_opt, schedule_phase='train'):
        if self.schedule_phase is None or self.schedule_phase != schedule_phase:
            self.schedule_phase = schedule_phase
            if isinstance(self.netG, nn.DataParallel):
                self.netG.module.set_new_noise_schedule(
                    schedule_opt, self.device)
            else:
                self.netG.set_new_noise_schedule(schedule_opt, self.device)

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self, need_LR=True, sample=False):
        out_dict = OrderedDict()
        if sample:
            out_dict['SAM'] = self.SR.detach().float().cpu()
        else:
            out_dict['SR'] = self.SR.detach().float().cpu()
            out_dict['INF'] = self.data['SR'].detach().float().cpu()
            out_dict['HR'] = self.data['HR'].detach().float().cpu()
            if need_LR and 'LR' in self.data:
                out_dict['LR'] = self.data['LR'].detach().float().cpu()
            else:
                out_dict['LR'] = out_dict['INF']
        return out_dict

    def print_network(self):
        s, n = self.get_network_description(self.netG)
        if isinstance(self.netG, nn.DataParallel):
            net_struc_str = '{} - {}'.format(self.netG.__class__.__name__,
                                             self.netG.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netG.__class__.__name__)

        logger.info(
            'Network G structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
        logger.info(s)

    def save_network(self, epoch, iter_step):
        gen_path = os.path.join(
            self.opt['path']['checkpoint'], 'I{}_E{}_gen.pth'.format(iter_step, epoch))
        opt_path = os.path.join(
            self.opt['path']['checkpoint'], 'I{}_E{}_opt.pth'.format(iter_step, epoch))
        # gen
        network = self.netG
        if isinstance(self.netG, nn.DataParallel):
            network = network.module
        state_dict = network.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        torch.save(state_dict, gen_path)
        # opt
        opt_state = {'epoch': epoch, 'iter': iter_step,
                     'scheduler': None, 'optimizer': None}
        opt_state['optimizer'] = self.optG.state_dict()
        torch.save(opt_state, opt_path)

        logger.info(
            'Saved model in [{:s}] ...'.format(gen_path))


    def load_network(self):
        load_path = self.opt['path']['resume_state']
        if load_path is not None:
            logger.info(
                'Loading pretrained model for G [{:s}] ...'.format(load_path))
            gen_path = '{}_gen.pth'.format(load_path)
            opt_path = '{}_opt.pth'.format(load_path)
            
            # gen
            network = self.netG
            if isinstance(self.netG, nn.DataParallel):
                network = network.module
            
            # Get current state dict
            current_state_dict = network.state_dict()
            
            # Load saved state dict
            saved_state_dict = torch.load(gen_path)
            
            # Handle finetune_norm case
            if self.opt['model']['finetune_norm']:
                # Only load non-transformer parameters
                for name, param in saved_state_dict.items():
                    if name.find('transformer') < 0 and name in current_state_dict:
                        current_state_dict[name].copy_(param)
            else:
                # Handle the case of different loss types
                # We'll load all matching parameters and skip any missing ones
                # (like the GLoss kernels when loading from L1/L2 checkpoint)
                
                # 1. First get the list of parameters to load
                matched_params = {}
                missing_keys = []
                unexpected_keys = []
                
                for name, param in saved_state_dict.items():
                    if name in current_state_dict:
                        if current_state_dict[name].shape == param.shape:
                            matched_params[name] = param
                        else:
                            missing_keys.append(name)
                    else:
                        unexpected_keys.append(name)
                
                # 2. Log any discrepancies
                if missing_keys:
                    logger.info('Missing keys in state_dict (shape mismatch): {}'.format(missing_keys))
                if unexpected_keys:
                    logger.info('Unexpected keys in state_dict: {}'.format(unexpected_keys))
                
                # 3. Load the matched parameters
                current_state_dict.update(matched_params)
                network.load_state_dict(current_state_dict, strict=False)
                
                # 4. Special handling for GLoss kernels if they weren't in the checkpoint
                if hasattr(network, 'loss_func') and isinstance(network.loss_func, GLoss):
                    logger.info('Initializing GLoss with default kernels')
            
            if self.opt['phase'] == 'train':
                # optimizer
                opt = torch.load(opt_path)
                self.optG.load_state_dict(opt['optimizer'])
                self.begin_step = opt['iter']
                self.begin_epoch = opt['epoch']
