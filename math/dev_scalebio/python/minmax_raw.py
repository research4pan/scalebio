#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
All rights reserved.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import numpy as np
import time
import torch
import os
import shutil

from training_utils import evaluate, loss_on_sampler
from training_utils import get_optimizer, update_optimizer_lr
from training_utils import get_lr_scheduler
from utils import logging_stat_dict
from samp_p import SampProb

from transformers import AutoModelForCausalLM, AutoTokenizer
import collections
from accelerate import Accelerator
from torch.distributed.fsdp import FullyShardedDataParallel
from accelerate.logging import get_logger
from peft import get_peft_model, LoraConfig, TaskType


def save_model(accelerator, accelerate_model, tokenizer, save_dir, **kargs):

    accelerator.wait_for_everyone()
    accelerator.print(f"saving model at {save_dir} ...")
    unwrapped_model = accelerator.unwrap_model(accelerate_model)
    unwrapped_model.save_pretrained(
        save_dir,
        is_main_process=accelerator.is_main_process,
        save_function=accelerator.save,
        state_dict=accelerator.get_state_dict(accelerate_model),
        max_shard_size="2GB"
    )
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        tokenizer.save_pretrained(save_dir)
    

def evaluate_and_save(step, valid_loss, model, tokenizer, best_valid_loss_and_step, accelerator, save_dir):

    worse_valid_loss, best_step = best_valid_loss_and_step[0], best_valid_loss_and_step[1]
    
    if save_dir is not None and valid_loss <= worse_valid_loss:
        ## if not exists, create it
        tokenizer=AutoTokenizer.from_pretrained(tokenizer)
        if (not os.path.exists(save_dir)) and accelerator.is_main_process:
            os.makedirs(save_dir)
            tokenizer.save_pretrained(save_dir)

        ## delete old
        if accelerator.is_main_process:
            delete_dir = save_dir + f'/checkpoint-{best_valid_loss_and_step[1]}'
            if os.path.exists(delete_dir):
                accelerator.print(f"delete model at {delete_dir} ...")
                shutil.rmtree(delete_dir)
        
        ## save
        save_new = save_dir + f'/checkpoint-{step}'
        save_model(accelerator, model, tokenizer, save_new)

        ## update validation metrics
        best_valid_loss_and_step = [valid_loss, step]
    
    return best_valid_loss_and_step


class MinmaxConfig():
    """Minmax algorithm configs.

    Exists so that it is easy to separate those configs instead of messing up
    with hyperparameters of other algorithms.
    """
    def __init__(
        self,
        args,
        optimizer_args_dict=None,
        lr_scheduler_args_dict=None,
    ):
        """All required hyperparameters."""
        self.model = args.model
        self.num_partitions=args.num_partitions

        self.sampler = args.sampler

        self.num_outer_iter = args.num_outer_iter
        self.num_inner_iter = args.num_inner_iter
        self.tau = args.tau
        self.init_lr = args.init_lr
        self.init_alpha = args.init_alpha

        self.minmax_init_lr_w = args.minmax_init_lr_w if args.minmax_init_lr_w is not None else args.init_lr
        self.minmax_init_lr_u = args.minmax_init_lr_u if args.minmax_init_lr_u is not None else args.init_lr
        self.minmax_init_lr_lambda = args.minmax_init_lr_lambda if args.minmax_init_lr_lambda is not None else args.init_lr

        self.eval_frequency = args.eval_frequency
        self.validation_model_mode = args.validation_model_mode

        self.args = args
        self.optimizer_args_dict = optimizer_args_dict
        self.lr_scheduler_args_dict = lr_scheduler_args_dict

        # self.fp16=args.fp16
        self.save_dir=args.save_dir

        grad_accumulation_steps = args.global_batch_size // args.micro_batch_size
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        grad_accumulation_steps = grad_accumulation_steps // world_size
        self.gradient_accumulation_steps = grad_accumulation_steps

        if args.minmax_validation_sampler is not None:
            self.validation_sampler = args.minmax_validation_sampler
        else:
            self.validation_sampler = args.sampler
        # fp16, args.g_batch_sz, micro_bsz


class Minmax():
    def __init__(self, config: MinmaxConfig):
        """Initializes the algorithm."""
        self.config = config

    def run(
        self,
        train_loader,
        val_loader_for_train,
        val_loader_for_eval,
        test_loader,
        use_wandb: bool=False,
        accelerator=None
    ):
        """Runs the algorithm."""
        config = self.config

        logger = get_logger('accelerator')
        self.best_valid_loss_and_step=[99999999, -1]
        # Prepares model, two separate set of parameters u, w
        backbone_model = AutoModelForCausalLM.from_pretrained(config.model, attn_implementation="flash_attention_2", torch_dtype=torch.bfloat16)

        model_u = backbone_model.to(accelerator.device)
        model_w = copy.deepcopy(backbone_model).to(accelerator.device)

        if self.config.args.lora:
            lora_config = LoraConfig(
                        r=16,
                        task_type=TaskType.CAUSAL_LM,
                        lora_alpha=32,
                        lora_dropout=0.05
                    )
            model_w = get_peft_model(model_w, lora_config)
            model_u = get_peft_model(model_u, lora_config)

        samp_prob_p=SampProb(num_partitions=self.config.num_partitions).to(accelerator.device)#.to(torch.float16)

        # Defines validation loss (L1) and training loss (L2)
        # L1 = crossentropy_loss(x_validation, y_validation, w)
        # L2 = crossentropy_loss(x_train, y_train, w) + lambda w^2
        def train_loss(model, samp_prob_p, loss_scale_factor, reuse_last_samples=False, id_str=''):
            try:
                return loss_on_sampler(
                    model=model,
                    samp_prob_p=samp_prob_p,
                    accelerator=accelerator,
                    data_loader=train_loader,
                    data_loader_iter=self.train_loader_iter,
                    sampler_type=config.sampler,
                    reuse_last_samples=reuse_last_samples,
                    loss_scale_factor=loss_scale_factor,
                    return_grad=False,
                    zero_grad=False,
                    id_str=id_str,
                    args=config.args,
                )
            except StopIteration:
                self.train_loader_iter = iter(train_loader)
                return loss_on_sampler(
                    model=model,
                    samp_prob_p=samp_prob_p,
                    accelerator=accelerator,
                    data_loader=train_loader,
                    data_loader_iter=self.train_loader_iter,
                    sampler_type=config.sampler,
                    reuse_last_samples=reuse_last_samples,
                    loss_scale_factor=loss_scale_factor,
                    return_grad=False,
                    zero_grad=False,
                    id_str=id_str,
                    args=config.args,
                )

        def val_loss(model, loss_scale_factor, reuse_last_samples=False, id_str=''):
            try:
                return loss_on_sampler(
                    model=model,
                    samp_prob_p=None,
                    accelerator=accelerator,
                    data_loader=val_loader_for_train,
                    data_loader_iter=self.val_loader_for_train_iter,
                    sampler_type=config.validation_sampler,
                    reuse_last_samples=reuse_last_samples,
                    loss_scale_factor=loss_scale_factor,
                    return_grad=False,
                    zero_grad=False,
                    id_str=id_str,
                    mode=config.validation_model_mode,
                    args=config.args,
                )
            except StopIteration:
                self.val_loader_for_train_iter = iter(val_loader_for_train)
                return loss_on_sampler(
                    model=model,
                    samp_prob_p=None,
                    accelerator=accelerator,
                    data_loader=val_loader_for_train,
                    data_loader_iter=self.val_loader_for_train_iter,
                    sampler_type=config.validation_sampler,
                    reuse_last_samples=reuse_last_samples,
                    loss_scale_factor=loss_scale_factor,
                    return_grad=False,
                    zero_grad=False,
                    id_str=id_str,
                    mode=config.validation_model_mode,
                    args=config.args,
                )

        def evaluate_and_logging(model, outer_iter, inner_iter, global_step, start_time):
            if inner_iter < 0 or (inner_iter % config.eval_frequency == 0):
                eval_val_loss, eval_val_acc = evaluate(model, accelerator, val_loader_for_eval, config.args)
                eval_test_loss, eval_test_acc = evaluate(model, accelerator, test_loader, config.args)
                self.best_valid_loss_and_step = evaluate_and_save(global_step, eval_val_loss, model, self.config.args.tokenizer_name, self.best_valid_loss_and_step, accelerator, save_dir=self.config.save_dir)
                if inner_iter < 0:
                    prefix = f'At the end of outer iteration i = {outer_iter}:'
                else:
                    prefix = f'At the beginning of i = {outer_iter}, k = {inner_iter}:'

                suffix = ' (evaluation mode of model)'

                stat_dict = {
                    'validation loss': eval_val_loss,
                    'validation ppl': eval_val_acc,
                    'test loss': eval_test_loss,
                    'test ppl': eval_test_acc,
                    'step': global_step,
                    'time': time.time() - start_time,
                }
                logging_stat_dict(stat_dict, prefix, suffix, use_wandb, accelerator)

        # ========== Main Algorithm ==========
        logger.info("start training")
        num_outer_iter = config.num_outer_iter     # N
        num_inner_iter = config.num_inner_iter     # K
        if num_inner_iter is None:
            num_inner_iter = len(train_loader.dataset) // self.config.args.global_batch_size * self.config.args.epoch
        accelerator.print(f'num_inner_iter: {num_inner_iter}, grad_accu_steps: {self.config.gradient_accumulation_steps}')

        tau = config.tau
        lr = config.init_lr
        alpha = config.init_alpha
        global_step = 0
        start_time = time.time()

        optimizer_args_dict = config.optimizer_args_dict
        optimizer_args_dict['lr'] = lr
        optim_lmd = get_optimizer(samp_prob_p.parameters(), optimizer_args_dict)
        
        optimizer_args_dict['weight_decay'] = 5e-4
        optim_w = get_optimizer(model_w.parameters(), optimizer_args_dict)

        optimizer_args_dict['maximize'] = True
        optim_u = get_optimizer(model_u.parameters(), optimizer_args_dict)

        optim_u.zero_grad()
        optim_w.zero_grad()
        optim_lmd.zero_grad()

        lr_scheduler = get_lr_scheduler(config.lr_scheduler_args_dict)        

        model_w, optim_w, train_loader, val_loader_for_train, val_loader_for_eval, test_loader = accelerator.prepare(model_w, optim_w, train_loader, val_loader_for_train, val_loader_for_eval, test_loader)
        model_u, optim_u=accelerator.prepare(model_u, optim_u)
        # samp_prob_p, optim_lmd=accelerator.prepare(samp_prob_p, optim_lmd)

        # print(model_w.module.dtype, model_u.module.dtype, samp_prob_p.module.p.dtype)
        # Prepares data loader iters used by sampler, we initializes iter only
        # once since this operation is slow
        self.train_loader_iter = iter(train_loader)
        self.val_loader_for_train_iter = iter(val_loader_for_train)
        self.val_loader_for_eval_iter = iter(val_loader_for_eval)
        self.test_loader_iter = iter(test_loader)

        for i in range(num_outer_iter):
            alpha *= tau

            for k in range(num_inner_iter):
                lr = tau**(-i) * lr_scheduler(
                    init_lr=config.init_lr,
                    current_step=global_step,
                    num_steps=num_outer_iter * num_inner_iter,
                )
                logger.info(f'step {global_step}: lr = {lr}')
                # Updates optimizer learning rates
                lr_u = lr / config.init_lr * config.minmax_init_lr_u
                lr_w = lr / config.init_lr * config.minmax_init_lr_w
                lr_lambda = lr / config.init_lr * config.minmax_init_lr_lambda

                update_optimizer_lr(optim_u, lr_u)
                update_optimizer_lr(optim_w, lr_w)
                update_optimizer_lr(optim_lmd, lr_lambda)

                # Computes gradients (remained in the computation graph)
                # For the function
                #   L1(w, lambd) + alpha L2(w, lambd) + (-alpha) * L2(u, lambd)
                loss_u_from_L2=0.
                loss_w_from_L2=0.
                loss_w_from_L1=0.
                p_L_u_L2=0.
                p_L_w_L2=0.
                
                # with torch.profiler.profile(
                #             activities=[
                #                 torch.profiler.ProfilerActivity.CPU,
                #                 torch.profiler.ProfilerActivity.CUDA,
                #             ],
                #             on_trace_ready=torch.profiler.tensorboard_trace_handler('./logs_bf16_full'),
                #             profile_memory=True,
                #             record_shapes=True,
                #             with_stack=True
                # ) as profiler:
                # with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                for accu_step in range(self.config.gradient_accumulation_steps):
                    # st=time.time()
                    loss_u_from_L2_acc, p_L_u_L2_acc = train_loss(
                        model_u,
                        samp_prob_p,
                        loss_scale_factor=-alpha / self.config.gradient_accumulation_steps,
                        id_str='L2(u, lambd)',
                    )
                    loss_w_from_L2_acc, p_L_w_L2_acc = train_loss(
                        model_w,
                        samp_prob_p,
                        reuse_last_samples=True,    # Reuses data samples in L2(u, lambd)
                        loss_scale_factor=alpha / self.config.gradient_accumulation_steps,
                        id_str='L2(w, lambd)',
                    )
                    loss_w_from_L1_acc, _ = val_loss(
                        model_w,
                        loss_scale_factor=1.0,
                        id_str='L1(w, lambd)',
                    )
                    # tol=time.time()-st
                    # print(f'device: {accelerator.device}, time for forward and backward: {tol}')
                    loss_u_from_L2+=accelerator.gather(loss_u_from_L2_acc).detach().cpu().mean()
                    loss_w_from_L2+=accelerator.gather(loss_w_from_L2_acc).detach().cpu().mean()
                    loss_w_from_L1+=accelerator.gather(loss_w_from_L1_acc).detach().cpu().mean()
                    p_L_u_L2+=accelerator.gather(p_L_u_L2_acc).detach()
                    p_L_w_L2+=accelerator.gather(p_L_w_L2_acc).detach()
                            # profiler.step()
                    # print(f'device: {accelerator.device}, p_L_u_L2: {p_L_u_L2}, p_L_w_L2: {p_L_w_L2}')
                p_L_u_L2=p_L_u_L2.reshape(-1, samp_prob_p.prob.shape[0]).mean(dim=0)
                p_L_w_L2=p_L_w_L2.reshape(-1, samp_prob_p.prob.shape[0]).mean(dim=0)
                # Prints losses
                # evaluate_and_logging(
                #     model_w,
                #     outer_iter=i,
                #     inner_iter=k,
                #     global_step=global_step,
                #     start_time=start_time,
                # )

                # print(f'device: {accelerator.device}, samp_p.grad: {samp_prob_p.p.grad}, samp_p.data: {samp_prob_p.p.data}')
                grad=samp_prob_p.p.grad
                grad=accelerator.reduce(grad, 'mean')
                samp_prob_p.p.grad=grad
                partition_map=train_loader.dataset.source_dict
                # with FullyShardedDataParallel.summon_full_params(samp_prob_p, with_grads=True):
                stat_dict = {
                        'L1(w, lambda)': loss_w_from_L1,
                        'L2(u, lambda)': loss_u_from_L2 / -alpha,
                        'L2(w, lambda) - L2(u, lambda)': (loss_w_from_L2 + loss_u_from_L2) / alpha,
                        'step': global_step,
                        'time': time.time() - start_time,
                        'samp prob': collections.OrderedDict(sorted({k: samp_prob_p.prob[v] for k, v in partition_map.items()}.items())),
                        'grad of p': samp_prob_p.p.grad
                    }
                logging_stat_dict(
                    stat_dict,
                    prefix=f'At the beginning of i = {i}, k = {k},',
                    suffix='',
                    use_wandb=use_wandb,
                    accelerator=accelerator
                )

                grad_p = (p_L_w_L2 + p_L_u_L2).detach()
                # with FullyShardedDataParallel.summon_full_params(samp_prob_p, with_grads=True):
                samp_prob=samp_prob_p.prob
                s=samp_prob.reshape(-1, 1)
                jacobian=torch.diagflat(s)-torch.mul(s, s.T)
                jacobian=jacobian.to(accelerator.device)
                grad_p = torch.matmul(grad_p, jacobian)
                grad_p = accelerator.gather(grad_p).cpu()
                # print(samp_prob.shape)
                grad = grad_p.reshape(-1, samp_prob.shape[0])
                fac=grad.mean(0)/samp_prob_p.p.grad.cpu()
                logger.info(f'grad p from loss: {grad.mean(0)}, factor: {fac}')

                # Updates model parameters
                optim_u.step()
                optim_w.step()
                optim_lmd.step()

                optim_u.zero_grad()
                optim_w.zero_grad()
                optim_lmd.zero_grad()
                
                # Updates global step
                global_step += 1

        evaluate_and_logging(
            model_w,
            outer_iter=num_outer_iter-1,
            inner_iter=-1,
            global_step=global_step,
            start_time=start_time,
        )
        # logging.info('Result regularizer parameters: ', model_reg.get_lambd_dict())
