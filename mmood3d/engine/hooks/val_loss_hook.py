# SPDX-License-Identifier: AGPL-3.0

import copy

import torch
from mmengine.hooks import Hook
from mmengine.model import is_model_wrapper

from mmdet3d.registry import HOOKS


@HOOKS.register_module()
class ValLossHook(Hook):

    priority = 'NORMAL'

    def __init__(self, set_model_eval: bool = True, batch_size: int = 1):
        self.set_model_eval = set_model_eval
        self.batch_size = batch_size

    def _build_dataloader(self, runner, batch_size):
        val_loader_cfg = copy.deepcopy(runner.cfg.val_dataloader)
        val_loader_cfg.batch_size = batch_size
        val_loader_cfg.dataset.test_mode = False
        val_loader_cfg.dataset.lazy_init = True
        
        test_pipeline = copy.deepcopy(runner.cfg.test_pipeline)
        test_pipeline.insert(0, dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, test_mode=False)) 
        test_pipeline[-1]['keys'] = ['points', 'gt_bboxes_3d', 'gt_labels_3d']
        val_loader_cfg.dataset.pipeline = test_pipeline
        
        diff_rank_seed = runner._randomness_cfg.get(
            'diff_rank_seed', False)
        return runner.build_dataloader(val_loader_cfg, seed=runner.seed, diff_rank_seed=diff_rank_seed)
    
    @torch.no_grad()
    def after_train_epoch(self, runner) -> None:
        model = runner.model
        # TODO: refactor after mmengine using model wrapper
        if is_model_wrapper(model):
            model = model.module

        if self.set_model_eval:
            model.eval()
            model.ood_head.validating = True
        
        loss = 0.0
        
        val_dataloader = self._build_dataloader(runner, self.batch_size)
        for idx, data_batch in enumerate(val_dataloader):
            # same as model.train_step, but does not update params
            data = model.data_preprocessor(data_batch, True)
            losses = model._run_forward(data, mode='loss')
            parsed_losses, log_vars = model.parse_losses(losses)
            loss += parsed_losses
            if self.every_n_inner_iters(idx, 50):
                runner.logger.info(f'Epoch(train-val) [{runner.epoch + 1}][{idx}/{len(val_dataloader)}]    validation_loss: {parsed_losses.item():.4f}')

        loss /= len(val_dataloader)
        loss = loss.item()
        runner.logger.info(f'Epoch(train-val) [{runner.epoch + 1}]    final_validation_loss: {loss:.4f}')

        if self.set_model_eval:
            model.train()
            model.ood_head.validating = False
