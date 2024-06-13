
"""

CODE ADAPTED FROM: https://github.com/GraphPKU/BREC/

"""
import logging
import time

import numpy as np
import torch
from torch.nn import CosineEmbeddingLoss
from torch_geometric.graphgym.checkpoint import load_ckpt, save_ckpt, clean_ckpt
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.loader import load_dataset
from torch_geometric.graphgym.loss import compute_loss
from torch_geometric.graphgym.model_builder import create_model
from torch_geometric.graphgym.optim import create_optimizer, \
    create_scheduler, OptimizerConfig
from torch_geometric.graphgym.register import register_train
from torch_geometric.graphgym.utils.epoch import is_eval_epoch, is_ckpt_epoch
from torch_geometric.graphgym.utils.comp_budget import params_count
from torch_geometric.loader import DataLoader

from graphgps.optimizer.extra_optimizers import ExtendedSchedulerConfig
from graphgps.loss.subtoken_prediction_loss import subtoken_cross_entropy
from graphgps.utils import cfg_to_dict, flatten_dict, make_wandb_name

# part_dict: {graph generation type, range}
part_dict = {
    "Basic": (0, 60),
    "Regular": (60, 160),
    "Extension": (160, 260),
    "CFI": (260, 360),
    "4-Vertex_Condition": (360, 380),
    "Distance_Regular": (380, 400),
}
# used for indexing the BREC dataset
MARGIN = 0.0
BATCH_SIZE = 16
NUM_RELABEL = 32
SAMPLE_NUM = 400
EPSILON_CMP = 1e-6
THRESHOLD = 72.34
EPOCH = 20
LOSS_THRESHOLD = 0.2
OUTPUT_DIM = 16
EPSILON_MATRIX = 1e-8


# TODO: add option for larger output dimension

def new_optimizer_config(cfg):
    return OptimizerConfig(
        optimizer=cfg.optim.optimizer,
        base_lr=cfg.optim.base_lr,
        weight_decay=cfg.optim.weight_decay,
        momentum=cfg.optim.momentum
    )

def new_scheduler_config(cfg):
    return ExtendedSchedulerConfig(
        scheduler=cfg.optim.scheduler,
        steps=cfg.optim.steps, lr_decay=cfg.optim.lr_decay,
        max_epoch=cfg.optim.max_epoch, reduce_factor=cfg.optim.reduce_factor,
        schedule_patience=cfg.optim.schedule_patience, min_lr=cfg.optim.min_lr,
        num_warmup_epochs=cfg.optim.num_warmup_epochs,
        train_mode=cfg.train.mode, eval_period=cfg.train.eval_period
    )

def cov(X):
    D = X.shape[-1]
    mean = torch.mean(X, dim=-1).unsqueeze(-1)
    X = X - mean
    # return torch.matmul(X, X.T) / (D - 1)
    return 1 / (D - 1) * X @ X.transpose(-1, -2)

def T2_calculation(model, dataset, logger=None, log_flag=False):
    device = cfg.accelerator
    with torch.no_grad():
        loader = DataLoader(dataset, batch_size=BATCH_SIZE)
        pred_0_list = []
        pred_1_list = []
        for data in loader:
            pred, _ = model(data.to(device))
            pred = pred.detach()
            pred_0_list.extend(pred[0::2])
            pred_1_list.extend(pred[1::2])
        X = torch.cat([x.reshape(1, -1) for x in pred_0_list], dim=0).T
        Y = torch.cat([x.reshape(1, -1) for x in pred_1_list], dim=0).T
        if log_flag:
            logger.info(f"X_mean = {torch.mean(X, dim=1)}")
            logger.info(f"Y_mean = {torch.mean(Y, dim=1)}")
        D = (X - Y).cpu()
        D_mean = torch.mean(D, dim=1).reshape(-1, 1)
        S = cov(D)
        inv_S = torch.linalg.pinv(S)
        return torch.mm(torch.mm(D_mean.T, inv_S), D_mean)


@register_train('BREC')
def brec_train(loggers, loaders, model, optimizer, scheduler):
    # unused by this method, but needed to make function signature compatible 
    del loggers, loaders, model, optimizer, scheduler 
    # TODO: figure out loggers
    # TODO: make wandb logging more extensive
    if cfg.wandb.use:
        try:
            import wandb
        except:
            raise ImportError('WandB is not installed.')
        wandb_run = wandb.run

    device = cfg.accelerator
    dataset = load_dataset()

    cnt = 0
    correct_list = []
    fail_in_reliability = 0
    loss_func = CosineEmbeddingLoss(margin=MARGIN)

    for part_name, part_range in part_dict.items():
        # logger.info(f"{part_name} part starting ---")

        cnt_part = 0
        fail_in_reliability_part = 0

        for id in range(part_range[0], part_range[1]):
            # logger.info(f"ID: {id}")
            model = create_model(dim_in=cfg.gnn.dim_in, dim_out=OUTPUT_DIM) # TODO: revisit this, update other create_model
            if id == 0:
                cfg.params = params_count(model)
            optimizer = create_optimizer(model.parameters(), new_optimizer_config(cfg))
            scheduler = create_scheduler(optimizer, new_scheduler_config(cfg))
            dataset_traintest = dataset[
                id * NUM_RELABEL * 2 : (id + 1) * NUM_RELABEL * 2
            ]
            dataset_reliability = dataset[
                (id + SAMPLE_NUM) * NUM_RELABEL * 2 
                : (id + SAMPLE_NUM + 1) * NUM_RELABEL * 2
            ]
            model.train()
            for _ in range(cfg.optim.max_epoch):
                # training loop for each pair of graphs
                # model = create_model()
                traintest_loader = DataLoader(
                    dataset_traintest, batch_size=BATCH_SIZE
                )
                loss_all = 0
                for data in traintest_loader:
                    optimizer.zero_grad()
                    pred, _ = model(data.to(device))
                    loss = loss_func(
                        pred[0::2],
                        pred[1::2],
                        torch.tensor([-1] * (len(pred) // 2)).to(device),
                    )
                    loss.backward()
                    optimizer.step()
                    loss_all += len(pred) / 2 * loss.item()
                loss_all /= NUM_RELABEL
                # logger.info(f"Loss: {loss_all}")
                if loss_all < LOSS_THRESHOLD:
                    # logger.info("Early Stop Here")
                    break
                if cfg.optim.scheduler == 'reduce_on_plateau':
                    scheduler.step(loss_all)
                else:
                    scheduler.step()
            # evaluate on a single pair
            model.eval()
            T_square_traintest = T2_calculation(model, dataset_traintest, True)
            T_square_reliability = T2_calculation(model, dataset_reliability, True)

            isomorphic_flag = False
            reliability_flag = False
            if T_square_traintest > THRESHOLD and not torch.isclose(
                T_square_traintest, T_square_reliability, atol=EPSILON_CMP
            ):
                isomorphic_flag = True
            if T_square_reliability < THRESHOLD:
                reliability_flag = True

            if isomorphic_flag:
                cnt += 1
                cnt_part += 1
                correct_list.append(id)
            if not reliability_flag:
                # TODO: figure this out
                fail_in_reliability += 1
                fail_in_reliability_part += 1
            if cfg.wandb.use:
                # TODO: update stats for each run
                stats = {
                    "Total/Correct" : cnt,
                    "Total/Accuracy" : cnt/SAMPLE_NUM,
                    f"{part_name}/Correct" : cnt_part,
                    f"{part_name}/Accuracy" : cnt_part/(part_range[1]-part_range[0])
                }
                wandb_run.log(stats, step=id)

    if cfg.wandb.use:
        wandb_run.finish()
    
