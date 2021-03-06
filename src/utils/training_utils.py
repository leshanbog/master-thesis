import torch
import wandb
import os

def init_wandb(run_name, config, run_id=None):
    os.environ['WANDB_LOG_MODEL'] = 'false'
    os.environ['WANDB_WATCH'] = 'false'

    wandb.login()

    if run_id:
        wandb.init(project='master-thesis', name=run_name, resume=True, id=run_id)
    else:
        wandb.init(project='master-thesis', name=run_name)

    if run_id is None:
        for k, v in config.items():
            setattr(wandb.config, k, v)

        setattr(wandb.config, 'run_id', wandb.run.id)


def get_separate_lr_optimizer(model, enc_lr, dec_lr, warmup_steps, total_train_steps):
    from transformers import get_linear_schedule_with_warmup
    enc = []
    dec = []

    for name, param in model.named_parameters():
        if name.startswith('encoder'):
            enc.append(param)
        elif name.startswith('decoder'):
            dec.append(param)
        else:
            raise ValueError

    optimizer = torch.optim.AdamW([
        {'params': dec, 'lr': dec_lr},
        {'params': enc, 'lr': enc_lr}
    ])

    return optimizer, get_linear_schedule_with_warmup(optimizer, warmup_steps, total_train_steps)
