import torch

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