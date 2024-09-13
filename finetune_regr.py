import json
import os
from os.path import join, exists, basename
import argparse
from pathlib import Path
import io

import numpy as np
import torch

from torch import nn, optim
from torch.optim.lr_scheduler import LambdaLR

from torch.utils.data import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from datasets import EgeognnFinetuneDataset
from models.downstream import DownstreamModel
from utils import WarmCosine, exempt_parameters, init_distributed_mode


def train(
        model, device,
        loader, criterion,
        encoder_opt, head_opt,
        args
):
    list_loss = []
    model.train()

    pbar = tqdm(loader, desc="Iteration", disable=args.disable_tqdm)
    for step, batch in enumerate(pbar):
        input_params = {
            "AtomBondGraph_edges": batch.AtomBondGraph_edges.to(device),
            "BondAngleGraph_edges": batch.BondAngleGraph_edges.to(device),
            "AngleDihedralGraph_edges": batch.AngleDihedralGraph_edges.to(device),
            "x": batch.node_feat.to(device),
            "bond_attr": batch.edge_attr.to(device),
            "bond_lengths": batch.bond_lengths.to(device),
            "bond_angles": batch.bond_angles.to(device),
            "dihedral_angles": batch.dihedral_angles.to(device),
            "num_graphs": batch.num_graphs,
            "num_atoms": batch.n_atoms.to(device),
            "num_bonds": batch.n_bonds.to(device),
            "num_angles": batch.n_angles.to(device),
            "atom_batch": batch.batch.to(device)
        }

        if len(batch.labels) < args.batch_size * 0.5:
            continue
        if batch.node_feat.shape[0] == 1 or batch.batch[-1] == 0:
            continue

        encoder_opt.zero_grad()
        head_opt.zero_grad()

        loss, pred = model(**input_params)

        loss.backward()
        encoder_opt.step()
        head_opt.step()

        labels = batch.labels
        label_mean = batch.label_mean
        label_std = batch.label_std

        scaled_labels = (labels - label_mean) / (label_std + 1e-5)
        scaled_labels = paddle.to_tensor(scaled_labels, 'float32')

        loss = process_nan_value(preds, scaled_labels, criterion)
        loss.backward()

    #     encoder_opt.step()
    #     head_opt.step()
    #     encoder_opt.clear_grad()
    #     head_opt.clear_grad()
    #     list_loss.append(loss.numpy())
    # return np.mean(list_loss)


def evaluate(
        args,
        model, label_mean, label_std,
        test_dataset, collate_fn, metric):
    """
    Define the evaluate function
    In the dataset, a proportion of labels are blank. So we use a `valid` tensor
    to help eliminate these blank labels in both training and evaluation phase.
    """
    data_gen = test_dataset.get_data_loader(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        collate_fn=collate_fn)
    total_pred = []
    total_label = []

    model.eval()

    for atom_bond_graphs, bond_angle_graphs, angle_dihedral_graphs, labels in data_gen:
        atom_bond_graphs = atom_bond_graphs.tensor()
        bond_angle_graphs = bond_angle_graphs.tensor()
        angle_dihedral_graphs = angle_dihedral_graphs.tensor()
        labels = paddle.to_tensor(labels, 'float32')

        scaled_preds = model(atom_bond_graphs, bond_angle_graphs, angle_dihedral_graphs)
        preds = scaled_preds.numpy() * label_std + label_mean
        total_pred.append(preds)
        total_label.append(labels.numpy())
    total_pred = np.concatenate(total_pred, 0)
    total_label = np.concatenate(total_label, 0)

    if metric == 'rmse':
        return calc_rmse(total_label, total_pred)
    else:
        return calc_mae(total_label, total_pred)


def get_label_stat(dataset):
    """tbd"""
    labels = np.array([data['label'] for data in dataset])

    nan_mask = np.isnan(labels)

    return np.min(labels[~nan_mask]), np.max(labels[~nan_mask]), np.mean(labels[~nan_mask])


def get_metric(dataset_name):
    """tbd"""
    if dataset_name in [
        'esol', 'freesolv', 'lipophilicity',
        'logpow', 'solubility', 'boilingpoint',
        'pka', 'diy', 'special_pc'
    ] or 'sp' in dataset_name:
        return 'rmse'

    if dataset_name in ['qm7', 'qm8', 'qm9', 'qm9_gdb', 'quandb']:
        return 'mae'

    raise ValueError(dataset_name)


def main(args):
    device = torch.device(args.device)
    with open(args.config_path) as f:
        config = json.load(f)

    dataset = EgeognnFinetuneDataset(
        root=args.dataset_root,
        dataset_name=args.dataset_name,
        remove_hs=args.remove_hs,
        base_path=args.dataset_base_path,
        atom_names=config["atom_names"],
        bond_names=config["bond_names"],
        dev=args.dev
    )

    split_idx = dataset.get_idx_split()

    train_loader = DataLoader(
        dataset[split_idx["train"]],
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
    )

    valid_loader = DataLoader(
        dataset[split_idx["valid"]],
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )

    if args.distributed:
        sampler_train = DistributedSampler(dataset)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset)

    if args.model_ver == 'gat':
        from models.gat import EGeoGNNModel
    else:
        from models.gin import EGeoGNNModel

    encoder_params = {
        "latent_size": args.latent_size,
        "hidden_size": args.encoder_hidden_size,
        "n_layers": args.num_encoder_layers,
        "use_layer_norm": True,
        "layernorm_before": True,
        "use_bn": True,
        "dropnode_rate": args.dropnode_rate,
        "encoder_dropout": args.encoder_dropout,
        "num_message_passing_steps": args.num_message_passing_steps,
        "atom_names": config["atom_names"],
        "bond_names": config["bond_names"],
        "global_reducer": 'sum'
    }
    compound_encoder = EGeoGNNModel(**encoder_params)

    model_params = {
        "compound_encoder": compound_encoder,
        "n_layers": args.num_layers,
        "hidden_size": args.hidden_size,
        "dropout_rate": args.dropout_rate,
        "task_type": args.task_type,
        "num_tasks": 1
    }
    model = DownstreamModel(**model_params).to(device)

    model_without_ddp = model
    args.disable_tqdm = False

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank])
        model_without_ddp = model.module

        args.checkpoint_dir = "" if args.rank != 0 else args.checkpoint_dir
        args.enable_tb = False if args.rank != 0 else args.enable_tb
        args.disable_tqdm = args.rank != 0

    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    with io.open(
            os.path.join(args.checkpoint_dir, "log.txt"), "a", encoding="utf8", newline="\n"
    ) as tgt:
        print(args, file=tgt)

    if args.eval_from is not None:
        assert os.path.exists(args.eval_from)
        checkpoint = torch.load(args.eval_from, map_location=device)["model_state_dict"]
        model_without_ddp.load_state_dict(checkpoint)

    num_params = sum(p.numel() for p in model_without_ddp.parameters())
    print(f"#Params: {num_params}")

    encoder_params = compound_encoder.parameters()
    head_params = exempt_parameters(model_without_ddp.parameters(), encoder_params)

    if args.use_adamw:
        encoder_optimizer = optim.AdamW(
            encoder_params,
            lr=args.encoder_lr,
            betas=(0.9, args.beta),
            weight_decay=args.weight_decay,
        )
        head_optimizer = optim.AdamW(
            head_params,
            lr=args.head_lr,
            betas=(0.9, args.beta),
            weight_decay=args.weight_decay,
        )
    else:
        encoder_optimizer = optim.Adam(
            encoder_params,
            lr=args.encoder_lr,
            betas=(0.9, args.beta),
            weight_decay=args.weight_decay,
        )
        head_optimizer = optim.Adam(
            head_params,
            lr=args.head_lr,
            betas=(0.9, args.beta),
            weight_decay=args.weight_decay,
        )

    # if not args.lr_warmup:
    #     scheduler = LambdaLR(optimizer, lambda x: 1.0)
    # else:
    #     lrscheduler = WarmCosine(tmax=len(train_loader) * args.period, warmup=int(4e3))
    #     scheduler = LambdaLR(optimizer, lambda x: lrscheduler.step(x))

    criterion = nn.MSELoss(reduction='none')

    for epoch in range(1, args.epochs + 1):
        print("=====Epoch {}".format(epoch))

        if args.distributed:
            sampler_train.set_epoch(epoch)

        print("Training...")
        train(
            model=model, device=device,
            loader=train_loader, criterion=criterion,
            encoder_opt=encoder_optimizer, head_opt=head_optimizer,
            args=args
        )

        print("Evaluating...")
        valid_dict = evaluate(model, device, valid_loader, args)


def main_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev", action='store_true', default=False)
    parser.add_argument("--device", type=str, default='cpu')

    parser.add_argument("--config-path", type=str)

    parser.add_argument("--dataset-root", type=str)
    parser.add_argument("--dataset-base-path", type=str)
    parser.add_argument("--dataset-name", type=str)

    parser.add_argument("--remove-hs", action='store_true', default=False)

    parser.add_argument("--latent-size", type=int, default=128)
    parser.add_argument("--encoder-hidden-size", type=int, default=256)
    parser.add_argument("--num-encoder-layers", type=int, default=4)
    parser.add_argument("--dropnode-rate", type=float, default=0.1)
    parser.add_argument("--encoder-dropout", type=float, default=0.1)
    parser.add_argument("--num-message-passing-steps", type=int, default=2)

    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--dropout-rate", type=float, default=0.1)

    parser.add_argument("--checkpoint-dir", type=str, default="")
    parser.add_argument("--eval-from", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--use-adamw", action='store_true', default=False)

    parser.add_argument("--encoder_lr", type=float, default=1e-3)
    parser.add_argument("--head_lr", type=float, default=1e-3)

    parser.add_argument("--beta", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--lr-warmup", action='store_true', default=False)

    parser.add_argument("--distributed", action='store_true', default=False)

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--log-interval", type=int, default=5)
    parser.add_argument("--grad-norm", type=float, default=None)

    parser.add_argument("--model-ver", type=str, default='gat')

    args = parser.parse_args()
    print(args)

    init_distributed_mode(args)

    main(args)


if __name__ == '__main__':
    main_cli()

