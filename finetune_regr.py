import argparse
import io
import json
import os
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import optim
from torch.utils.data import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from datasets import EgeognnFinetuneDataset
from models.downstream import DownstreamModel
from utils import exempt_parameters, init_distributed_mode


def train(
        model, device, loader,
        encoder_opt, head_opt,
        args
):
    model.train()

    loss_accum_dict = defaultdict(float)
    counter = {}
    pbar = tqdm(loader, desc="Iteration", disable=args.disable_tqdm)

    for step, batch in enumerate(pbar):
        input_params = {
            "AtomBondGraph_edges": batch.AtomBondGraph_edges.to(device),
            "BondAngleGraph_edges": batch.BondAngleGraph_edges.to(device),
            "AngleDihedralGraph_edges": batch.AngleDihedralGraph_edges.to(device),
            "pos": batch.atom_poses.to(device),
            "x": batch.node_feat.to(device),
            "bond_attr": batch.edge_attr.to(device),
            "bond_lengths": batch.bond_lengths.to(device),
            "bond_angles": batch.bond_angles.to(device),
            "dihedral_angles": batch.dihedral_angles.to(device),
            "num_graphs": batch.num_graphs,
            "num_atoms": batch.n_atoms.to(device),
            "num_bonds": batch.n_bonds.to(device),
            "num_angles": batch.n_angles.to(device),
            "atom_batch": batch.batch.to(device),
            "tgt_endpoints": batch.endpoint
        }

        if len(batch.label) < args.batch_size * 0.5:
            continue
        if batch.node_feat.shape[0] == 1 or batch.batch[-1] == 0:
            continue

        if encoder_opt is not None:
            encoder_opt.zero_grad()
        head_opt.zero_grad()

        pred = model(**input_params)

        label, label_mean, label_std = batch.label.reshape(-1, 3).T
        label = label.to(device)
        label_mean = label_mean.to(device)
        label_std = label_std.to(device)

        if args.distributed:
            loss, loss_dict = model.module.compute_loss(
                pred, label, batch.endpoint
            )
        else:
            loss, loss_dict = model.compute_loss(pred, label, batch.endpoint)

        loss.backward()
        if encoder_opt is not None:
            encoder_opt.step()
        head_opt.step()

        for k, v in loss_dict.items():
            counter[k] = counter.get(k, 0) + 1
            loss_accum_dict[k] += v

        if step % args.log_interval == 0:
            description = f"Iteration loss: {loss_accum_dict['loss'] / (step + 1):6.4f}"
            pbar.set_description(description)

    for k in loss_accum_dict.keys():
        loss_accum_dict[k] /= counter[k]

    return loss_accum_dict


def evaluate(model, device, loader, args):
    model.eval()

    loss_accum_dict = defaultdict(float)
    counter = {}
    pbar = tqdm(loader, desc="Iteration", disable=args.disable_tqdm)

    for step, batch in enumerate(pbar):
        input_params = {
            "AtomBondGraph_edges": batch.AtomBondGraph_edges.to(device),
            "BondAngleGraph_edges": batch.BondAngleGraph_edges.to(device),
            "AngleDihedralGraph_edges": batch.AngleDihedralGraph_edges.to(device),
            "pos": batch.atom_poses.to(device),
            "x": batch.node_feat.to(device),
            "bond_attr": batch.edge_attr.to(device),
            "bond_lengths": batch.bond_lengths.to(device),
            "bond_angles": batch.bond_angles.to(device),
            "dihedral_angles": batch.dihedral_angles.to(device),
            "num_graphs": batch.num_graphs,
            "num_atoms": batch.n_atoms.to(device),
            "num_bonds": batch.n_bonds.to(device),
            "num_angles": batch.n_angles.to(device),
            "atom_batch": batch.batch.to(device),
            "tgt_endpoints": batch.endpoint
        }

        if len(batch.label) < args.batch_size * 0.5:
            continue
        if batch.node_feat.shape[0] == 1 or batch.batch[-1] == 0:
            continue

        with torch.no_grad():
            pred = model(**input_params)

        label, label_mean, label_std = batch.label.reshape(-1, 3).T
        label = label.to(device)
        label_mean = label_mean.to(device)
        label_std = label_std.to(device)

        if args.distributed:
            loss, loss_dict = model.module.compute_loss(
                pred, label, batch.endpoint
            )
        else:
            loss, loss_dict = model.compute_loss(pred, label, batch.endpoint)

        for k, v in loss_dict.items():
            counter[k] = counter.get(k, 0) + 1
            loss_accum_dict[k] += v

        if step % args.log_interval == 0:
            description = f"Iteration loss: {loss_accum_dict['loss'] / (step + 1):6.4f}"
            pbar.set_description(description)

    for k in loss_accum_dict.keys():
        loss_accum_dict[k] /= counter[k]

    return loss_accum_dict


def get_label_stat(dataset):
    """tbd"""
    labels = np.array([data['label'] for data in dataset])

    nan_mask = np.isnan(labels)

    return np.min(labels[~nan_mask]), np.max(labels[~nan_mask]), np.mean(labels[~nan_mask])


def main(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)

    device = torch.device(args.device)
    with open(args.config_path) as f:
        config = json.load(f)

    dataset = EgeognnFinetuneDataset(
        root=args.dataset_root,
        remove_hs=args.remove_hs,
        base_path=args.dataset_base_path,
        atom_names=config["atom_names"],
        bond_names=config["bond_names"],
        dev=args.dev,
        endpoints=args.endpoints,
        useMPI=args.useMPI
    )

    if args.dataset:
        return None

    total_num = len(dataset)
    train_num = int(total_num * 0.8)
    # valid_num = total_num - train_num

    train_idx = list(random.sample(range(total_num), train_num))
    valid_idx = list(set(range(total_num)) - set(train_idx))
    # test_idx = range(train_num + valid_num, total_num)

    split_idx = {
        "train": torch.tensor(train_idx, dtype=torch.long),
        "valid": torch.tensor(valid_idx, dtype=torch.long)
        # "test": torch.tensor(test_idx, dtype=torch.long),
    }

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
        shuffle=False,
        num_workers=args.num_workers
    )

    # test_loader = DataLoader(
    #     dataset[split_idx["test"]],
    #     batch_size=args.batch_size,
    #     shuffle=False,
    #     num_workers=args.num_workers
    # )

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
        "global_reducer": 'sum',
    }
    compound_encoder = EGeoGNNModel(**encoder_params)

    if args.encoder_eval_from is not None:
        assert os.path.exists(args.encoder_eval_from)
        checkpoint = torch.load(args.encoder_eval_from, map_location=device)["compound_encoder_state_dict"]
        compound_encoder.load_state_dict(checkpoint)

    model_params = {
        "compound_encoder": compound_encoder,
        "n_layers": args.num_layers,
        "hidden_size": args.hidden_size,
        "dropout_rate": args.dropout_rate,
        "task_type": args.task_type,
        "endpoints": args.endpoints,
        "frozen_encoder": args.frozen_encoder
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

    num_params = sum(p.numel() for p in model_without_ddp.parameters())
    print(f"#Params: {num_params}")

    encoder_params = list(compound_encoder.parameters())
    head_params = exempt_parameters(model_without_ddp.parameters(), encoder_params)
    if args.frozen_encoder:
        for params in encoder_params:
            params.requires_grad = False

    if args.use_adamw:
        encoder_optimizer = optim.AdamW(
            encoder_params,
            lr=args.encoder_lr,
            betas=(0.9, args.beta),
            weight_decay=args.weight_decay,
        ) if not args.frozen_encoder else None
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
        ) if not args.frozen_encoder else None
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

    train_curve = []
    valid_curve = []
    # test_curve = []

    for epoch in range(1, args.epochs + 1):
        print("=====Epoch {}".format(epoch))

        if args.distributed:
            sampler_train.set_epoch(epoch)

        print("Training...")
        train_dict = train(
            model=model,
            device=device,
            loader=train_loader,
            encoder_opt=encoder_optimizer,
            head_opt=head_optimizer,
            args=args
        )

        print("Evaluating...")
        valid_dict = evaluate(model, device, valid_loader, args)
        # test_dict = evaluate(model, device, test_loader, args)

        if args.checkpoint_dir:
            print(f"Setting {os.path.basename(os.path.normpath(args.checkpoint_dir))}...")

        train_pref = train_dict['loss']
        valid_pref = valid_dict['loss']
        # test_pref = test_dict['loss']

        train_curve.append(train_pref)
        valid_curve.append(valid_pref)
        # test_curve.append(test_pref)

        if args.checkpoint_dir:
            logs = {"epoch": epoch, "Train": train_dict, "Valid": valid_dict}
            with io.open(
                    os.path.join(args.checkpoint_dir, "log.txt"), "a", encoding="utf8", newline="\n"
            ) as tgt:
                print(json.dumps(logs), file=tgt)

            checkpoint = {
                "epoch": epoch,
                "compound_encoder": compound_encoder.state_dict(),
                "model_state_dict": model_without_ddp.state_dict(),
                "encoder_optimizer_state_dict": encoder_optimizer.state_dict() if not args.frozen_encoder else {},
                "head_optimizer_state_dict": head_optimizer.state_dict(),
                # "scheduler_state_dict": scheduler.state_dict(),
                "args": args,
            }
            torch.save(checkpoint, os.path.join(args.checkpoint_dir, f"checkpoint_{epoch}.pt"))
            if args.enable_tb:
                tb_writer = SummaryWriter(args.checkpoint_dir)
                tb_writer.add_scalar("evaluation/train", train_pref, epoch)
                tb_writer.add_scalar("evaluation/valid", valid_pref, epoch)
                # tb_writer.add_scalar("evaluation/test", test_pref, epoch)
                for k, v in train_dict.items():
                    tb_writer.add_scalar(f"training/{k}", v, epoch)


def main_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev", action='store_true', default=False)
    parser.add_argument("--dataset", action='store_true', default=False)
    parser.add_argument("--device", type=str, default='cpu')

    parser.add_argument("--task-type", type=str)

    parser.add_argument("--config-path", type=str)

    parser.add_argument("--dataset-root", type=str)
    parser.add_argument("--dataset-base-path", type=str)

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
    parser.add_argument("--encoder-eval-from", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--use-adamw", action='store_true', default=False)

    parser.add_argument("--encoder_lr", type=float, default=1e-3)
    parser.add_argument("--head_lr", type=float, default=1e-3)

    parser.add_argument("--beta", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--lr-warmup", action='store_true', default=False)

    parser.add_argument("--distributed", action='store_true', default=False)
    parser.add_argument("--useMPI", action='store_true', default=False)
    parser.add_argument("--endpoints", type=str, nargs="+")

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--log-interval", type=int, default=5)
    parser.add_argument("--grad-norm", type=float, default=None)

    parser.add_argument("--model-ver", type=str, default='gat')
    parser.add_argument("--frozen-encoder", action='store_true', default=False)
    parser.add_argument("--enable-tb", action='store_true', default=False)

    parser.add_argument("--seed", type=int, default=2024)

    args = parser.parse_args()
    print(args)

    init_distributed_mode(args)

    main(args)


if __name__ == '__main__':
    main_cli()
