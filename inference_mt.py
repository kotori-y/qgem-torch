import argparse
import json
import os
import random

import numpy as np
import torch
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from datasets import EgeognnInferenceDataset
from models.downstream import DownstreamMTModel
from models.gin import EGeoGNNModel
from models.inference import InferenceMTModel


def inference(model: InferenceMTModel, device, loader, endpoints, return_value=True):
    model.eval()

    prediction = []
    smiles = []
    pbar = tqdm(loader, desc="Iteration")

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
        }

        with torch.no_grad():
            pred = model(**input_params)

        smiles += batch.smiles
        prediction.append(pred)

    prediction = np.hstack(prediction)

    if return_value:
        return prediction

    return {
        k: {ki: vi for ki, vi in zip(endpoints, v)}
        for k, v in zip(smiles, prediction)
    }


def main_inference(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)

    device = torch.device(args.device)

    with open(args.config_path) as f:
        config = json.load(f)

    with open(args.status_path) as f:
        endpoint_statuses = json.load(f)

    dataset = EgeognnInferenceDataset(
        smiles_list=args.smiles_list,
        remove_hs=args.remove_hs,
        atom_names=config["atom_names"],
        bond_names=config["bond_names"],
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )

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
        "device": device
    }
    compound_encoder = EGeoGNNModel(**encoder_params)

    if args.eval_from is not None:
        assert os.path.exists(args.eval_from)
        checkpoint = torch.load(args.eval_from, map_location=device)["compound_encoder"]
        compound_encoder.load_state_dict(checkpoint)
    compound_encoder.eval()

    model_params = {
        "compound_encoder": compound_encoder,
        "n_layers": args.num_layers,
        "hidden_size": args.hidden_size,
        "dropout_rate": args.dropout_rate,
        "task_type": args.task_type,
        "endpoints": args.endpoints,
        "frozen_encoder": True
    }
    downstream_model = DownstreamMTModel(**model_params)

    if args.eval_from is not None:
        assert os.path.exists(args.eval_from)
        checkpoint = torch.load(args.eval_from, map_location=device)["model_state_dict"]
        downstream_model.load_state_dict(checkpoint)

    model = InferenceMTModel(
        pred_model=downstream_model,
        endpoints=args.endpoints,
        endpoint_statuses=endpoint_statuses
    ).to(device)

    return inference(
        model=model,
        device=device,
        loader=loader,
        endpoints=args.endpoints
    )


def main_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev", action='store_true', default=False)
    parser.add_argument("--device", type=str, default='cpu')

    parser.add_argument("--task-type", type=str)

    parser.add_argument("--config-path", type=str)
    parser.add_argument("--status-path", type=str)

    parser.add_argument("--smiles-list", type=str, nargs="+")
    parser.add_argument("--endpoints", type=str, nargs="+")

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

    parser.add_argument("--eval-from", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=1)

    parser.add_argument("--seed", type=int, default=2024)

    args = parser.parse_args()
    print(args)

    prediction = main_inference(args)
    print(prediction)


if __name__ == '__main__':
    main_cli()
