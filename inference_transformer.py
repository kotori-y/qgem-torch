import argparse
import json
import os


import pandas as pd
import numpy as np
import torch
from rdkit import Chem
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from datasets import EgeognnInferenceDataset
from featurize import calculate_descriptor
from models.gin import EGeoGNNModel
from models.downstream import DownstreamTransformerModel
import random


def inference(model: DownstreamTransformerModel, device, loader, endpoints):
    model.eval()
    model.compound_encoder.eval()
    results = []
    smiles = []

    for step, batch in enumerate(loader):
        tgt_endpoints = torch.tensor(range(len(endpoints))).int().reshape(-1, 1)
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
            "tgt_endpoints": tgt_endpoints
        }

        with torch.no_grad():
            pred_scaled, _ = model(**input_params)
            results.append(pred_scaled.detach().cpu().numpy().flatten())
            smiles = [*smiles, *batch.smiles]

    results = np.hstack(results).reshape(-1, len(endpoints))
    return results, smiles[0::len(endpoints)]


def main(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)

    device = torch.device(args.device)
    with open(args.model_config_path) as f:
        config = json.load(f)

    with open(args.endpoint_status_file) as f:
        endpoint_statuses = json.load(f)

    dataset = EgeognnInferenceDataset(
        remove_hs=args.remove_hs,
        atom_names=config["atom_names"],
        bond_names=config["bond_names"],
        smiles_list=args.smiles_list
    )

    loader = DataLoader(
        dataset,
        batch_size=len(args.endpoints),
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
        "device": device,
        "without_dihedral": False,
    }
    compound_encoder = EGeoGNNModel(**encoder_params)

    if args.encoder_eval_from is not None:
        assert os.path.exists(args.encoder_eval_from)
        checkpoint = torch.load(args.encoder_eval_from, map_location=device)["compound_encoder_state_dict"]
        compound_encoder.load_state_dict(checkpoint)
        print(f"load params from {args.encoder_eval_from}")

    model_params = {
        "compound_encoder": compound_encoder,
        "n_layers": args.num_layers,
        "hidden_size": args.hidden_size,
        "dropout_rate": args.dropout_rate,
        "task_type": args.task_type,
        "endpoints": args.endpoints,
        "frozen_encoder": True,
        "device": device
    }
    model = DownstreamTransformerModel(**model_params).to(device)
    if args.model_eval_from is not None:
        assert os.path.exists(args.model_eval_from)
        checkpoint = torch.load(args.model_eval_from, map_location=device)["model_state_dict"]
        model.load_state_dict(checkpoint)
        print(f"load params from {args.model_eval_from}")

    model_without_ddp = model
    args.disable_tqdm = False

    num_params = sum(p.numel() for p in model_without_ddp.parameters())
    print(f"#Params: {num_params}")

    pred_scaled, smiles = inference(
        model=model,
        device=device,
        loader=loader,
        endpoints=args.endpoints
    )
    if args.task_type == "classification":
        pred_scaled = pred_scaled.argmax(dim=1, keepdim=True)

    pred_scaled_df = pd.DataFrame(pred_scaled, index=smiles, columns=args.endpoints)

    for endpoint in args.endpoints:
        mean = endpoint_statuses[endpoint]["mean"]
        std = endpoint_statuses[endpoint]["std"]
        pred_scaled_df[endpoint] = pred_scaled_df[endpoint] * std + mean

    if args.representation:
        mols = [Chem.MolFromSmiles(x) for x in smiles]
        desc = calculate_descriptor(mols)
        return np.hstack([desc, pred_scaled_df.values]).tolist()

    if args.return_json:
        return pred_scaled_df.T.to_dict()

    pred_scaled_df.to_csv(args.out_file, index_label='SMILES')
    return pred_scaled_df


def main_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default='cpu')

    parser.add_argument("--task-type", type=str)

    parser.add_argument("--model-config-path", type=str)
    parser.add_argument("--endpoint-status-file", type=str, default=None)
    parser.add_argument("--out-file", type=str)

    parser.add_argument("--smiles-list", type=str, nargs="+", default=None)
    parser.add_argument("--smiles-file", type=str)
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

    parser.add_argument("--encoder-eval-from", type=str, default=None)
    parser.add_argument("--model-eval-from", type=str, default=None)

    # parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)

    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--return-json", action='store_true', default=False)
    parser.add_argument("--representation", action='store_true', default=False)

    args = parser.parse_args()

    assert args.task_type in ['regression', 'classification']

    if args.representation:
        assert args.smiles_list is not None
        args.endpoints = [
            "alphaOpt", "density", "eGapOpt", "eLumoOpt", "espAvg", "espcMin", "espVarNeg", "hCorre", "mpi", "npaMin",
            "sa", "saPos", "uCorre", "zpe", "cm5Max", "e0K", "eHOMO", "eOpt", "espAvgNeg", "espMax", "espVarPos", "H",
            "mu", "nu", "saNeg", "S", "U", "cm5Min", "e", "eHomoOpt", "ese", "espAvgPos", "espMin", "gCorre",
            "hirshfeldMax", "muOpt", "pi", "saNonpolar", "theta", "volumeIMT", "cv", "eGap", "eLUMO", "eseOpt",
            "espcMax", "espVar", "G", "hirshfeldMin", "npaMax", "productOfSigmaSquareAndNu", "saPolar", "thetaOpt",
            "volumeMC"
        ]
        args.task_type = "regression"
        args.status_file = "./configs/tox_pc_mtl/qc_demo_status.json"

    if args.smiles_list is None:
        data = pd.read_csv(args.smiles_file, header=None)
        args.smiles_list = list(np.repeat(data[0].values, len(args.endpoints)))
    else:
        args.smiles_list = list(np.repeat(args.smiles_list, len(args.endpoints)))

    print(args)
    return main(args)
    # print(prediction)


if __name__ == '__main__':
    result = main_cli()
    print(result)
