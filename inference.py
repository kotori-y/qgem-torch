import argparse
import json
import os

import numpy as np
import torch
from sklearn.metrics import r2_score
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from datasets import EgeognnInferenceDataset
from models.downstream import DownstreamModel
from models.gat import EGeoGNNModel
from models.inference import InferenceModel
from utils import init_distributed_mode


def inference(model: InferenceModel, device, loader, endpoints, args):
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

    prediction = np.hstack(prediction).T
    return {k: dict(zip(endpoints, v)) for k, v in zip(smiles, prediction)}


def main(args):
    device = torch.device(args.device)

    with open(args.config_path) as f:
        config = json.load(f)

    with open(args.status_path) as f:
        endpoint_statuses = json.load(f)

    dataset = EgeognnInferenceDataset(
        remove_hs=args.remove_hs,
        atom_names=config["atom_names"],
        bond_names=config["bond_names"],
        smiles_list=args.smiles_list
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
    }
    compound_encoder = EGeoGNNModel(**encoder_params)

    toxicity_endpoints = [
        "Cat_Intravenous_LD50",
        "Cat_Oral_LD50",
        "Chicken_Oral_LD50",
        "Dog_Oral_LD50",
        "Duck_Oral_LD50",
        "Guineapig_Oral_LD50",
        "Mouse_Intramuscular_LD50",
        "Mouse_Intraperitoneal_LD50",
        "Mouse_Intravenous_LD50",
        "Mouse_Oral_LD50",
        "Mouse_Subcutaneous_LD50",
        "Rabbit_Intravenous_LD50",
        "Rabbit_Oral_LD50",
        "Rat_Inhalation_LC50",
        "Rat_Intraperitoneal_LD50",
        "Rat_Intravenous_LD50",
        "Rat_Oral_LD50",
        "Rat_Skin_LD50",
        "Rat_Subcutaneous_LD50"
    ]

    physchem_endpoints = [
        "Density",
        "Vapor_Pressure",
        "Melting_Point",
        "Boiling_Point",
        "Flash_Point",
        "Decomposition",
        "Surface_Tension",
        "Drug_Half_Life",
        "Viscosity",
        "LogS",
        "Refractive_Index",
        "LogP",
        "Solubility"
    ]

    toxicity_model_params = {
        "compound_encoder": compound_encoder,
        "n_layers": args.num_layers,
        "hidden_size": args.hidden_size,
        "dropout_rate": args.dropout_rate,
        "task_type": "regression",
        "endpoints": toxicity_endpoints,
        "frozen_encoder": True,
        "inference": True
    }

    physchem_model_params = {
        "compound_encoder": compound_encoder,
        "n_layers": args.num_layers,
        "hidden_size": args.hidden_size,
        "dropout_rate": args.dropout_rate,
        "task_type": "regression",
        "endpoints": physchem_endpoints,
        "frozen_encoder": True,
        "inference": True
    }

    toxicity_model = DownstreamModel(**toxicity_model_params)
    physchem_model = DownstreamModel(**physchem_model_params)

    if args.toxicity_eval_from is not None:
        assert os.path.exists(args.toxicity_eval_from)
        checkpoint = torch.load(args.toxicity_eval_from, map_location=device)["model_state_dict"]
        toxicity_model.load_state_dict(checkpoint)

    if args.physchem_eval_from is not None:
        assert os.path.exists(args.physchem_eval_from)
        checkpoint = torch.load(args.physchem_eval_from, map_location=device)["model_state_dict"]
        physchem_model.load_state_dict(checkpoint)

    model = InferenceModel(
        toxicity_model=toxicity_model,
        physchem_model=physchem_model,
        toxicity_endpoints=toxicity_endpoints,
        physchem_endpoints=physchem_endpoints,
        endpoint_statuses=endpoint_statuses
    ).to(device)

    return inference(model=model, device=device, loader=loader, args=args, endpoints=toxicity_endpoints+physchem_endpoints)


def main_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default='cpu')

    parser.add_argument("--task-type", type=str)

    parser.add_argument("--config-path", type=str)
    parser.add_argument("--status-path", type=str)

    parser.add_argument("--smiles-list", type=str, nargs="+")

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

    parser.add_argument("--toxicity-eval-from", type=str, default=None)
    parser.add_argument("--physchem-eval-from", type=str, default=None)

    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)

    args = parser.parse_args()
    print(args)

    prediction = main(args)
    print(prediction)


if __name__ == '__main__':
    main_cli()
