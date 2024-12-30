import argparse

import numpy as np
from rdkit import Chem

from featurize import calculate_descriptor
from inference_mt import main_inference


def main(args):
    mols = [Chem.MolFromSmiles(x) for x in args.smiles_list]
    mols = [mol for mol in mols if mol]

    desc = calculate_descriptor(mols)
    quantum = main_inference(args)

    return np.hstack([desc, quantum])


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

    feat = main(args)
    print(feat)


if __name__ == "__main__":
    main_cli()
