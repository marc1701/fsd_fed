import copy
import random
import datetime
import os, glob, argparse
import torch, torchaudio
import matplotlib.pyplot as plt
import numpy as np, pandas as pd
from torch import nn
from torch.utils.data import Dataset, DataLoader, IterableDataset
from pytorch_lightning.metrics.functional import average_precision
from pytorchtools import EarlyStopping

import pandas as pd
import seaborn as sns; sns.set()

# custom code
from fed_fsd import *
from metrics_plots import *

from ray import tune
from ray.tune import CLIReporter
from functools import partial

# command line arguments
parser = argparse.ArgumentParser(
    description='Conducts and tunes federated learning on FSD50k dataset')
parser.add_argument('-p', '--fsd50k_path', required=True,
                    help='Path to the FSD50K dataset')
parser.add_argument('-o', '--out_path', required=True,
                    help='path of working directory')
parser.add_argument('--max_rounds', nargs='?', default=43, type=int,
                    help='maximum number of comms rounds')
args = parser.parse_args()


def federated_train(config,
                    subdir=None, checkpoint_dir='checkpoints', model_pt=None,
                    train_batch_size=64, eval_batch_size=300):
    '''modified version of this function from fed_fsd.py
    for compatibility with ray.tune'''
    # # wait for GPU. see: https://docs.ray.io/en/master/tune/user-guide.html
    # this crashes since not implemented yet.
    # tune.util.wait_for_gpu()
    #
    rounds=args.max_rounds
    train_data = FSD50K_MelSpec1s(split='train', subdir=subdir)
    train_dataloader = DataLoader(train_data,
        batch_size=train_batch_size, num_workers=1, shuffle=True)
    print('Loaded Training Data')

    val_data = FSD50K_MelSpec1s(split='val', subdir=subdir)
    val_dataloader = DataLoader(val_data,
        batch_size=eval_batch_size, num_workers=1, shuffle=True)
    print('Loaded Val Data')

    # make copies of training data and set for each individual uploader
    clients = [copy.copy(train_data)
        for n in range(len(train_data.all_uploaders))]
    for n, uploader in enumerate(train_data.all_uploaders):
        clients[n].set_uploader(uploader)
    print('Data successfully allocated to clients')

    device = (torch.device('cuda') if torch.cuda.is_available()
        else torch.device('cpu'))
    print(f"Training on {device}")

    glob_model = FSD_VGG().to(device) # random init 'global' model
    if model_pt: # load previous model if provided
        glob_model.load_state_dict(torch.load(model_pt))
    loss_func = nn.BCELoss()

    n_clients_to_select = round(config['C'] * len(clients))

    for comms_round in range(rounds):
        print("Round", comms_round + 1, "of", rounds)

        # randomly select clients for this round
        this_rounds_clients = random.sample(clients, n_clients_to_select)

        # set up dict with dataloaders and models for each client
        this_rounds_data = dict()
        for client in this_rounds_clients:
            this_rounds_data[client.uploader] = [
                client,
                DataLoader(client, batch_size=train_batch_size, num_workers=1,
                           shuffle=True),
                copy.deepcopy(glob_model)]
        len_data = sum([len(client) for client in this_rounds_clients])

        # train client models
        for client_i, (key, item) in enumerate(this_rounds_data.items(), 1):
            client, dataloader, model = item
            # optim must get the client model parameters
            # ? weight decay
            optim = torch.optim.Adam(model.parameters(), lr=5e-4)
            print(f"\n[{client_i}/{n_clients_to_select}] Training {key}'s",
                  "model", end="")
            train(n_epochs=config['E'],
                  optimiser=optim,
                  model=model,
                  loss_func=loss_func,
                  device=device,
                  dataloader=dataloader)

        # zero out global model parameters
        zero_model_params(glob_model)

        # add weighted parameters from this round's local models
        for key, item in this_rounds_data.items():
            client, _, client_model = item
            client_weight = len(client) / len_data
            fed_weight_update(glob_model, client_model, client_weight)

        print('\nEvaluating global model...')
        auc_val = ray_model_eval(glob_model, val_dataloader, val_data, device)

        print('\nGlobal Val PR-AUC: {:.4f}\n'.format(float(auc_val)))

        with tune.checkpoint_dir(comms_round) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint.pt")
            print(path)
            torch.save(glob_model.state_dict(), path)
        tune.report(auc_val=auc_val)


def ray_model_eval(model, test_dataloader, test_data, device):

    model.eval()
    # initialise output aggregation array
    test_clip_scores = torch.zeros(len(test_data.info), 200).to(device)

    with torch.no_grad():
        num_eval_iters = len(test_dataloader)
        for test_i, (x, y_true, filepath) in enumerate(test_dataloader, 1):
            print(f" eval round [{test_i}/{num_eval_iters}]")
            # send data to GPU if available
            x = x.to(device); y_true = y_true.to(device)

            # get predictions
            y_pred = model(x)

            # save predictions for aggregation across whole clips
            for i, output in enumerate(y_pred):
                clip_number = int(filepath[i].split('/')[-1].split('.')[0])
                clip_index = np.where(test_data.clip_order == clip_number)[0][0]
                test_clip_scores[clip_index] += output

        # divide scores by number of segments per clip
        # (averaging across segments)
        test_clip_scores /= test_data.n_segs.to(device)

        auc_val = float(
            pr_auc(test_clip_scores.cpu(), test_data.ground_truth.cpu()))

    return auc_val

def main():
    """
    """
    config = {
        'C'  : tune.grid_search([0.1, 0.3, 0.5, 0.7]),
        'E'  : tune.grid_search([1, 3, 5])
    }

    reporter = CLIReporter(
        parameter_columns=['C', 'E'],
        metric_columns=['auc_val']
    )

    result = tune.run(
        partial(federated_train,
                subdir=args.fsd50k_path),
        resources_per_trial={'cpu': 1, 'gpu': 1},
        config=config,
        local_dir=args.out_path,
        name='fed_fsd',
        progress_reporter=reporter)

    return result

result = main()

# write output dataframe to file
filepath = os.path.join(args.out_path, 'hpt_result.pkl')
file_to_write = open(filepath, 'wb')
pickle.dump(result.dataframe(), file_to_write)
file_to_write.close()
