import argparse

import mlflow
import torch
from torch.optim import *

import numpy as np

from configs.parser import YAMLParser
from dataloader.h5 import H5Loader
from loss.flow import EventWarping
from models.model import (
    FireNet_Sparsify,
    FireNet,
    RNNFireNet,
    LeakyFireNet,
    FireFlowNet,
    LeakyFireFlowNet,
    E2VID,
    EVFlowNet,
    RecEVFlowNet,
    LeakyRecEVFlowNet,
    RNNRecEVFlowNet,
)
from models.model import (
    LIFFireNet,
    PLIFFireNet,
    ALIFFireNet,
    XLIFFireNet,
    LIFFireFlowNet,
    SpikingRecEVFlowNet,
    PLIFRecEVFlowNet,
    ALIFRecEVFlowNet,
    XLIFRecEVFlowNet,
)
from utils.gradients import get_grads
from utils.utils import load_model, save_csv, save_diff, save_model_best, save_model_latest, save_model_tail
from utils.visualization import Visualization


def train(args, config_parser):
    mlflow.set_tracking_uri(args.path_mlflow)

    # configs
    config = config_parser.config
    if config["data"]["mode"] == "frames":
        print("Config error: Training pipeline not compatible with frames mode.")
        raise AttributeError
    
    config["sparsify"]["regularizer_weight_voltage"] = float(args.regularizer_weight_voltage)
    config["sparsify"]["regularizer_weight_threshold"] = float(args.regularizer_weight_threshold)

    # log config
    mlflow.set_experiment(config["experiment"])
    mlflow.start_run()
    mlflow.log_params(config)
    mlflow.log_param("prev_runid", args.prev_runid) # pre-trained model to use as starting point
    config = config_parser.combine_entries(config)
    print("MLflow dir:", mlflow.active_run().info.artifact_uri[:-9])

    # log git diff
    save_diff("train_diff.txt")

    regularizer_weight_threshold = float(config["sparsify"]["regularizer_weight_threshold"])
    regularizer_weight_voltage = float(config["sparsify"]["regularizer_weight_voltage"])
    print("regularizer_weight_threshold:", regularizer_weight_threshold, "regularizer_weight_voltage:", regularizer_weight_voltage)

    # initialize settings
    device = config_parser.device
    kwargs = config_parser.loader_kwargs

    # visualization tool
    if config["vis"]["enabled"]:
        vis = Visualization(config)

    # data loader
    data = H5Loader(config, config["model"]["num_bins"], config["model"]["round_encoding"])
    dataloader = torch.utils.data.DataLoader(
        data,
        drop_last=True,
        batch_size=config["loader"]["batch_size"],
        collate_fn=data.custom_collate,
        worker_init_fn=config_parser.worker_init_fn,
        **kwargs,
    )

    # loss function
    loss_function = EventWarping(config, device)

    # model initialization and settings
    model = eval(config["model"]["name"])(config["model"].copy()).to(device)
    model = load_model(args.prev_runid, model, device)

    model.train()

    # optimizers
    optimizer = eval(config["optimizer"]["name"])(model.parameters(), lr=config["optimizer"]["lr"], weight_decay=config["optimizer"]["weight_decay"])
    optimizer.zero_grad()

    if config["optimizer"]["OneCycleLR_scheduler"]:
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer=optimizer, max_lr=config["optimizer"]["lr"], epochs=config["loader"]["n_epochs"]+1, steps_per_epoch=int(500000/config["data"]["window_loss"]*145/config["loader"]["batch_size"]))
        # 500000 is the number of events in each training sequence, 145 is the number of training sequences, 
        use_OneCycleLR_scheduler = True
    else:
        use_OneCycleLR_scheduler = False

    # simulation variables
    train_loss = 0
    best_loss = 1.0e6
    end_train = False
    grads_w = []

    sparsity_sum = {'0:input': 0, '1:head': 0, '2:G1': 0, '3:R1a': 0, '4:R1b': 0, '5:G2': 0, '6:R2a': 0, '7:R2b': 0, '8:pred': 0}

    if config["sparsify"]["regularizer_voltage"] == "Hoyer":
        regularizer_voltage = 0
    elif config["sparsify"]["regularizer_voltage"] == "L1":
        regularizer_voltage = 1
    else:
        assert(config["sparsify"]["regularizer_voltage"] == "L2")
        regularizer_voltage = 2

    if config["sparsify"]["regularizer_threshold"] == "Hoyer":
        regularizer_threshold = 0
    elif config["sparsify"]["regularizer_threshold"] == "L1":
        regularizer_threshold = 1
    else:
        assert(config["sparsify"]["regularizer_threshold"] == "L2")
        regularizer_threshold = 2

    data_counter = 0
    regularizer_loss_voltage = 0
    regularizer_loss_thresh = 0
    regularizer_print = 0

    # training loop
    data.shuffle()
    print("Learning rate in epoch {:02d}: {:.4e}".format(data.epoch, optimizer.state_dict()['param_groups'][0]['lr']))
    while True: 
        for inputs in dataloader:
            
            # for key in inputs.keys():
            #     print(key, inputs[key].shape)

            if data.new_seq:
                data.new_seq = False

                loss_function.reset()
                model.reset_states()
                optimizer.zero_grad()

            if data.seq_num >= len(data.files): # finished an epoch
                mlflow.log_metric("loss", train_loss / (data.samples + 1), step=data.epoch)
                print("\nEpoch: {:04d} finished. Avg Loss: {:.6f} \n".format(data.epoch, train_loss / (data.samples + 1)))

                with torch.no_grad():
                    if train_loss / (data.samples + 1) < best_loss:
                        save_model_best(model)
                        print("save the best model")
                        best_loss = train_loss / (data.samples + 1)

                data.epoch += 1
                data.samples = 0
                train_loss = 0
                data.seq_num = data.seq_num % len(data.files)

                # save grads to file
                if config["vis"]["store_grads"]:
                    save_csv(grads_w, "grads_w.csv")
                    grads_w = []

                # finish training loop
                if data.epoch == config["loader"]["n_epochs"]:
                    end_train = True

                print("Sparsity:", end=" ")
                for key, value in sparsity_sum.items():
                    print(key, value.data/data_counter, end="  ")
                print("\n")

                sparsity_sum = {'0:input': 0, '1:head': 0, '2:G1': 0, '3:R1a': 0, '4:R1b': 0, '5:G2': 0, '6:R2a': 0, '7:R2b': 0, '8:pred': 0}
                
                data_counter = 0

                print("Learning rate in epoch {:02d}: {:.4e}".format(data.epoch, optimizer.state_dict()['param_groups'][0]['lr']))

            # forward pass
            x = model(inputs["event_voxel"].to(device), inputs["event_cnt"].to(device), log=True) # log=True for monitoring activation sparcity

            # event flow association
            loss_function.event_flow_association(
                x["flow"],
                inputs["event_list"].to(device),
                inputs["event_list_pol_mask"].to(device),
                inputs["event_mask"].to(device),
            )

            data_counter += 1

            sparsity = {}
            # apply sparsity regularizer 
            for key, value in x["acti_after_thresholding"].items():

                sparsity[key] = torch.count_nonzero(value.detach()) / torch.numel(value.detach()) # sparsity of each layer, the avg of this batch

                sparsity_sum[key] += sparsity[key]

            sparsity.pop("0:input")
            sparsity.pop("8:pred")

            max_density_layer = max(sparsity, key=sparsity.get) # find the densest layer and give it a larger weight in the regularizer

            if config["sparsify"]["regularize_voltage_enable"]:
                for key, voltage in x["activity"].items(): # one regularizer for each layer
                    if key != "0:input" and key != "8:pred":

                        if max_density_layer == key:
                            reg_weight_layer = 2.0
                        else:
                            reg_weight_layer = 1.0

                        if regularizer_voltage == 0:
                            regularizer_loss_voltage += reg_weight_layer * torch.pow(torch.sum(torch.abs(voltage)), 2) / (torch.sum(torch.pow(voltage, 2)) + 1e-5) # Hoyer sparsity regularizer ANN neuron states NOTE it is applied to each layer. TODO each channel? all neurons in the network?
                        elif regularizer_voltage == 1:
                            regularizer_loss_voltage += reg_weight_layer * torch.sum(torch.abs(voltage)) # L1 sparsity regularizer
                        else:
                            regularizer_loss_voltage += reg_weight_layer * torch.sum(torch.pow(voltage, 2)) # L2 sparsity regularizer
                        # print(key)

            if config["sparsify"]["regularize_threshold_enable"]:
                # to make network sparse, push threshold (SNN firing thre or ANN FATReLU shre) to be big NOTE
                if "LIF" in config["model"]["name"]:
                    threshold_name = 'thresh' # SNN
                else:
                    threshold_name = 'thre_' # ANN
                for name, param in model.named_parameters():
                    if threshold_name in name:
                        if name.split('.')[0] == max_density_layer.split(':')[1]:
                            reg_weight_layer = 2.0
                        else:
                            reg_weight_layer = 1.0
                        if regularizer_threshold == 2: # currently we only use L2 regularizer for threshold
                            regularizer_loss_thresh += reg_weight_layer * torch.sum(torch.pow(1.0/(torch.abs(param)+1e-9), 2)) # L2 regularizer_thresh
                        elif regularizer_threshold == 0:
                            regularizer_loss_thresh += reg_weight_layer * torch.pow(torch.sum(1.0/torch.abs(param)), 2) / (torch.sum(torch.pow((1.0/torch.abs(param)), 2)) + 1e-5) # Hoyer regularizer_thresh

            # backward pass
            if loss_function.num_events >= config["data"]["window_loss"]:

                # overwrite intermediate flow estimates with the final onesW
                if config["loss"]["overwrite_intermediate"]:
                    loss_function.overwrite_intermediate_flow(x["flow"])

                regularizer_loss = regularizer_loss_voltage * regularizer_weight_voltage + regularizer_loss_thresh * regularizer_weight_threshold

                if regularizer_loss != 0:
                    regularizer_print += regularizer_loss.item() / 10

                regularizer_loss = regularizer_loss / (10 * config["loader"]["batch_size"]) # the avg loss for each sample (sum of all layers). 10 is the number of frames before backward pass.

                # loss
                deblur_loss = loss_function()
                loss = deblur_loss + regularizer_loss
                train_loss += deblur_loss.item()
                
                # update number of loss samples seen by the network
                data.samples += config["loader"]["batch_size"]

                loss.backward()

                regularizer_loss_voltage = 0
                regularizer_loss_thresh = 0

                # clip and save grads
                if config["loss"]["clip_grad"] is not None:
                    torch.nn.utils.clip_grad.clip_grad_norm_(model.parameters(), config["loss"]["clip_grad"])
                if config["vis"]["store_grads"]:
                    grads_w.append(get_grads(model.named_parameters()))

                optimizer.step()
                if use_OneCycleLR_scheduler:
                    lr_scheduler.step()

                optimizer.zero_grad()

                # mask flow for visualization
                flow_vis = x["flow"][-1].clone()
                if model.mask and config["vis"]["enabled"] and config["loader"]["batch_size"] == 1:
                    flow_vis *= loss_function.event_mask

                model.detach_states()
                loss_function.reset()

                # visualize
                with torch.no_grad():
                    if config["vis"]["enabled"] and config["loader"]["batch_size"] == 1:
                        vis.update(inputs, flow_vis, None)

            # print training info
            if config["vis"]["verbose"]:
                if config["model"]["train_FATReLU_thre_enabled"]:
                    print(
                        "Train Epoch: {:04d} [{:03d}/{:03d} ({:03d}%)]  Loss: {:.6f}  Reg: {:.3f}  FATReLU_t: {:.6f}  {:.6f}  {:.6f}  {:.6f}  {:.6f}  {:.6f}  {:.6f}".format(
                            data.epoch,
                            data.seq_num,
                            len(data.files),
                            int(100 * data.seq_num / len(data.files)),
                            train_loss / (data.samples + 1),
                            regularizer_print / (data.samples + 1),
                            torch.mean(model.thre_x1),
                            torch.mean(model.thre_x2),
                            torch.mean(model.thre_x3),
                            torch.mean(model.thre_x4),
                            torch.mean(model.thre_x5),
                            torch.mean(model.thre_x6),
                            torch.mean(model.thre_x7)
                        ),
                        end="\r",
                    )
                else:
                    print(
                        "Train Epoch: {:04d} [{:03d}/{:03d} ({:03d}%)]  DeblurLoss: {:.6f}  Reg: {:.3f} ".format(
                            data.epoch,
                            data.seq_num,
                            len(data.files),
                            int(100 * data.seq_num / len(data.files)),
                            train_loss / (data.samples + 1),
                            regularizer_print / (data.samples + 1)
                        ),
                        end="\r",
                    )
        if end_train:
            break

    mlflow.end_run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/train_flow.yml",
        help="training configuration",
    )
    parser.add_argument(
        "--path_mlflow",
        default="",
        help="location of the mlflow ui",
    )
    parser.add_argument(
        "--prev_runid",
        default="",
        help="pre-trained model to use as starting point",
    )
    parser.add_argument(
        "--regularizer_weight_voltage",
        default="",
        help="",
    )
    parser.add_argument(
        "--regularizer_weight_threshold",
        default="",
        help="",
    )
    args = parser.parse_args()

    # launch training
    train(args, YAMLParser(args.config))