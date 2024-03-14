import os
import torch
import numpy as np

import pickle

# class AverageMeter(object):
#     """Computes and stores the average and current value"""

#     def __init__(obj):
#         obj.reset()

#     def reset(obj):
#         obj.val = 0
#         obj.avg = 0
#         obj.sum = 0
#         obj.count = 0

#     def update(obj, val, n=1):
#         obj.val = val
#         obj.sum += val * n
#         obj.count += n
#         obj.avg = obj.sum / obj.count

#     def __repr__(obj):
#         # return '{:.3e} (avg: {:.3e})'.format(obj.val, obj.avg)
#         return 'avg: {:.3e}'.format(obj.avg)

class Activation_Statistic:

    def __init__(self, num_layers=9): # including rnn hidden states
        
        self.num_layers = num_layers
        self.layer_list = [[]] * num_layers
        # print(self.layer_list)
        self.layer_idx = 0

        for i in range(num_layers):
            if i == 0 or i == num_layers-1:
                self.layer_list[i] = [torch.empty(0)] * 2
                # print(self.layer_list[i]) # [tensor([]), tensor([])]
            else:
                self.layer_list[i] = [torch.empty(0)] * 32

    def update(self, tensor): # tensor is the output of a layer with multiple channels

        # print(self.layer_idx)

        for i in range(tensor.size()[1]): # i is the channel index
            feature_map = tensor[:,i,:,:]
            # print(feature_map)
            spikes = feature_map[feature_map>0]
            self.layer_list[self.layer_idx][i] = torch.cat((self.layer_list[self.layer_idx][i], spikes))

        self.layer_idx += 1
        if self.layer_idx == self.num_layers:
            self.layer_idx = 0
        # "0:input","1:head","2:G1","2:G1_state","3:R1a","4:R1b","5:G2","5:G2_state","6:R2a","7:R2b","8:pred"

    def save(self, path): 
        for layer_index in range(self.num_layers):
            for i in range(len(self.layer_list[layer_index])): # i is the channel index
                spikes_arr = self.layer_list[layer_index][i].numpy()
                np.save(path+"/"+str(layer_index)+"_"+str(i), spikes_arr)

    # def load(self, path):
    #     for layer_index in range(self.num_layers):
    #         for i in range(len(self.layer_list[layer_index])): # i is the channel index
    #             spikes_arr = self.layer_list[layer_index][i].numpy()
    #             np.save(path+"/"+str(layer_index)+"_"+str(i), spikes_arr)

    def __call__(self, layer_index, channel_index):
        return self.layer_list[layer_index][channel_index]
    
    # def __repr__(self):
    #     return # self.layer_list

# def activation_statistic(layer_dict):

#     assert(len(layer_dict)==9)
#     # average_activation_channel

#     for key, value in layer_dict.items():
#         print(value.size())
#         for i in range(value.size()[1]):
#             feature_map = value[:,i,:,:]
#             # print(feature_map)
#             spike_tensor = feature_map[feature_map>0]
#             print(spike_tensor.size())



#     dddd
#     return 
