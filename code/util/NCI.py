from typing import Any
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import util.openood_comm as comm


def get_output(net, data):
    net.to(torch.device("cuda"))
    data = data.cpu()
    dataset = TensorDataset(data)
    loader = DataLoader(dataset, batch_size=128, shuffle=False)
    outs, features = [], []
    with torch.no_grad():
        for d in loader:
            d = d[0].cuda()
            out, feature = net(d, return_feature=True)
            outs.append(out.detach().cpu())
            features.append(feature.detach().cpu())
    outs = torch.cat(outs, dim=0)
    features = torch.cat(features, dim=0)
    return outs, features


class BasePostprocessor:
    def __init__(self, config):
        self.config = config

    def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict):
        pass

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        output = net(data)
        score = torch.softmax(output, dim=1)
        conf, pred = torch.max(score, dim=1)
        return pred, conf

    def inference(self,
                  net: nn.Module,
                  data_loader: DataLoader,
                  progress: bool = True):
        pred_list, conf_list, label_list = [], [], []
        for batch in tqdm(data_loader,
                          disable=not progress or not comm.is_main_process()):
            data = batch['data'].cuda()
            label = batch['label'].cuda()
            pred, conf = self.postprocess(net, data)

            pred_list.append(pred.cpu())
            conf_list.append(conf.cpu())
            label_list.append(label.cpu())

        # convert values into numpy array
        pred_list = torch.cat(pred_list).numpy().astype(int)
        conf_list = torch.cat(conf_list).numpy()
        label_list = torch.cat(label_list).numpy().astype(int)

        return pred_list, conf_list, label_list


class NCIPostprocessor(BasePostprocessor):
    def __init__(self, config):
        super(NCIPostprocessor, self).__init__(config)
        self.APS_mode = True
        self.setup_flag = False
        self.train_mean = None
        self.w = None
        self.alpha = 0.0001
        self.activation_log = None

        # self.args = self.config.postprocessor.postprocessor_args
        # self.alpha = self.args.alpha
        # self.args_dict = self.config.postprocessor.postprocessor_sweep

    def setup(self, net: nn.Module, id_loader, ood_loader_dict):
        if not self.setup_flag:
            # collect training mean
            activation_log = []
            net.eval()
            with torch.no_grad():
                for data, label in id_loader:
                    # data = batch['data'].cuda()
                    # data = data.float()
                    data = data.cuda()
                    _, feature = net(data, return_feature=True)

                    activation_log.append(feature.data.cpu().numpy())

            activation_log_concat = np.concatenate(activation_log, axis=0)
            self.activation_log = activation_log_concat
            self.train_mean = torch.from_numpy(
                np.mean(activation_log_concat, axis=0))

            # compute denominator matrix
            for i, param in enumerate(net.fc.parameters()):
                if i == 0:
                    self.w = param.data

            self.setup_flag = True

        else:
            pass

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        # output, feature = net(data, return_feature=True)
        output, feature = get_output(net, data)
        values, nn_idx = output.max(1)
        self.w = self.w.cpu()
        score = torch.sum(self.w[nn_idx] * (feature - self.train_mean), axis=1) / torch.norm(feature - self.train_mean,
                                                                                             dim=1) + self.alpha * torch.norm(
            feature, p=1, dim=1)
        return nn_idx, score

    def set_hyperparam(self, hyperparam: list):
        self.alpha = hyperparam[0]

    def get_hyperparam(self):
        return self.alpha
