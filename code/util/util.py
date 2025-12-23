import hashlib

import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader


def adjust_lr(optimizer, target_lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = target_lr
    return optimizer


# class TrainScheduler(torch.optim.lr_scheduler._LRScheduler):
#     def __init__(self, optimizer, lr):
#         self.optimizer = optimizer
#         self.lr = lr
#         super().__init__(optimizer, -1)
#
#     def get_lr(self):
#         self.lr = self.lr / 10
#         return self.lr


def balance_dataset(train_set, test_set):
    train_data = train_set.data
    test_data = test_set.data

    train_label = train_set.label
    test_label = test_set.label

    train_data_ls = []
    test_data_ls = []
    train_label_ls = []
    test_label_ls = []

    for i in range(100):
        mask = train_label == i
        train_d = train_data[mask]
        train_l = train_label[mask]

        mask = test_label == i
        test_d = test_data[mask]
        test_l = test_label[mask]

        length = min(len(train_d), len(test_d))

        train_data_ls.append(train_d[:length])
        test_data_ls.append(test_d[:length])

        train_label_ls.append(train_l[:length])
        test_label_ls.append(test_l[:length])

    train_data_ls = torch.row_stack(train_data_ls)
    test_data_ls = torch.row_stack(test_data_ls)

    train_label_ls = torch.cat(train_label_ls)
    test_label_ls = torch.cat(test_label_ls)

    train_set.data = train_data_ls
    test_set.data = test_data_ls

    train_set.label = train_label_ls
    test_set.label = test_label_ls

    return train_set, test_set


def unique(dataset, known_features_num):
    data = dataset.data
    label = dataset.label
    data_prefix = data[:, :known_features_num].numpy()
    data_prefix = data_prefix.astype(str)
    data_prefix = np.apply_along_axis(lambda x: ''.join(x), axis=1, arr=data_prefix)
    # factor = np.asarray([2 ** i for i in range(known_features_num)])
    # data_prefix_hash = np.dot(data_prefix, factor)  # (n, 1)
    data_prefix_hash, index = np.unique(data_prefix, return_index=True, axis=0)
    data = data[index]
    label = label[index]
    dataset.data = data
    dataset.label = label
    return dataset


def get_grads(model, data_loader):
    grad_ls = []
    label_ls = []
    loss_fn = torch.nn.CrossEntropyLoss()
    for data, label in data_loader:
        data, label = data.cuda(), label.cuda()
        data.requires_grad = True
        model.zero_grad()
        out = model(data)
        loss = loss_fn(out, label)
        loss.backward()
        grad = data.grad.detach().cpu().numpy()
        grad_ls.append(grad)
        label_ls.append(label.detach().cpu().numpy())
    grad_ls = np.row_stack(grad_ls)
    label_ls = np.concatenate(label_ls)
    return grad_ls, label_ls


# def get_estimated_grads(model, data_loader, m=50, epsilon=0.1, n_classes=100):
#     model.eval()
#     grad_ls = []
#     label_ls = []
#     loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
#     with torch.no_grad():
#         for data, label in data_loader:
#             bs, dim = data.shape
#             u = np.random.randn(bs, m, dim)
#             d = np.sqrt(np.sum(u ** 2, axis=2)).reshape(-1, m, 1)
#             u = torch.Tensor(u / d)
#             u = torch.cat((u, torch.zeros(bs, 1, dim)), dim=1)
#             u = u.view(-1, m + 1, dim)
#             evaluation_points = (data.view(-1, 1, dim).cpu() + epsilon * u).view(-1, dim)
#             pred = model(evaluation_points.cuda()).detach().cpu()
#             new_label = label.reshape((-1, 1)).repeat(1, m + 1).reshape(-1, )
#             loss_values = loss_fn(pred, new_label).view(-1, m + 1)  # (m*bs,)
#             differences = loss_values[:, :-1] - loss_values[:, -1].view(-1, 1)
#             differences = differences.view(-1, m, 1)
#             gradient_estimates = (1 / epsilon) * differences * u[:, :-1]
#             gradient_estimates = gradient_estimates.mean(dim=1).view(-1, dim)  # / n_classes
#             grad_ls.append(gradient_estimates)
#             label_ls.append(label)
#         grad_ls = torch.row_stack(grad_ls).numpy()
#         label_ls = torch.cat(label_ls).numpy()
#     return grad_ls, label_ls


def thre_setting(tr_values, te_values):
    value_list = np.concatenate((tr_values, te_values))
    max_acc = 0
    thresh = 0
    for value in value_list:
        tp = np.sum(tr_values <= value)
        tn = np.sum(te_values > value)
        fp = np.sum(te_values <= value)
        fn = np.sum(tr_values > value)

        acc = (tp + tn) / (tp + tn + fp + fn)
        if acc > max_acc:
            max_acc = acc
            thresh = value
    return thresh, max_acc


def inference(train_mad_ls, test_mad_ls, thresh, reverse=False):
    tp = np.sum(train_mad_ls <= thresh)
    fp = np.sum(test_mad_ls <= thresh)
    tn = np.sum(test_mad_ls > thresh)
    fn = np.sum(train_mad_ls > thresh)
    if reverse:
        return fn, tn, fp, tp
    return int(tp), int(fp), int(tn), int(fn)


def generate_data(data_loader, model, known_features_num, generate_lr, generate_epochs=1, init="zero", front=False):
    # criterion = nn.MSELoss()
    criterion = torch.nn.CrossEntropyLoss(reduction="none")
    gen_datas = []
    gen_labels = []
    if init == "zero":
        init_func = torch.zeros_like
    elif init == "random":
        init_func = torch.randn_like
    elif init == "one":
        init_func = torch.ones_like
    else:
        raise BaseException("undefined init function")

    for data, label in tqdm(data_loader):
        data[:, known_features_num:] = init_func(data[:, known_features_num:])
        # optimizer = optim.SGD([data], lr=generate_lr)
        data, label = data.cuda(), label.cuda()
        for epoch in range(generate_epochs):
            data = data.detach().clone()
            model.zero_grad()
            data.requires_grad = True
            # optimizer.zero_grad()
            out = model(data)
            loss = criterion(out, label)
            loss.backward(torch.ones_like(loss))
            with torch.no_grad():
                if front:
                    data.grad[:, known_features_num:] = 0
                    data = data + generate_lr * data.grad
                else:
                    data.grad[:, :known_features_num] = 0
                    data = data - generate_lr * data.grad
        gen_datas.append(data.detach().cpu())
        gen_labels.append(label.detach().cpu())
    return torch.row_stack(gen_datas), torch.cat(gen_labels)


def generate_data_maskd(data_loader, model, generate_lr, generate_epochs=1, init="zero", mask=None, grad_norm=False):
    # criterion = nn.MSELoss()
    criterion = torch.nn.CrossEntropyLoss(reduction="none")
    gen_datas = []
    gen_labels = []
    if init == "zero":
        init_func = torch.zeros_like
    elif init == "random":
        init_func = torch.randn_like
    elif init == "one":
        init_func = torch.ones_like
    else:
        raise BaseException("undefined init function")

    pos_k = torch.where(~mask)
    pos_u = torch.where(mask)
    for data, label in tqdm(data_loader):
        if len(mask.shape) > 1:
            data[:, :, pos_u[0], pos_u[1]] = init_func(data)[:, :, pos_u[0], pos_u[1]]
        else:
            data[:, pos_u[0]] = init_func(data)[:, pos_u[0]]
        # data[:, known_features_num:] = init_func(data[:, known_features_num:])
        # optimizer = optim.SGD([data], lr=generate_lr)
        data, label = data.cuda(), label.cuda()
        for epoch in range(generate_epochs):
            data = data.detach().clone()
            model.zero_grad()
            data.requires_grad = True
            # optimizer.zero_grad()
            out = model(data)
            loss = criterion(out, label)
            loss.backward(torch.ones_like(loss))
            with torch.no_grad():
                if grad_norm:
                    data.grad = data.grad / torch.norm(data.grad, dim=list(range(1, len(data.shape))), p=2,
                                                       keepdim=True)
                if len(mask.shape) > 1:
                    data.grad[:, :, pos_k[0], pos_k[1]] = 0
                else:
                    data.grad[:, pos_k[0]] = 0

                data = data - generate_lr * data.grad
        gen_datas.append(data.detach().cpu())
        gen_labels.append(label.detach().cpu())
    return torch.row_stack(gen_datas), torch.cat(gen_labels)


def generate_data_black_box(data_loader, model, generate_lr, generate_epochs=1, init="zero", mask=None,
                            grad_norm=False):
    gen_datas = []
    gen_labels = []
    if init == "zero":
        init_func = torch.zeros_like
    elif init == "random":
        init_func = torch.randn_like
    elif init == "one":
        init_func = torch.ones_like
    else:
        raise BaseException("undefined init function")

    pos_k = torch.where(~mask)
    pos_u = torch.where(mask)
    with torch.no_grad():
        for data, label in tqdm(data_loader):
            if len(mask.shape) > 1:
                data[:, :, pos_u[0], pos_u[1]] = init_func(data)[:, :, pos_u[0], pos_u[1]]
            else:
                data[:, pos_u[0]] = init_func(data)[:, pos_u[0]]
            data, label = data.cuda(), label.cuda()
            for epoch in range(generate_epochs):
                # zoro-order grad estimation
                with torch.no_grad():
                    grad = zero_order_grad_estimation(model, data, label, num_directions=100, eps=0.01)
                    if grad_norm:
                        grad = grad / torch.norm(grad, dim=list(range(1, len(data.shape))), p=2,
                                                 keepdim=True)
                    if len(mask.shape) > 1:
                        grad[:, :, pos_k[0], pos_k[1]] = 0
                    else:
                        grad[:, pos_k[0]] = 0

                    data = data - generate_lr * grad
            gen_datas.append(data.detach().cpu())
            gen_labels.append(label.detach().cpu())
    return torch.row_stack(gen_datas), torch.cat(gen_labels)


def zero_order_grad_estimation(model, data, label, num_directions=1000, eps=1e-1):
    # print("zero grad optimzation")
    loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
    shape = data.shape
    batch_size = shape[0]
    dim = data[0].numel()  # flatten 后的特征维度

    # 展平为 (batch, dim)
    data_flat = data.reshape(batch_size, -1)

    # 随机方向 (m, dim)
    u = torch.randn(num_directions, dim)
    u = u / (u.norm(dim=1, keepdim=True) + 1e-12)  # 单位化方向

    # 扩展维度 (batch, m, dim)
    x_expand = data_flat.unsqueeze(1).cpu()  # (batch, 1, dim)
    u_expand = u.unsqueeze(0).expand(batch_size, -1, -1)  # (batch, m, dim)

    # f(x)
    out = model(data)
    out = loss_fn(out, label.long().cuda()).cpu()  # (batch, 1)

    # 显存不够
    dataset = TensorDataset(x_expand, u_expand, label)
    loader = DataLoader(dataset, batch_size=128, shuffle=False)

    out_collection = []
    for x_epd, u_epd, l in loader:
        x_epd = x_epd.cuda()
        u_epd = u_epd.cuda()
        x_eps = (x_epd + eps * u_epd).reshape(-1, dim)  # (batch*m, dim)
        x_eps_reshaped = x_eps.reshape(-1, *data.shape[1:])  # 还原原始输入形状
        out_eps = model(x_eps_reshaped).cpu()

        l = l.reshape((-1, 1))
        l = l.expand(-1, num_directions).reshape((-1,)).cpu()
        out_eps = loss_fn(out_eps, l.long()).reshape(x_epd.shape[0], num_directions)  # (batch, m)
        out_collection.append((out_eps.detach().cpu()))
    out_collection = torch.cat(out_collection, dim=0)
    diff = out_collection - out.reshape((-1, 1))
    diff = (diff / eps).unsqueeze(-1)
    grad_est = (dim * (diff * u_expand)).mean(dim=1)
    grad_est = grad_est.reshape(shape)
    return grad_est.cuda()


def generate_data_shap(data_loader, model, generate_lr, generate_epochs=1, init="zero", grad_norm=False):
    # criterion = nn.MSELoss()
    criterion = torch.nn.CrossEntropyLoss(reduction="none")
    gen_datas = []
    gen_labels = []
    if init == "zero":
        init_func = torch.zeros_like
    elif init == "random":
        init_func = torch.randn_like
    elif init == "one":
        init_func = torch.ones_like
    else:
        raise BaseException("undefined init function")


    # pos_k = torch.where(~mask)
    # pos_u = torch.where(mask)
    # flag = True
    for data, label, mask in tqdm(data_loader):
        if len(mask.shape) == 3:
            mask = mask.unsqueeze(1)  # (B, 1, H, W)
        if data.shape[1] == 3 and mask.shape[1] == 1:
            mask = mask.repeat(1, 3, 1, 1)

        # if len(mask.shape) > 1:
        #     data[:, :, pos_u[0], pos_u[1]] = init_func(data)[:, :, pos_u[0], pos_u[1]]
        # else:
        #     data[:, pos_u[0]] = init_func(data)[:, pos_u[0]]
        data[mask] = init_func(data)[mask]
        data, label = data.cuda(), label.cuda()
        for epoch in range(generate_epochs):
            data = data.detach().clone()
            model.zero_grad()
            data.requires_grad = True
            # optimizer.zero_grad()
            out = model(data)
            loss = criterion(out, label)
            loss.backward(torch.ones_like(loss))
            with torch.no_grad():
                data.grad[~mask] = 0
                # if len(mask.shape) > 1:
                #     data.grad[] = 0
                # else:
                #     data.grad[:, pos_k[0]] = 0
                if grad_norm:
                    data.grad = data.grad / torch.norm(data.grad, dim=list(range(1, len(data.shape))), keepdim=True)
                data = data - generate_lr * data.grad
        gen_datas.append(data.detach().cpu())
        gen_labels.append(label.detach().cpu())
    return torch.row_stack(gen_datas), torch.cat(gen_labels)


def cal_mad(train_data, train_label, test_data, test_label, other_data, other_label, classes=100):
    # if known_feature_num is not None:
    #     train_data, test_data, other_data = train_data[:, known_feature_num:], test_data[:,
    #                                                                            known_feature_num:], other_data[
    #                                                                                                 :,
    #                                                                                                 known_feature_num:]
    # 计算other集合中每个类别的质心
    center_ls = []
    other_dis_ls = []
    other_label_ls = []
    for i in range(classes):
        mask = other_label == i
        other_d = other_data[mask]
        other_l = other_label[mask]
        center = np.mean(other_d, axis=0)
        center_ls.append(center)
        dis = np.sqrt(np.sum(np.square((other_d - center)), axis=1))
        other_dis_ls.append(dis)
        other_label_ls.append([i] * len(other_d))

    other_dis_ls = np.concatenate(other_dis_ls)  # (len(other),)
    other_label_ls = np.concatenate(other_label_ls)

    center_ls = np.row_stack(center_ls)  # texas100 (100,6169) ; purchase100 (100,600)
    # 计算每个gen_train和gen_test样本和质心的距离
    train_dis_mean = []
    test_dis_mean = []

    train_dis_ls = []
    train_label_ls = []

    test_dis_ls = []
    test_label_ls = []
    for i in range(classes):
        mask1 = train_label == i
        mask2 = test_label == i

        train_d = train_data[mask1]
        test_d = test_data[mask2]

        center = center_ls[i]
        train_dis = np.sqrt(np.sum(np.square((train_d - center)), axis=1))  # (sum(mask),1)
        test_dis = np.sqrt(np.sum(np.square((test_d - center)), axis=1))

        train_dis_ls.append(train_dis)
        test_dis_ls.append(test_dis)
        train_label_ls.append([i] * len(train_d))
        test_label_ls.append([i] * len(test_d))

        train_dis_mean.append(np.mean(train_dis))
        test_dis_mean.append(np.mean(test_dis))
    train_dis_mean = np.asarray(train_dis_mean)
    test_dis_mean = np.asarray(test_dis_mean)
    train_dis_ls = np.concatenate(train_dis_ls)
    test_dis_ls = np.concatenate(test_dis_ls)
    train_label_ls = np.concatenate(train_label_ls)
    test_label_ls = np.concatenate(test_label_ls)

    # 比较质心距离的差距。需要计算other集合中每个样本和质心之间的距离
    '''
    如何比较
    1. 根据每个类别的std来决定
    2. MAD（参考CADE）
    '''
    MAD_ls = []
    train_mad_ls = []
    test_mad_ls = []
    for i in range(classes):
        mask = other_label_ls == i
        other_dis = other_dis_ls[mask]
        i_median = np.median(other_dis)
        diff_median = np.fabs(other_dis - i_median)
        mad = np.median(diff_median)
        MAD_ls.append(mad)

        mask1 = train_label_ls == i
        train_dis = train_dis_ls[mask1]
        train_mad = np.fabs(train_dis - i_median) / (mad + 1e-6)
        train_mad_ls.append(train_mad)

        mask2 = test_label_ls == i
        test_dis = test_dis_ls[mask2]
        test_mad = np.fabs(test_dis - i_median) / (mad + 1e-6)
        test_mad_ls.append(test_mad)

    train_mad_ls = np.concatenate(train_mad_ls)
    test_mad_ls = np.concatenate(test_mad_ls)
    return train_mad_ls, test_mad_ls


def compare_score(train_score, test_score, val_score, thresh=1):
    train_energy = np.array([int(np.sum((train_s / val_score) > thresh)) for train_s in train_score])
    test_energy = np.array([int(np.sum((test_s / val_score) > thresh)) for test_s in test_score])
    return train_energy, test_energy


def generate_mask(shape=(32, 32), unknown_num=324):
    """生成包含372个True的固定数量掩码"""
    if len(shape) > 1:
        mask = torch.zeros(shape, dtype=torch.bool)
        indices = torch.randperm(shape[0] * shape[1])[:unknown_num]  # 随机选择372个位置
    else:
        mask = torch.zeros(shape, dtype=torch.bool)
        indices = torch.randperm(shape[0])[:unknown_num]
    mask.view(-1)[indices] = True
    return mask


def get_ds_param(dataset):
    if dataset == "texas100":
        classes = 100
        mask_shape = (6169,)
        f_num = 6169
    elif dataset == "purchase100":
        classes = 100
        mask_shape = (600,)
        f_num = 600
    elif dataset == "adult":
        classes = 2
        mask_shape = (10,)
        f_num = 10
    elif dataset == "cifar10":
        classes = 10
        mask_shape = (32, 32)
        f_num = 32 * 32
    elif dataset == "cifar100":
        classes = 100
        mask_shape = (32, 32)
        f_num = 32 * 32
    elif dataset == "fcifar10":
        classes = 10
        mask_shape = (32 * 32 * 3,)
        f_num = 32 * 32 * 3

    elif dataset == "fashion":
        classes = 10
        mask_shape = (28, 28)
        f_num = 28 * 28

    elif dataset == "stl10":
        classes = 10
        mask_shape = (96, 96)
        f_num = 96 * 96
    elif dataset == "epsilon":
        classes = 2
        mask_shape = (2000,)
        f_num = 2000
    elif dataset == "diabetic":
        classes = 2
        mask_shape = (44,)
        f_num = 44
    else:
        raise Exception("unknown dataset")

    return None, classes, mask_shape, f_num


def get_pred_vec(model, data):
    with torch.no_grad():
        # return torch.softmax(model(data), dim=1)
        return model(data)
