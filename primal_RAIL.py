import clip
import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
import random
import numpy as np
import os
import yaml
from easydict import EasyDict

from scenario_datasets import build_dataset
from scenario_datasets.merged_dataset import MergedDataset
from scenario_datasets.utils import build_data_loader
from scenario_datasets.collections import CIFAR100, MNIST
from utils import *

class continual_clip_adaptor(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.clip_model, self.train_preprocess, self.val_preprocess = clip.load(cfg.backbone, device=cfg.device, jit=False)
        self.feature_dim = 512  # related to backbone encoder
        self.analytic_adaptor = None
        self.hidden_dim = cfg.hidden_dim
        self.expansion_layer = nn.Sequential(
            nn.Linear(self.feature_dim, self.hidden_dim, bias=False, dtype=torch.float),
            nn.ReLU()
            ).to(cfg.device)
        self.clip_model.eval()

    def encode_images(self, images):
        features = self.clip_model.encode_image(images)
        features /= features.norm(dim=-1, keepdim=True)  # normalization to unit vector
        return features

    def analytic_adaption(self, task_id, cfg, train_loader, R):
        if task_id == 0:
            if cfg.feature_expansion:
                # initialize adaptor layer based on 1st dataset class number
                self.analytic_adaptor = nn.Linear(self.hidden_dim, cfg.current_class_num, bias=False).to(cfg.device)
                auto_cor = torch.zeros(self.hidden_dim, self.hidden_dim).to(cfg.device)
                crs_cor = torch.zeros(self.hidden_dim, cfg.current_class_num).to(cfg.device)
            else:
                self.analytic_adaptor = nn.Linear(self.feature_dim, cfg.current_class_num, bias=False).to(cfg.device)
                auto_cor = torch.zeros(self.feature_dim, self.feature_dim).to(cfg.device)
                crs_cor = torch.zeros(self.feature_dim, cfg.current_class_num).to(cfg.device)

            # first task: initialize R_0
            with torch.no_grad():
                for i, (images, target) in \
                        enumerate(tqdm(train_loader, desc=f'Re-Alignment on task-{task_id + 1}', total=len(train_loader),
                                       unit='batch')):
                    images, target = images.to(cfg.device), target.to(cfg.device)
                    train_features = self.encode_images(images)

                    if cfg.feature_expansion:
                        train_features = self.expansion_layer(train_features)

                    train_labels_one_hot = F.one_hot(target, cfg.current_class_num).float()

                    auto_cor += torch.t(train_features) @ train_features
                    crs_cor += torch.t(train_features) @ (train_labels_one_hot)

            R = np.mat(auto_cor.cpu().numpy() + cfg.regularization * np.eye(train_features.size(1))).I
            R = torch.tensor(R).float().to(cfg.device)

            Delta = R @ crs_cor
            self.analytic_adaptor.weight = torch.nn.parameter.Parameter(torch.t(1.0 * Delta.float()))
            return R

        else:
            # Recursively solving R_t
            w = self.analytic_adaptor.weight.t()
            if cfg.feature_expansion:
                w = torch.cat([w, torch.zeros(self.hidden_dim, cfg.increment).to(cfg.device)], dim=1)
                self.analytic_adaptor = nn.Linear(self.hidden_dim, cfg.current_class_num, bias=False).to(cfg.device)
            else:
                w = torch.cat([w, torch.zeros(self.feature_dim, cfg.increment).to(cfg.device)], dim=1)
                self.analytic_adaptor = nn.Linear(self.feature_dim, cfg.current_class_num, bias=False).to(cfg.device)

            with torch.no_grad():
                for i, (images, target) in \
                        enumerate(tqdm(train_loader, desc=f'Re-Alignment on task-{task_id + 1}', total=len(train_loader),
                                       unit='batch')):
                    target += cfg.trained_class_num
                    images, target = images.to(cfg.device), target.to(cfg.device)
                    train_features = self.encode_images(images)

                    if cfg.feature_expansion:
                        train_features = self.expansion_layer(train_features)

                    train_labels_one_hot = F.one_hot(target, cfg.current_class_num).float()

                    R = R - R @ train_features.t() @ torch.pinverse(torch.eye(images.size(0)).to(cfg.device) +
                                                                    train_features @ R @ train_features.t()) @ train_features @ R
                    w = w + R @ train_features.t() @ (train_labels_one_hot - train_features @ w)

            self.analytic_adaptor.weight = torch.nn.parameter.Parameter(torch.t(w.float()))
            return R

    def forward(self, images):
        features = self.encode_images(images)
        if cfg.feature_expansion:
            features = self.expansion_layer(features)
        outputs = self.analytic_adaptor(features)
        return outputs

    def zero_shot(self, images, clip_weights):
        features = self.encode_images(images)
        clip_logits = 100. * features @ clip_weights

        return clip_logits


def test_acc(test_loader, cfg):
    top1, top5, test_num = 0.0, 0.0, 0.0
    for inputs, targets in tqdm(test_loader, desc='Evaluating on current dataset',
                                total=len(test_loader), unit='batch'):
        targets += cfg.previous_class_num
        inputs, targets = inputs.to(cfg.device), targets.to(cfg.device)
        with torch.no_grad():
            outputs = continual_clip_adaptor(inputs)
        acc1, acc5 = cls_acc(outputs, targets, topk=(1, 5))
        top1 += acc1
        top5 += acc5
        test_num += inputs.size(0)

    top1, top5 = (top1 / test_num) * 100, (top5 / test_num) * 100
    return top1, top5


cfg_file = "configs/analytic_clip.yaml"
cfg = yaml.load(open(cfg_file, 'r'), Loader=yaml.Loader)
cfg = EasyDict(cfg)

seed = cfg.seed
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)
cfg.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

dataset_sequence = cfg.datasets
print("Multi-task dataset sequence: ", dataset_sequence)

# Results
fusion_acc_table = np.zeros((len(dataset_sequence), len(dataset_sequence)))
adapter_acc_table = np.zeros((len(dataset_sequence), len(dataset_sequence)))
in_domain_acc_list = []

cfg.previous_class_num = 0
current_class_names = []
R = None

merged_classnames = []
for _, train_dataset in enumerate(dataset_sequence):
    if train_dataset == "cifar100":
        dataset = CIFAR100(num_shots=-1, preprocess=None, val_transform=None, batch_size=cfg.batch_size)
    elif train_dataset == "mnist":
        dataset = MNIST(num_shots=-1, preprocess=None, val_transform=None, batch_size=cfg.batch_size)
    else:
        dataset = build_dataset(train_dataset, os.path.join(DIR_PATH, 'datasets'), cfg.num_shots)
    merged_classnames += dataset.classnames
    print(len(dataset.classnames))
print(f'Size of cross-domain category set: {len(merged_classnames)}')

"""
Loading model
"""
print('Loading pretrained CLIP model...')

continual_clip_adaptor = continual_clip_adaptor(cfg)
continual_clip_adaptor.clip_model.eval()

train_transform = continual_clip_adaptor.train_preprocess
val_preprocess = continual_clip_adaptor.val_preprocess

"""
Training on dataset sequence
"""
for task_id, train_dataset in enumerate(dataset_sequence):
    print(f"------------------ Start training on task-{task_id + 1}: dataset-{train_dataset}. ---------------------")

    if train_dataset == "cifar100":
        dataset = CIFAR100(num_shots=cfg.num_shots, preprocess=train_transform, val_transform=val_preprocess,
                           batch_size=cfg.batch_size)
    elif train_dataset == "mnist":
        dataset = MNIST(num_shots=cfg.num_shots, preprocess=train_transform, val_transform=val_preprocess,
                        batch_size=cfg.batch_size)
    else:
        dataset = build_dataset(train_dataset, os.path.join(DIR_PATH, 'datasets'), cfg.num_shots)

    current_class_names += dataset.classnames
    cfg.increment = len(dataset.classnames)
    cfg.current_class_num = len(current_class_names)

    print(f"Currently existing a total of {cfg.current_class_num} classes.")

    if train_dataset == "cifar100" or train_dataset == "mnist":
        train_loader = dataset.train_loader
    else:
        train_loader = build_data_loader(data_source=dataset.train_x, batch_size=cfg.batch_size, tfm=train_transform,
                                         is_train=True, shuffle=True, augmentation_time=cfg.augmentation_time)
    R = continual_clip_adaptor.analytic_adaption(task_id, cfg, train_loader, R)

    cfg.trained_class_num = cfg.current_class_num

    """
    Testing stage: test on every dataset (both trained & untrained) after training on each dataset
    """
    if cfg.eval_last and task_id < len(dataset_sequence) - 1:
        continue

    tested_cls_num = 0
    for test_id, test_dataset in enumerate(dataset_sequence):
        if cfg.eval_adapter and test_id > task_id:
            continue

        print(f"Evaluating on dataset-{test_id + 1}: {test_dataset}")
        if test_dataset == "cifar100":
            test_set = CIFAR100(num_shots=-1, preprocess=None, val_transform=val_preprocess, batch_size=cfg.batch_size)
        elif test_dataset == "mnist":
            test_set = MNIST(num_shots=-1, preprocess=None, val_transform=val_preprocess, batch_size=cfg.batch_size)
        else:
            test_set = build_dataset(test_dataset, os.path.join(DIR_PATH, 'datasets'), cfg.num_shots)

        if test_dataset == "cifar100" or test_dataset == "mnist":
            test_loader = test_set.test_loader
        else:
            test_loader = build_data_loader(data_source=test_set.test, batch_size=cfg.batch_size, is_train=False,
                                            tfm=val_preprocess, shuffle=False)

        template = ['a photo of a {}.']
        clip_weights = clip_classifier(merged_classnames, template, continual_clip_adaptor.clip_model, device=cfg.device)

        class_range_min, class_range_max = tested_cls_num, tested_cls_num + len(test_set.classnames)
        in_domain, in_domain_acc = 0.0, 0.0
        adapter_in_domain, adapter_in_domain_acc = 0.0, 0.0

        top1, top5, test_num = 0.0, 0.0, 0.0
        fusion_top1, fusion_top5 = 0.0, 0.0
        adapt_top1, adapt_top5 = 0.0, 0.0

        for inputs, targets in tqdm(test_loader, desc=f'Evaluating on dataset-{test_id + 1}: {test_dataset}',
                                    total=len(test_loader), unit='batch'):
            test_num += inputs.size(0)

            inputs, targets = inputs.to(cfg.device), targets.to(cfg.device)
            targets += tested_cls_num

            with torch.no_grad():
                outputs = continual_clip_adaptor.zero_shot(inputs, clip_weights)  # (B, C_all)
                outputs = F.softmax(outputs, dim=-1)

            # In domain acc
            predict_cls = torch.argmax(outputs, dim=-1)
            in_range = (predict_cls >= class_range_min) & (predict_cls < class_range_max)
            in_domain += torch.sum(in_range)

            # Zero-shot acc
            acc1, acc5 = cls_acc(outputs, targets, topk=(1, 5))
            top1 += acc1
            top5 += acc5

            # Select samples that belong to learned domains determined by CLIP zero-shot
            mask = predict_cls < cfg.current_class_num  # if less than trained class number
            if torch.sum(mask) > 0:
                samples_to_adapt = inputs[mask]
                with torch.no_grad():
                    outputs_adapted = continual_clip_adaptor(samples_to_adapt)

                padding_right = outputs.size(-1) - outputs_adapted.size(-1)
                outputs_adapted = F.pad(outputs_adapted, pad=(0, padding_right, 0, 0), mode='constant', value=0)

                adapter_pred = torch.argmax(outputs_adapted, dim=-1)
                adapter_in_range = (adapter_pred >= class_range_min) & (adapter_pred < class_range_max)
                adapter_in_domain += torch.sum(adapter_in_range)

                outputs[mask] = (1-cfg.fusion_weight) * outputs[mask] + cfg.fusion_weight * outputs_adapted

            # Fusion acc
            fusion_acc1, fusion_acc5 = cls_acc(outputs, targets, topk=(1, 5))
            fusion_top1 += fusion_acc1
            fusion_top5 += fusion_acc5

            if test_id <= task_id:
                with torch.no_grad():
                    outputs = continual_clip_adaptor(inputs)
                adapt_acc1, adapt_acc5 = cls_acc(outputs, targets, topk=(1, 5))
                adapt_top1 += adapt_acc1
                adapt_top5 += adapt_acc5

        if test_id <= task_id:
            pure_adapter_acc = (adapt_top1 / test_num) * 100
            print(f"Pure adapter acc for dataset-{test_id + 1}: {test_dataset}: {pure_adapter_acc}")
            adapter_acc_table[task_id, test_id] = pure_adapter_acc

        in_domain_acc = (in_domain / test_num) * 100
        print(f"In-domain top-1 acc for dataset-{test_id + 1}: {test_dataset}: {in_domain_acc}")

        top1, top5 = (top1 / test_num) * 100, (top5 / test_num) * 100
        print(f"Zero-shot top-1 acc for dataset-{test_id + 1}: {test_dataset}: {top1}")

        fusion_acc = (fusion_top1 / test_num) * 100
        print(f"***** Fusion top-1 acc for dataset-{test_id + 1}: {test_dataset}: {fusion_acc} *****")
        fusion_acc_table[task_id, test_id] = fusion_acc

        tested_cls_num += len(test_set.classnames)

upper_triangle_no_diag = np.triu(fusion_acc_table, k=1)
masked_matrix = np.ma.masked_equal(upper_triangle_no_diag, 0)
transfer_acc = np.mean(masked_matrix, axis=0)
transfer_avg_acc = np.mean(transfer_acc)
avg_acc = np.mean(fusion_acc_table, axis=0)
avg_avg_acc = np.mean(avg_acc)
print('average transfer acc: ', transfer_avg_acc)
print('average average acc: ', avg_avg_acc)
print('average last acc: ', np.mean(fusion_acc_table[-1, :]))

