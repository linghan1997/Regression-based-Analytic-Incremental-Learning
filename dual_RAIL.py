"""
multi-domain transfer CLIP under KRR version (dual space)
"""

import clip
import random
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import yaml
from easydict import EasyDict
from tqdm import tqdm

from scenario_datasets import build_dataset
from scenario_datasets.utils import build_data_loader
from scenario_datasets.collections import CIFAR100, MNIST
from utils import *

cfg_file = "configs/analytic_clip.yaml"
cfg = yaml.load(open(cfg_file, 'r'), Loader=yaml.Loader)
cfg = EasyDict(cfg)

seed = cfg.seed
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)

cfg.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset_sequence = cfg.datasets
print("Multi-task dataset sequence: ", dataset_sequence)

merged_classnames = []  # for clip zero-shot
for _, train_dataset in enumerate(dataset_sequence):
    if train_dataset == "cifar100":
        dataset = CIFAR100(num_shots=-1, preprocess=None, val_transform=None, batch_size=cfg.batch_size)
    elif train_dataset == "mnist":
        dataset = MNIST(num_shots=-1, preprocess=None, val_transform=None, batch_size=cfg.batch_size)
    else:
        dataset = build_dataset(train_dataset, os.path.join(DIR_PATH, 'datasets'), cfg.num_shots)
    merged_classnames += dataset.classnames
print(f'Size of cross-domain category set: {len(merged_classnames)}')

fusion_acc_table = np.zeros((len(dataset_sequence), len(dataset_sequence)))
adapter_acc_table = np.zeros((len(dataset_sequence), len(dataset_sequence)))
in_domain_acc_list = []

cfg.previous_class_num = 0
current_class_names = []

"""
Loading model
"""
print('Loading pretrained CLIP model...')
clip_model, train_transform, val_preprocess = clip.load(cfg.backbone, device=cfg.device, jit=False)
template = ['a photo of a {}.']
clip_model.eval()
krr = kernel_ridge_regression(lamda=0.001, gamma=cfg.gamma)

cfg.trained_class_num = 0
feature_memory = None
y = None

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

    current_train_features = []
    current_train_one_hot_labels = []
    with torch.no_grad():
        for i, (images, target) in \
                enumerate(tqdm(train_loader, desc=f'Extracting training features', total=len(train_loader),
                               unit='batch')):
            target += cfg.trained_class_num
            images, target = images.to(cfg.device), target.to(cfg.device)
            img_embeddings = encode_images(clip_model, images)

            train_labels_one_hot = F.one_hot(target, cfg.current_class_num).float()

            current_train_features.append(img_embeddings)
            current_train_one_hot_labels.append(train_labels_one_hot)

    current_train_features = torch.cat(current_train_features, dim=0)
    current_train_one_hot_labels = torch.cat(current_train_one_hot_labels, dim=0).cpu().numpy()

    if task_id == 0:
        feature_memory = current_train_features
        y = current_train_one_hot_labels
    else:
        feature_memory = torch.cat([feature_memory, current_train_features], dim=0)
        y = np.concatenate([y, np.zeros((y.shape[0], cfg.increment))], axis=1)
        y = np.concatenate([y, current_train_one_hot_labels], axis=0)

    print(f"Size of Gram matrix in task-{task_id + 1}: {feature_memory.size(0)}")

    alpha = krr.train(feature_memory, y)  # obtain the dual parameter

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

        clip_weights = clip_classifier(merged_classnames, template, clip_model)
        class_range_min, class_range_max = tested_cls_num, tested_cls_num + len(test_set.classnames)
        in_domain, in_domain_acc = 0.0, 0.0
        adapter_in_domain, adapter_in_domain_acc = 0.0, 0.0

        top1, top5, test_num = 0.0, 0.0, 0.0
        fusion_top1, fusion_top5 = 0.0, 0.0
        adapt_top1, adapt_top5 = 0.0, 0.0

        with torch.no_grad():
            for inputs, targets in tqdm(test_loader, desc=f'Evaluating on dataset-{test_id + 1}: {test_dataset}',
                                        total=len(test_loader), unit='batch'):
                test_num += inputs.size(0)

                inputs, targets = inputs.to(cfg.device), targets.to(cfg.device)
                targets += tested_cls_num

                test_features = encode_images(clip_model, inputs)
                outputs = 100. * test_features @ clip_weights
                outputs = F.softmax(outputs, dim=-1)

                # In domain acc
                predict_cls = torch.argmax(outputs, dim=-1)
                in_range = (predict_cls >= class_range_min) & (predict_cls < class_range_max)
                in_domain += torch.sum(in_range)

                # Zero-shot acc
                zs_outputs = outputs
                acc1 = cls_acc(zs_outputs, targets)
                top1 += acc1[0]

                # Select samples that belong to learned domains determined by CLIP zero-shot
                mask = predict_cls < cfg.current_class_num
                if torch.sum(mask) > 0:
                    samples_to_adapt = test_features[mask]
                    outputs_adapted = krr.predict(samples_to_adapt, feature_memory)
                    outputs_adapted = torch.tensor(outputs_adapted, device=cfg.device, dtype=torch.float)
                    # Zero-padding to (N_ad, C_all)
                    padding_right = outputs.size(-1) - outputs_adapted.size(-1)
                    outputs_adapted = F.pad(outputs_adapted, pad=(0, padding_right, 0, 0), mode='constant', value=0)

                    adapter_pred = torch.argmax(outputs_adapted, dim=-1)
                    adapter_in_range = (adapter_pred >= class_range_min) & (adapter_pred < class_range_max)
                    adapter_in_domain += torch.sum(adapter_in_range)

                    outputs[mask] = (1-cfg.fusion_weight) * outputs[mask] + cfg.fusion_weight * outputs_adapted

                fusion_acc1 = cls_acc(outputs, targets)
                fusion_top1 += fusion_acc1[0]

                if test_id <= task_id:
                    outputs = krr.predict(test_features, feature_memory)  # (N_ad, C_adapter)
                    outputs = torch.tensor(outputs, device=cfg.device, dtype=torch.float)
                    adapt_acc1 = cls_acc(outputs, targets)
                    adapt_top1 += adapt_acc1[0]

            if test_id <= task_id:
                pure_adapter_acc = (adapt_top1 / test_num) * 100
                print(f"Pure adapter acc for dataset-{test_id + 1}: {test_dataset}: {pure_adapter_acc}")
                adapter_acc_table[task_id, test_id] = pure_adapter_acc

            in_domain_acc = (in_domain / test_num) * 100
            print(f"In-domain top-1 acc for dataset-{test_id + 1}: {test_dataset}: {in_domain_acc}")

            top1 = (top1 / test_num) * 100
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
