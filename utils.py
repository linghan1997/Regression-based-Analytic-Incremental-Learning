import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
import numpy as np
import math
from sklearn.metrics import pairwise_distances
from tqdm import tqdm


class kernel_layer(nn.Module):
    def __init__(self, sv, gamma):
        super(kernel_layer, self).__init__()
        self.sv = sv
        self.gamma = gamma

    def forward(self, x):
        return kernel(x, self.sv, gamma=self.gamma)


def cls_acc(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [
        float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())
        for k in topk
    ]


def one_hot_cls_acc(output, target):
    if isinstance(output, np.ndarray) and isinstance(target, np.ndarray):
        pred = np.argmax(output, axis=1)
        labels = np.argmax(target, axis=1)
        correct_predictions = np.equal(pred, labels)
        acc = np.mean(correct_predictions.astype(float)) * 100
    elif torch.is_tensor(output) and torch.is_tensor(target):
        pred = torch.argmax(output, dim=1)
        labels = torch.argmax(target, dim=1)
        correct_predictions = torch.eq(pred, labels)
        acc = torch.mean(correct_predictions.float()) * 100
    else:
        raise ValueError('Unsupported types for prediction and target.')
    return acc


def clip_classifier(classnames, template, clip_model, device="cuda"):
    with torch.no_grad():
        clip_weights = []

        for classname in classnames:
            # Tokenize prompts
            classname = classname.replace('_', ' ')
            texts = [t.format(classname) for t in template]
            texts = clip.tokenize(texts).to(device)

            class_embeddings = clip_model.encode_text(texts)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)

            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            clip_weights.append(class_embedding)

        clip_weights = torch.stack(clip_weights, dim=1).to(device)
    return clip_weights


def encode_images(clip_model, images):
    """
        forward pass of CLIP image encoder to extract unit vector features
    """
    features = clip_model.encode_image(images)
    features /= features.norm(dim=-1, keepdim=True)


    return features

def kernel(x, X, gamma):
    """
    Args:
        x: input data
        X: static center embeddings
        gamma: Guassian kernel hyperparameter
    """
    with torch.no_grad():
        btch = 32
        ker = torch.exp(((X[:btch, :] - x.unsqueeze(1)) ** 2).sum(dim=-1).mul_(-1. * gamma))
        for i in range(1, math.ceil(X.size(0) / btch)):
            ker_new = torch.exp(
                ((X[i * btch:(i + 1) * btch, :] - x.unsqueeze(1)) ** 2).sum(dim=-1).mul_(-1. * gamma))
            ker = torch.cat((ker, ker_new), 1)
    return ker


def gaussian_kernel(x, X, gamma):
    distance = pairwise_distances(x, X, metric='euclidean', squared=True)
    return np.exp(-gamma * distance)


def linear_kernel(x, X):
    return x @ X.T


def cos_kernel(x, X):
    return 1 - linear_kernel(x, X)


def sample_per_class(dataset, n, num_classes=1000):
    indices_per_class = [[] for _ in range(num_classes)]
    for idx, (_, label) in enumerate(dataset.imgs):
        indices_per_class[label].append(idx)

    sampled_indices = [idx for indices in indices_per_class for idx in np.random.choice(indices, n, replace=False)]
    return sampled_indices


class kernel_ridge_regression:
    def __init__(self, lamda=0.1, gamma=0.1):
        self.lamda = lamda
        self.gamma = gamma
        self.alpha = None
        self.kernel = None

    def train(self, X, Y):
        """
        Gaussian kernel only
        """
        self.kernel = kernel(X, X, gamma=self.gamma).cpu().numpy()
        self.alpha = np.mat(self.kernel + self.lamda * np.eye(self.kernel.shape[0])).I @ Y
        return self.alpha

    def predict(self, X, X_train):
        """
        Args:
            X: on-device tensor
            X_train: on-device tensor
        Returns:
            on-cpu numpy
        """
        predictions = kernel(X, X_train, gamma=self.gamma).cpu().numpy() @ self.alpha
        return predictions
