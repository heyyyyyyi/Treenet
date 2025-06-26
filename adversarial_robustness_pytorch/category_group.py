import os
import json
import torch
import numpy as np
import torch.nn as nn
from sklearn.metrics import pairwise_distances
from core.data import get_data_info, load_data
from core.models import create_model
from core.utils import Logger, parser_eval, seed
from torch.autograd import Variable
from core.attacks import create_attack  # Import attack creation utility
from core.attacks import CWLoss  # Import CWLoss if needed
from core.utils import ctx_noparamgrad_and_eval

# ----------------- 解析参数 -------------------
parse = parser_eval()
args = parse.parse_args()

LOG_DIR = os.path.join(args.log_dir, args.desc)
with open(os.path.join(LOG_DIR, 'args.txt'), 'r') as f:
    old = json.load(f)
    args.__dict__ = dict(vars(args), **old)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA_DIR = os.path.join(args.data_dir, args.data)
WEIGHTS = os.path.join(LOG_DIR, 'weights-best.pt')
logger = Logger(os.path.join(LOG_DIR, 'log-coarse.log'))
logger.log(f'Using device: {device}')

# ----------------- 加载数据 -------------------
seed(args.seed)
_, _, train_loader, test_loader = load_data(DATA_DIR, args.batch_size, args.batch_size_validation,
                                            use_augmentation=False, shuffle_train=False)

loader = train_loader if args.train else test_loader
x_list, y_list = zip(*[(x, y) for (x, y) in loader])
x_test = torch.cat(x_list, 0).to(device)
y_test = torch.cat(y_list, 0).to(device)

# ----------------- 加载模型 -------------------
logger.log(f'Creating model: {args.model}')
info = get_data_info(DATA_DIR)
model = create_model(args.model, args.normalize, info, device)
checkpoint = torch.load(WEIGHTS)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# ----------------- Clustering 类 -------------------
class Clustering(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes

    def confusion_matrix(self, ypred, ytrue):
        ypred = ypred.cpu().numpy()
        ytrue = ytrue.cpu().numpy()
        assert len(np.unique(ypred)) == len(np.unique(ytrue))
        N = self.num_classes
        F = np.zeros((N, N), dtype=np.int32)
        for i in range(N):
            for j in range(N):
                F[i][j] = np.sum((ypred == i) & (ytrue == j))
        return F

    def spectral_clustering(self, confusion_mat, K=2, gamma=10):
        # Ensure confusion_mat is valid
        if np.any(np.isnan(confusion_mat)) or np.any(np.isinf(confusion_mat)):
            raise ValueError("Confusion matrix contains NaNs or infinities. Check input data.")

        # Normalize confusion matrix
        confusion_norm = confusion_mat / (confusion_mat.sum(axis=1, keepdims=True) + 1e-8)
        A = 0.5 * ((np.eye(self.num_classes) - confusion_norm) + (np.eye(self.num_classes) - confusion_norm).T)
        np.fill_diagonal(A, 1e-8)  # Add small value to diagonal for stability

        # Compute degree matrix and its inverse square root
        D = A.sum(axis=1)
        D = np.maximum(D, 1e-8)  # Add a threshold to avoid zeros
        D_inv = np.diag(1.0 / np.sqrt(D))

        L = D_inv @ A @ D_inv
        L = (L + L.T) / 2  # Ensure symmetry

        # Check for NaNs or infinities in L
        if np.any(np.isnan(L)) or np.any(np.isinf(L)):
            raise ValueError("Matrix L contains NaNs or infinities. Check input data for numerical stability.")

        # Compute eigenvectors for clustering
        from scipy.linalg import eigh
        s, V = eigh(L)
        eigenvector = V[:, -K:]  # shape: [num_classes, K]
        eigenvector = eigenvector / (np.linalg.norm(eigenvector, axis=1, keepdims=True) + 1e-8)

        # Use KMeans for clustering
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=K, random_state=0)
        assignments = km.fit_predict(eigenvector)
        return assignments


def get_all_assign(net, trainloader, device, adversarial=False):
    required_data = []
    required_targets = []
    # ----------------- 生成对抗样本 -------------------
    if adversarial:
        attack = create_attack(model, CWLoss, args.attack, args.attack_eps, args.attack_iter, args.attack_step)
        logger.log(f'Generating adversarial examples using {args.attack}')

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        inputs, targets = Variable(inputs), Variable(targets)
        if adversarial:
            with ctx_noparamgrad_and_eval(net):
                inputs, _ = attack.perturb(inputs, targets)
        batch_required_data = net(inputs)
        batch_required_targets = targets

        batch_required_data = batch_required_data.data.cpu().numpy()
        batch_required_targets = batch_required_targets.data.cpu().numpy()
        required_data = stack_or_create(required_data, batch_required_data, axis=0)
        required_targets = stack_or_create(required_targets, batch_required_targets, axis=1)

    return required_data, required_targets


def stack_or_create(all_array, local_array, axis=1):
    if isinstance(local_array, np.ndarray):
        if len(all_array) != 0:
            if axis == 1:
                all_array = np.hstack((all_array, local_array))
            else:
                all_array = np.vstack((all_array, local_array))
        else:
            all_array = local_array
    elif torch.is_tensor(local_array):
        if len(all_array) != 0:
            all_array = torch.cat((all_array, local_array), axis)
        else:
            all_array = local_array
    return all_array

# ----------------- 提取模型预测并聚类 -------------------
# with torch.no_grad():
#     coarse_outputs, coarse_target = get_all_assign(model, train_loader, device, adversarial=False)
#     coarse_outputs = torch.tensor(coarse_outputs, device=device)
#     coarse_target = torch.tensor(coarse_target, device=device)
#     coarse_preds = torch.argmax(coarse_outputs, dim=1)
coarse_outputs, coarse_target = get_all_assign(model, train_loader, device, adversarial=True)
coarse_outputs = torch.tensor(coarse_outputs, device=device)
coarse_target = torch.tensor(coarse_target, device=device)
coarse_preds = torch.argmax(coarse_outputs, dim=1) 

function = Clustering(num_classes=args.num_classes)
conf_matrix = function.confusion_matrix(coarse_preds, coarse_target)
cluster_result = function.spectral_clustering(conf_matrix, K=args.num_superclasses)

# ----------------- 输出 coarse 类别与 fine 类别对应关系 -------------------
print("===> Coarse 分类结果:")
for i in range(args.num_superclasses):
    group = [j for j in range(args.num_classes) if cluster_result[j] == i]
    print(f"Coarse Group {i}: {group}")
