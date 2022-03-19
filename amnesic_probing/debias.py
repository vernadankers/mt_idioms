from data import extract_sentences
import logging
from typing import Dict
import random
import sys
import pickle
import torch
import numpy as np
import scipy

from typing import List
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from wordfreq import zipf_frequency
sys.path.append('../data/')

logging.getLogger().setLevel(logging.INFO)


def set_seed(seed):
    """Set random seed."""
    if seed == -1:
        seed = random.randint(0, 1000)
    logging.info(f"Seed: {seed}")
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    # if you are using GPU
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class SKlearnClassifier(object):

    def __init__(self, m):

        self.model = m

    def train_network(self, X_train: np.ndarray, Y_train: np.ndarray,
                      X_dev: np.ndarray, Y_dev: np.ndarray) -> float:
        """
        :param X_train:
        :param Y_train:
        :param X_dev:
        :param Y_dev:
        :return: accuracy score on the dev set / Person's R in the case of regression
        """

        self.model.fit(X_train, Y_train)

        predictions = np.argmax(self.model.predict_proba(X_train), axis=-1)
        train_f1 = f1_score(Y_train, predictions, average="macro")

        predictions = np.argmax(self.model.predict_proba(X_dev), axis=-1)
        dev_f1 = f1_score(Y_dev, predictions, average="macro")
        return train_f1, dev_f1

    def get_weights(self) -> np.ndarray:
        """
        :return: final weights of the model, as np array
        """

        w = self.model.coef_
        if len(w.shape) == 1:
            w = np.expand_dims(w, 0)
        return w


def get_rowspace_projection(W: np.ndarray) -> np.ndarray:
    """
    :param W: the matrix over its nullspace to project
    :return: the projection matrix over the rowspace
    """

    if np.allclose(W, 0):
        w_basis = np.zeros_like(W.T)
    else:
        w_basis = scipy.linalg.orth(W.T)  # orthogonal basis

    P_W = w_basis.dot(w_basis.T)  # orthogonal projection on W's rowspace

    return P_W


def get_projection_to_intersection_of_nullspaces(
        rowspace_projection_matrices: List[np.ndarray], input_dim: int):
    """
    Given a list of rowspace projection matrices P_R(w_1), ..., P_R(w_n),
    this function calculates the projection to the intersection of all nullspasces of the matrices w_1, ..., w_n.
    uses the intersection-projection formula of Ben-Israel 2013 http://benisrael.net/BEN-ISRAEL-NOV-30-13.pdf:
    N(w1)∩ N(w2) ∩ ... ∩ N(wn) = N(P_R(w1) + P_R(w2) + ... + P_R(wn))
    :param rowspace_projection_matrices: List[np.array], a list of rowspace projections
    :param dim: input dim
    """

    I = np.eye(input_dim)
    Q = np.sum(rowspace_projection_matrices, axis=0)
    P = I - get_rowspace_projection(Q)
    return P


def get_debiasing_projection(
        cls_params, num_classifiers, input_dim, train_samples, test_samples,
        layer=6, baseline=False):
    """
    Args:
        - cls_params: a dictionary, containing the params for the sklearn classifier
        - num_classifiers: number of iterations
        - input_dim: size of input vectors
        - train_samples: list of custom Sentence objects
        - test_samples: lsit of custom Sentence objects
        - layer (int): layer to use
        - attention (bool): whether to use the attention query states
        - baseline (bool): whether to use baseline frequency labels

    Returns:
        - P, the debiasing projection;
        - rowspace_projections, the list of all rowspace projection;
        - Ws, the list of all calssifiers.
    """
    X_train, Y_train, mean_freq = load_data(
        train_samples, layer, baseline)
    X_dev, Y_dev, _ = load_data(
        test_samples, layer, baseline, mean_freq)
    X_train_cp = X_train.copy()
    X_dev_cp = X_dev.copy()

    rowspace_projections = []
    Ws = []

    first_f1 = None
    last_f1 = None
    f1s = []

    for i in range(num_classifiers):
        clf = SKlearnClassifier(LogisticRegression(**cls_params))
        _, f1_macro = clf.train_network(
            X_train_cp, Y_train,
            X_dev_cp, Y_dev)
        logging.info(f"layer {layer}, F1 {i} = {f1_macro}")
        f1s.append(f1_macro)

        W = clf.get_weights()
        Ws.append(W)
        P_rowspace_wi = get_rowspace_projection(
            W)  # projection to W's rowspace
        rowspace_projections.append(P_rowspace_wi)
        if i == 0:
            first_f1 = f1_macro
        else:
            last_f1 = f1_macro

        # Project to intersection of the nullspaces found so far
        # Use the intersection-projection formula of Ben-Israel 2013
        # N(w1)∩ N(w2) ∩ ... ∩ N(wn) = N(P_R(w1) + P_R(w2) + ... + P_R(wn))
        P = get_projection_to_intersection_of_nullspaces(
            rowspace_projections, input_dim)

        # Project while doing INLP
        X_train_cp = (P.dot(X_train.T)).T
        X_dev_cp = (P.dot(X_dev.T)).T

    # Calculate final projection matrix P=PnPn-1....P2P1
    # We use Ben-Israel's formula to increase stability
    # i.e., we explicitly project to intersection of all nullspaces
    P = torch.FloatTensor(get_projection_to_intersection_of_nullspaces(
        rowspace_projections, input_dim))
    logging.info(
        f"Computed P for layer {layer}, first F1 = {first_f1:.3f}, last F1 = {last_f1:.3f}, {i + 1} classifiers")
    return P, rowspace_projections, Ws


def load_data(samples, layer, baseline=False,
              mean_freq=None, average_pie=False):
    """Turn idiom sentences into matrices + labels.

    Args:
        - samples: custom Sentence objects from data.py (see ../data)
        - layer (int): 0 - 6 indicating the model's layer to use
        - baseline (bool): whether to use the baseline frequency label
        - mean_freq (float): mean Zipf frequency of training set
        - average_pie (bool): whether the PIE's states are to be averaged
    """
    vectors, labels = [], []
    if mean_freq is None:
        all_freqs = []
        for sample in samples:
            all_freqs.append(zipf_frequency(sample.idiom, 'en'))
        mean_freq = np.nanmean(all_freqs)

    for sample in samples:
        indices = torch.LongTensor(sample.index_select(1))
        sample_vectors = torch.index_select(
            sample.hidden_states[layer], dim=0, index=indices)
        assert 0 not in torch.sum(sample_vectors, dim=-1).tolist()

        if average_pie:
            sample_vectors = torch.mean(sample_vectors, dim=0).unsqueeze(0)

        # Store the labels assigned to the vectors
        for word_vector in sample_vectors:
            if baseline:
                labels.append(0 if zipf_frequency(
                    sample.idiom, 'en') < mean_freq else 1)
            else:
                labels.append(sample.label)

            vectors.append(word_vector)

    # To matrix / array
    vectors = torch.stack(vectors, dim=0).squeeze(1).numpy()
    labels = torch.FloatTensor(labels).numpy()
    return vectors, labels, mean_freq
