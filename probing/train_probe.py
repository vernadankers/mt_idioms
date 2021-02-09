#!/usr/bin/env python3
#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This example shows how to use higher to do Model Agnostic Meta Learning (MAML)
for few-shot Omniglot classification.
For more details see the original MAML paper:
https://arxiv.org/abs/1703.03400
This code has been modified from Jackie Loong's PyTorch MAML implementation:
https://github.com/dragen1860/MAML-Pytorch/blob/master/omniglot_train.py
Our MAML++ fork and experiments are available at:
https://github.com/bamos/HowToTrainYourMAMLPytorch
"""

import time
import copy
import random
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm
import higher
from sklearn.metrics import f1_score
from classifier import Classifier
from data import Probe, extract_encodings, set_seed
import torch
import logging


def test_single(sample, net, layer):
    net.eval()
    outputs = net(sample.vector[layer])
    return torch.round(outputs).item()

def preprocess(samples, layer=6, b_size=16):
    random.shuffle(samples)
    data = []
    for i in range(0, len(samples), b_size):
        x_spt = torch.stack([x.vector[layer] for x in samples[i : i + b_size]])
        y_spt = torch.FloatTensor([x.label for x in samples[i : i + b_size]])
        data.append((x_spt, y_spt))
    return data


def main(train_samples, test_samples, l0=False, layer=6, b_size=16,
         outer_lr=0.00025, epochs=10, pos_weight=0.2, inner_lr=0.1,
         n_inner_iter=5, meta_batch_size=1, batch_size=32, maml=False, lambd=0.001):
    # set_seed(1)
    device = torch.device('cpu')

    # Create a vanilla PyTorch neural network that will be
    # automatically monkey-patched by higher later.
    # Before higher, models could *not* be created like this
    # and the parameters needed to be manually updated and copied
    # for the updates.
    net = Probe(512, l0)

    # We will use Adam to (meta-)optimize the initial parameters
    # to be adapted.
    meta_opt = optim.Adam(net.parameters(), lr=outer_lr)
    all_micro, all_macro = [], []

    f1_micro, f1_macro = test(test_samples, net, layer, batch_size)
    logging.info(f"Before training: F1-micro: {f1_micro:.2f} | F1-macro: {f1_macro:.2f}")

    for _ in range(epochs):
        if maml:
            train(train_samples, net, device, meta_opt, pos_weight, inner_lr,
                  n_inner_iter, meta_batch_size, batch_size, layer, lambd)
        else:
            train_regular(train_samples, net, device, meta_opt, pos_weight,
                meta_batch_size, batch_size, layer, lambd)
        
    f1_micro, f1_macro = test(test_samples, net, layer, batch_size)
    logging.info(f"Testing: F1-micro: {f1_micro:.2f} | F1-macro: {f1_macro:.2f}")
    return net, f1_micro, f1_macro


def train(db, net, device, meta_opt, pos_weight=0.2, inner_lr=0.1,
          n_inner_iter=5, meta_batch_size=1, batch_size=32, layer=6, lambd=0.001):
    random.shuffle(db)
    db = preprocess(db, layer, batch_size)
    db = [(db[i], db[i + 1]) for i in range(0, len(db) - 1, 2)]
    db = [db[i: i + meta_batch_size]
          for i in range(0, len(db), meta_batch_size)]

    net.train()
    qry_losses = []
    qry_preds = []
    qry_labels = []

    start_time = time.time()
    for meta_batch in db:

        # Initialize the inner optimizer to adapt the parameters to
        # the support set.
        inner_opt = torch.optim.SGD(net.parameters(), lr=inner_lr)

        meta_opt.zero_grad()
        for i in range(len(meta_batch)):
            with higher.innerloop_ctx(
                net, inner_opt, copy_initial_weights=False,
                track_higher_grads=True
            ) as (fnet, diffopt):
                (x_spt, y_spt), (x_qry, y_qry) = meta_batch[i]
                # Optimize the likelihood of the support set by taking
                # gradient steps w.r.t. the model's parameters.
                # This adapts the model's meta-parameters to the task.
                # higher is able to automatically keep copies of
                # your network's parameters as they are being updated.
                for _ in range(n_inner_iter):
                    spt_logits = fnet(x_spt)
                    weights = copy.deepcopy(y_spt)
                    weights[weights == 1] = pos_weight
                    weights[weights == 0] = 1 - pos_weight
                    loss_fn = torch.nn.BCELoss(weight=weights)
                    spt_loss = loss_fn(spt_logits, y_spt) + (0 if not net.l0 else lambd * fnet.weights1.mask.l0_norm())
                    diffopt.step(spt_loss)

                # The final set of adapted parameters will induce some
                # final loss and accuracy on the query dataset.
                # These will be used to update the model's meta-parameters.
                qry_logits = fnet(x_qry)
                weights = copy.deepcopy(y_qry)
                weights[weights == 1] = pos_weight
                weights[weights == 0] = 1 - pos_weight
                loss_fn = torch.nn.BCELoss(weight=weights)
                qry_loss = loss_fn(qry_logits, y_qry) + (0 if not net.l0 else lambd * fnet.weights1.mask.l0_norm())
                qry_losses.append(qry_loss.detach())
                predictions = torch.round(qry_logits).tolist()
                qry_preds.extend(predictions)
                qry_labels.extend(y_qry.tolist())

                # Update the model's meta-parameters to optimize the query
                # losses across all of the tasks sampled in this batch.
                # This unrolls through the gradient steps.
                qry_loss.backward()

        meta_opt.step()
    f1_micro = f1_score(qry_labels, qry_preds, average='micro')
    f1_macro = f1_score(qry_labels, qry_preds, average='macro')
    iter_time = time.time() - start_time
    logging.info(f'Train Loss: {np.mean(qry_losses):.2f} | F1-micro: {f1_micro:.2f} | F1-macro: {f1_macro:.2f} | Time: {iter_time:.2f}')
    # if net.l0:
    #     nums = sum([1 for x in net.weights1.mask.forward() if x == 0])
    #     print(nums)


def train_regular(db, net, device, meta_opt, pos_weight=0.2, meta_batch_size=1, batch_size=32, layer=6, lambd=0.001):
    random.shuffle(db)
    db = preprocess(db, layer, batch_size)
    db = [(db[i], db[i + 1]) for i in range(0, len(db) - 1, 2)]
    db = [db[i: i + meta_batch_size]
          for i in range(0, len(db), meta_batch_size)]

    net.train()
    qry_losses = []
    qry_preds = []
    qry_labels = []

    start_time = time.time()
    for meta_batch in db:
        for (x_spt, y_spt), (x_qry, y_qry) in meta_batch:
            # Initialize the inner optimizer to adapt the parameters to
            # the support set.
            meta_opt.zero_grad()
            spt_logits = net(x_spt)
            weights = copy.deepcopy(y_spt)
            weights[weights == 1] = pos_weight
            weights[weights == 0] = 1 - pos_weight
            loss_fn = torch.nn.BCELoss(weight=weights)
            spt_loss = loss_fn(spt_logits, y_spt) + (0 if not net.l0 else lambd * net.weights1.mask.l0_norm())
            spt_loss.backward()
            meta_opt.step()

            # The final set of adapted parameters will induce some
            # final loss and accuracy on the query dataset.
            # These will be used to update the model's meta-parameters.
            meta_opt.zero_grad()
            qry_logits = net(x_qry)
            weights = copy.deepcopy(y_qry)
            weights[weights == 1] = pos_weight
            weights[weights == 0] = 1 - pos_weight
            loss_fn = torch.nn.BCELoss(weight=weights) 
            qry_loss = loss_fn(qry_logits, y_qry) + (0 if not net.l0 else lambd * net.weights1.mask.l0_norm())
            qry_loss.backward()
            meta_opt.step()

            qry_losses.append(qry_loss.detach())
            predictions = torch.round(qry_logits).tolist()
            qry_preds.extend(predictions)
            qry_labels.extend(y_qry.tolist())

    f1_micro = f1_score(qry_labels, qry_preds, average='micro')
    f1_macro = f1_score(qry_labels, qry_preds, average='macro')
    iter_time = time.time() - start_time
    logging.info(f'Train Loss: {np.mean(qry_losses):.2f} | F1-micro: {f1_micro:.2f} | F1-macro: {f1_macro:.2f} | Time: {iter_time:.2f}')
    # if net.l0:
    #     nums = sum([1 for x in net.weights1.mask.forward() if x == 0])
    #     print(nums)

def test(db, net, layer, batch_size):
    # Crucially in our testing procedure here, we do *not* fine-tune
    # the model during testing for simplicity.
    # Most research papers using MAML for this task do an extra
    # stage of fine-tuning here that should be added if you are
    # adapting this code for research.

    db = preprocess(db, layer, batch_size)

    net.eval()
    predictions = []
    labels = []
    for x, y in db:
        outputs = net(x)
        outputs = torch.round(outputs)
        predictions.extend(outputs.tolist())
        labels.extend(y.tolist())

    f1_model_micro = f1_score(labels, predictions, average='micro')
    f1_model_macro = f1_score(labels, predictions, average='macro')
    #nums = sum([1 for x in net.weights1.mask.forward() if x == 0])
    return f1_model_micro, f1_model_macro


if __name__ == '__main__':
    main()