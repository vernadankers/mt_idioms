import time
import copy
import random
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import f1_score
import torch
import logging


class Probe(torch.nn.Module):
    def __init__(self, in_features, l0=False):
        super().__init__()
        self.l0 = l0
        self.dropout = torch.nn.Dropout(0.1)
        if l0:
            self.weights1 = HardConcreteLinear(in_features, 1)
        else:
            self.weights1 = torch.nn.Linear(in_features, 1)

    def forward(self, input_vec):
        input_vec = self.dropout(input_vec)
        output = self.weights1(input_vec)
        return torch.sigmoid(output).squeeze(-1)


class Trainer():
    def __init__(self, lr, epochs, layer, batch_size, l0, lambd, average_over_pie=True):
        self.epochs = epochs
        self.l0 = l0
        self.layer = layer
        self.batch_size = batch_size
        self.lambd = lambd
        self.lr = lr
        self.init_model()
        self.average_over_pie = average_over_pie

    def init_model(self):
        self.model = Probe(512, self.l0)
        if torch.cuda.is_available():
            self.model.cuda()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

    def train(self, train_samples, test_samples, verbose=False):
        f1_micro, f1_macro = self.test(test_samples)
        if verbose:
            logging.info(f"Random: F1-micro: {f1_micro:.2f} | F1-macro: {f1_macro:.2f}")

        for _ in range(self.epochs):
            self.train_epoch(train_samples)

        f1_micro, f1_macro = self.test(test_samples)
        if verbose:
            logging.info(f"Testing: F1-micro: {f1_micro:.2f} | F1-macro: {f1_macro:.2f}")
        return round(f1_micro, 3), round(f1_macro, 3)

    def train_epoch(self, dataset):
        random.shuffle(dataset)
        self.model.train()
        loss_fn = torch.nn.BCELoss()
        losses, all_preds, all_labels = [], [], []

        for inputs, targets in self.preprocess(dataset):
            self.optimizer.zero_grad()
            if torch.cuda.is_available():
                preds = self.model(inputs.cuda()).cpu()
            else:
                preds = self.model(inputs)
            loss = loss_fn(preds, targets) + \
                (0 if not self.l0 else self.lambd * self.model.weights1.mask.l0_norm())
            loss.backward()
            self.optimizer.step()
            losses.append(loss.detach())
            all_preds.extend(torch.round(preds).tolist())
            all_labels.extend(targets.tolist())

        f1_micro = f1_score(all_labels, all_preds, average='micro')
        f1_macro = f1_score(all_labels, all_preds, average='macro')
        return np.mean(losses), f1_micro, f1_macro

    def test(self, samples):
        dataset = self.preprocess(samples, test=True)
        self.model.eval()
        predictions = []
        labels = []
        for x, y in dataset:
            if torch.cuda.is_available():
                outputs = self.model(x.cuda()).cpu()
            else:
                outputs = self.model(x)
            outputs = torch.round(outputs)
            predictions.extend(outputs.tolist())
            labels.extend(y.tolist())

        f1_model_micro = f1_score(labels, predictions, average='micro')
        f1_model_macro = f1_score(labels, predictions, average='macro')
        return f1_model_micro, f1_model_macro

    def preprocess(self, samples, test=False):
        if not test:
            samples_0 = [s for s in samples if s.label == 0]
            samples_1 = [s for s in samples if s.label == 1]
            samples = [random.choice(samples_0) for _ in range(max(len(samples_0), len(samples_1)))] + \
                      [random.choice(samples_1) for _ in range(max(len(samples_0), len(samples_1)))]
        random.shuffle(samples)
        data = []
        for i in range(0, len(samples), self.batch_size):
            inputs, targets = [], []
            for s in samples[i : i + self.batch_size]:
                if self.average_over_pie:
                    indices = s.index_select(1)
                    if not indices:
                        continue
                    indices = torch.LongTensor(indices) #[random.choice(indices)])
                    vec = torch.index_select(s.hidden_states[self.layer], dim=0, index=indices)
                    vec = torch.mean(vec, dim=0)
                else:
                    indices = s.index_select(1, tags=["NOUN"])
                    if not indices:
                        continue
                    indices = torch.LongTensor([random.choice(indices)])
                    vec = torch.index_select(s.hidden_states[self.layer], dim=0, index=indices)
                    vec = torch.mean(vec, dim=0)

                inputs.append(vec)
                targets.append(s.label)
            inputs = torch.stack(inputs)
            targets = torch.FloatTensor(targets)
            data.append((inputs, targets))

        return data
