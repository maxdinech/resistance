"""
Adversarial attacks

An "adversarial attack" is the creation of an adversarial example from a given
image. For this purpose, we will use the images from the test dataset, which
have never been used by the network during training, to simulate a real-life
situation.

First, some pratical functions:

Multiple attack methods are provided:

* GDA : Gradient Descent Attack. From an image

---

usage: python3 -i attack.py model dataset

positional arguments:
  model       Trained model to evaluate
  dataset     Dataset used for training

"""


import os
import sys
import argparse

import torch
from torch import nn
import matplotlib.pyplot as plt

from basics import load_model
import data_loader
import plot


# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Parameters parsing
parser = argparse.ArgumentParser()
parser.add_argument("model", type=str, help="Trained model to evaluate")
parser.add_argument("dataset", type=str, help="Dataset used for training")
args = parser.parse_args()

model_name = args.model
dataset_name = args.dataset


# Loads the model
model = load_model(dataset_name, model_name)


# Loads the specified subset from the specified database
images, labels = data_loader.test(dataset_name, None)


# BASIC FUNCTIONS
# ---------------

# Loads the #img_id image from the test database.
def load_image(img_id):
    return images[img_id].view(1, 1, 28, 28)  # .to(device)


# Loads the #img_id label from the test database.
def load_label(img_id):
    return labels[img_id].item()


# Returns the label prediction of an image.
def prediction(image):
    return model.eval()(image).max(1)[1].item()


# Returns the confidence of the network that the image is `digit`.
def confidence(image, category):
    return model.eval()(image)[0, category].item()


# Yields the indices of the first n wrong predictions.
def errors(n=len(images)):
    i = 0
    l = len(images)
    while i < l and n > 0:
        image, label = load_image(i), load_label(i)
        if prediction(image) != label:
            yield i
            n -= 1
        i += 1


# Yields the indices of the first n correct predictions.
def not_errors(n=len(images)):
    i = 0
    l = len(images)
    while i < l and n > 0:
        image, label = load_image(i), load_label(i)
        if prediction(image) == label:
            yield i
            n -= 1
        i += 1


# ATTACK FUNCTIONS
# ----------------

class Attacker(nn.Module):
    def __init__(self, p, lr):
        super(Attacker, self).__init__()
        self.p = p
        self.r = nn.Parameter(torch.zeros(1, 1, 28, 28))
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def forward(self, x):
        return (x + self.r).clamp(0, 1)

    def loss_fn(self, image, digit):
        adv = self.forward(image)
        conf = model(adv)[0, digit]
        norm = (adv - image).abs().pow(self.p).sum()
        if conf < 0.2:
            return norm
        elif conf < 0.9:
            return conf + norm
        else:
            return conf - norm


def GDA(image, steps=500, p=2, lr=1e-3):
    norms, confs = [], []
    digit = prediction(image)
    attacker = Attacker(p, lr)
    # attacker = attacker.to(device)
    optim = attacker.optimizer
    for i in range(steps):
        # Training step
        loss = attacker.loss_fn(image, digit)
        attacker.zero_grad()
        loss.backward()
        optim.step()
        # Prints results
        adv = attacker.forward(image)
        conf = confidence(adv, digit)
        norm = (adv - image).norm(p).item()
        print(f"Step {i:4} -- conf: {conf:0.4f}, L_{p}(r): {norm:0.20f}",
              end='\r')
        norms.append(norm)
        confs.append(conf)
    print()
    success = prediction(adv) != digit
    return(success, adv, norms, confs)


def GDA_break(image, max_steps=500, p=2, lr=1e-3):
    norms, confs = [], []
    digit = prediction(image)
    attacker = Attacker(p, lr)
    # attacker = attacker.to(device)
    optim = attacker.optimizer
    adv = attacker.forward(image)
    steps = 0
    while confidence(adv, digit) >= 0.2 and steps < max_steps:
        steps += 1
        # Training step
        loss = attacker.loss_fn(image, digit)
        attacker.zero_grad()
        loss.backward()
        optim.step()
        # Prints results
        adv = attacker.forward(image)
        conf = confidence(adv, digit)
        norm = (adv - image).norm(p).item()
        print(f"Step {steps:4} -- conf: {conf:0.4f}, L_{p}(r): {norm:0.10f}",
              end='\r')
        norms.append(norm)
        confs.append(conf)
    print()
    return(steps, adv, norms, confs)


def GDA_graph(img, steps=500, p=2, lr=1e-3):
    success, adv, norms, confs = GDA(img, steps, p, lr)
    plot.attack_history(norms, confs)
    plt.show()
    path = os.path.join("..", "results", "last_attack_his±±±ry.png")
    plt.savefig(path, transparent=True)
    confc = lambda i: confidence(i, prediction(img))
    # plot.space_exploration(img, adv, confc)
    # plt.show()
    if success:
        print("\nAttack suceeded")
        img_pred = prediction(img)
        img_conf = confidence(img, img_pred)
        adv_pred = prediction(adv)
        adv_conf = confidence(adv, adv_pred)
        plot.attack_result(model_name, p,
                           img, img_pred, img_conf,
                           adv, adv_pred, adv_conf)
        path = os.path.join("..", "results", "last_attack_result.png")
        plt.savefig(path, transparent=True)
        plt.show()
    else:
        print("\nAttack failed")


def GDA_break_graph(img, max_steps=500, p=2, lr=1e-3):
    success, adv, norms, confs = GDA_break(img, max_steps, p, lr)
    plot.attack_history(norms, confs)
    path = os.path.join("..", "results", "last_attack_history.png")
    plt.savefig(path, transparent=True)
    plt.show()
    img_pred = prediction(img)
    img_conf = confidence(img, img_pred)
    adv_pred = prediction(adv)
    adv_conf = confidence(adv, adv_pred)
    plot.attack_result(model_name, p,
                       img, img_pred, img_conf,
                       adv, adv_pred, adv_conf)
    path = os.path.join("..", "results", "last_attack_result.png")
    plt.savefig(path, transparent=True)
    plt.show()


# RESISTANCE FUNCTIONS
# --------------------

# A resistance value greater than 1000 is not possible.
# Which is why 10000 will represent infinity when the attack fails.

def resistance_N(image, steps=500):
    success, _, norms, _ = GDA(image, steps)
    if success:
        return norms[-1]
    return 10000


def resistance_max(image, steps=500):
    success, _, norms, _ = GDA(image, steps)
    if success:
        return max(norms)
    return 10000


def resistance_min(image, max_steps=500):
    steps = attack_break(image, max_steps)[0]
    if steps < max_steps:
        return steps
    return 10000


# Computes the N-resistance, max_resistance and min_resistance in a single pass
def resistances_3(image, steps=500):
    success, _, norms, confs = GDA(image, steps)
    if success:
        res_N = norms[-1]
        res_max = max(norms)
        res_min = 1 + next((i for i, c in enumerate(confs) if c <= 0.2), steps)
        return (res_N, res_max, res_min)
    else:
        return (10000, 10000, 10000)


def resistances_lists(images_list, steps=500):
    L_res_N, L_res_max, L_res_min = [], [], []
    for image in images_list:
        res_N, res_max, res_min = resistances_3(image, steps)
        L_res_N += [res_N]
        L_res_max += [res_max]
        L_res_min += [res_min]
    return (L_res_N, L_res_max, L_res_min)


# FOOLBOX PLAYGROUND
# ------------------

"""
GradientSignAttack :
    rapide
    marche souvent

IterativeGradientSignAttack :
    lente
    marche souvent
    pas forcément mieux que GradientSignAttack.

GradientAttack :
    rapide
    marche rarement

IterativeGradientAttack :
    lente
    marche rarement

FGSM
    rapide
    marche souvent

LBFGSAttack
"""

# TODO : Jongler plus facilement entre Pytorch et Numpy

import numpy as np
import foolbox
import shutil


fmodel = foolbox.models.PyTorchModel(model, (0, 1), num_classes=10,
                                     channel_axis=1,
                                     cuda=False)


def fb_attack(img_id, attack_name, p=0.1):
    path = f"../results/{dataset_name}/{attack_name}/"
    if not os.path.exists(path):
        os.mkdir(path)
    # criterion = foolbox.criteria.OriginalClassProbability(p)
    attack = getattr(foolbox.attacks, attack_name)(fmodel)
    img = load_image(img_id)
    img_pred = prediction(img)
    img_conf = confidence(img, img_pred)
    np_adv = attack(np.array(img).reshape(1, 28, 28), img_pred)
    try:
        adv = torch.Tensor(np_adv).view(1, 1, 28, 28)
        adv_pred = prediction(adv)
        adv_conf = confidence(adv, adv_pred)
        plot.attack_result(model_name, 2,
                           img, img_pred, img_conf,
                           adv, adv_pred, adv_conf)
        plt.close()
        shutil.move("../results/latest/attack_result.png",
                    path + f"{img_id:04d}.png")
        torch.save(adv, path + f"{img_id:04d}.pt")
        print("success")
        return adv
    except:
        print("failed")


def fb_attacks(size, attack_name, p=0.1):
    adv_list = []
    for img_id in not_errors():
        if len(adv_list) >= size:
            break
        adv = fb_attack(img_id, attack_name, p)
        if adv is not None:
            adv_list.append(adv)
    return adv_list
