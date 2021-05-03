import numpy as np
import matplotlib.pyplot as plt
from types import SimpleNamespace as SN
import fym


def get_data(path, style=dict(), with_info=False):
    dataset = SN()
    if with_info:
        data, info = fym.logging.load(path, with_info=with_info)
        dataset.info = info
        dataset.style = style | dict(label=info["cfg"].label)
    else:
        data = fym.logging.load(path)
        dataset.style = style
    dataset.data = data
    return dataset


def posing(n, subsize, width, top, bottom, left, hspace):
    refpoint = (bottom, left)
    figsize = (width, refpoint[1] + subsize[1] * n + hspace * (n - 1) + top)
    sub = np.divide(subsize, figsize)
    ref = np.divide(refpoint, figsize)

    h = hspace / figsize[1]
    poses = []
    for i in range(n):
        subref = ref + np.array([0, (h + sub[1]) * (n - 1 - i)])
        pos = np.vstack((subref, sub))
        poses.append(pos.ravel())

    return figsize, poses


def subplot(pos, index, **kwargs):
    digit = int(str(len(pos)) + str(1) + str(index + 1))
    return plt.subplot(digit, position=pos[index], **kwargs)


def eigvals(dataset, internal=True):
    plt.fill_between(dataset.data["t"],
                     dataset.data["eigs"][:, -1],
                     dataset.data["eigs"][:, 0],
                     # **fill_style)
                     facecolor=dataset.style["c"],
                     alpha=0.3)
    plt.plot(dataset.data["t"], dataset.data["eigs"][:, [0, -1]],
             **dict(dataset.style, alpha=0.7))
    if internal:
        plt.plot(dataset.data["t"], dataset.data["eigs"][:, 1:-1],
                 **dict(dataset.style, label="_", ls="--", alpha=0.5))


def tracking_error(dataset):
    plt.plot(dataset.data["t"],
             np.linalg.norm(dataset.data["e"].squeeze(), axis=1),
             **dataset.style)


def estimation_error(dataset):
    plt.plot(dataset.data["t"],
             np.linalg.norm(
                 (dataset.data["W"] - dataset.data["Wcirc"]).squeeze(), axis=1),
             **dataset.style)


def h(dataset):
    plt.plot(dataset.data["t"], dataset.data["h"].squeeze(),
             **dict(dataset.style))


def parameters(dataset, index=None):
    lines = plt.plot(dataset.data["t"], dataset.data["W"][:, index or slice(None), 0], **dataset.style)
    plt.setp(lines[1:], label=None)


def states_and_input(dataset, key, index):
    return plt.plot(dataset.data["t"], dataset.data[key][:, index], **dataset.style)


def performance_index(dataset):
    plt.plot(dataset.data["t"], dataset.data["J"][:, 0, 0], **dataset.style)


def HJB_error(dataset):
    plt.plot(dataset.data["t"], dataset.data["e_HJB"], **dataset.style)


def outputs(dataset, key, index, style=None):
    y = dataset.data[key][:, index]

    if index < 3:
        y = np.rad2deg(y)

    return plt.plot(dataset.data["t"], y, **style or dataset.style)


def vector_by_index(dataset, key, index, mult=1, style=None):
    y = dataset.data[key][:, index, 0] * mult
    return plt.plot(dataset.data["t"], y, **style or dataset.style)


def all(dataset, key, style=dict(), is_agent=False):
    style = dict(dataset.style, **style)
    if is_agent:
        dataset = dataset.data
    else:
        dataset = dataset.data
    lines = plt.plot(
        dataset["t"], dataset[key].reshape(dataset["t"].shape[0], -1), **style)
    plt.setp(lines[1:], label=None)
    return lines


def matrix_by_index(dataset, key, index, style=None):
    style = dict(dataset.style, **style)
    lines = plt.plot(dataset.data["t"], dataset.data[key][:, index, index], **style)
    plt.setp(lines[1:], label=None)
    return lines
