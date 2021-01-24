import numpy as np
import matplotlib.pyplot as plt


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


def eigvals(data, internal=True):
    plt.fill_between(data.agent["t"],
                     data.agent["eigs"][:, -1],
                     data.agent["eigs"][:, 0],
                     # **fill_style)
                     facecolor=data.style["c"],
                     alpha=0.3)
    plt.plot(data.agent["t"], data.agent["eigs"][:, [0, -1]],
             **dict(data.style, alpha=0.7))
    if internal:
        plt.plot(data.agent["t"], data.agent["eigs"][:, 1:-1],
                 **dict(data.style, label="_", ls="--", alpha=0.5))


def tracking_error(data):
    plt.plot(data.env["t"],
             np.linalg.norm(data.env["e"].squeeze(), axis=1),
             **data.style)


def estimation_error(data):
    plt.plot(data.env["t"],
             np.linalg.norm(
                 (data.env["W"] - data.env["Wcirc"]).squeeze(), axis=1),
             **data.style)


def h(data):
    plt.plot(data.agent["t"], data.agent["h"].squeeze(),
             **dict(data.style))


def parameters(data, index=None):
    lines = plt.plot(data.env["t"], data.env["W"][:, index or slice(None), 0], **data.style)
    plt.setp(lines[1:], label=None)


def states_and_input(data, key, index):
    return plt.plot(data.env["t"], data.env[key][:, index], **data.style)


def performance_index(data):
    plt.plot(data.env["t"], data.env["J"][:, 0, 0], **data.style)


def HJB_error(data):
    plt.plot(data.env["t"], data.env["e_HJB"], **data.style)


def outputs(data, key, index, style=None):
    y = data.env[key][:, index]

    if index < 3:
        y = np.rad2deg(y)

    return plt.plot(data.env["t"], y, **style or data.style)


def vector_by_index(data, key, index, mult=1, style=None):
    y = data.env[key][:, index, 0] * mult
    return plt.plot(data.env["t"], y, **style or data.style)


def all(data, key, style=None):
    return plt.plot(
        data.env["t"],
        data.env[key].reshape(data.env["t"].shape[0], -1),
        **style or data.style)


def matrix_by_index(data, key, index, style=None):
    return plt.plot(data.env["t"], data.env[key][:, index, index], **style or data.style)
