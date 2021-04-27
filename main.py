import numpy as np
import scipy
from types import SimpleNamespace as SN
from pathlib import Path
import itertools

from fym.core import BaseEnv, BaseSystem
from fym.agents import LQR
import fym.logging

import matplotlib
import matplotlib.pyplot as plt

import plot
import wingrock
import multirotor
import linear
import logs

np.warnings.filterwarnings("error", category=np.VisibleDeprecationWarning)
plt.rc("font", family="Times New Roman")
plt.rc("text", usetex=True)
plt.rc("lines", linewidth=1)
plt.rc("axes", grid=True)
plt.rc("grid", linestyle="--", alpha=0.8)


def run_simple(env):
    env.reset()

    while True:
        env.render()

        done = env.step()

        if done:
            break

    env.close()


def run_agent(env, agent):
    obs = env.reset()

    while True:
        env.render()

        action = agent.get_action(obs)
        next_obs, done = env.step(action)

        if done:
            break

        obs = next_obs

    env.close()
    agent.close()


def run_with_agent(env, agent):
    obs = env.reset()

    while True:
        env.render()

        action = agent.get_action(obs)
        next_obs, reward, done = env.step(action)

        agent.update(obs, action, next_obs, reward, done)

        if done:
            break

        obs = next_obs

    env.close()
    agent.close()


def exp1():
    basedir = Path("data/exp1")

    cfg = wingrock.cfg

    wingrock.load_config()
    cfg.dir = Path(basedir, "data00")
    cfg.label = "MRAC"
    run_simple(wingrock.MRACEnv())

    wingrock.load_config()
    cfg.dir = Path(basedir, "data01")
    cfg.label = "HMRAC"
    run_simple(wingrock.HMRACEnv())


def exp1_plot():
    def get_data(datadir):
        data = SN()
        env, info = fym.logging.load(list(datadir.glob("*env.h5"))[0],
                                     with_info=True)
        data.env = env
        data.info = info
        agentlist = list(datadir.glob("*agent.h5"))
        if agentlist != []:
            data.agent = fym.logging.load(agentlist[0])
        data.style = dict(label=info["cfg"].label)
        return data

    plt.rc("font", family="Times New Roman")
    plt.rc("text", usetex=True)
    plt.rc("lines", linewidth=1)
    plt.rc("axes", grid=True)
    plt.rc("grid", linestyle="--", alpha=0.8)

    datadir = Path("data", "exp1")
    mrac = get_data(Path(datadir, "data00"))
    hmrac = get_data(Path(datadir, "data01"))
    data = [mrac, hmrac]

    basestyle = dict(c="k", lw=0.7)
    cmdstyle = dict(basestyle, c="r", ls="--", label="Command")
    mrac.style.update(basestyle, c="g", ls="-")
    hmrac.style.update(basestyle, c="k", ls="-")

    # Figure common setup
    t_range = (0, 50)

    # All in inches
    subsize = (4.05, 0.946)
    width = 4.94
    top = 0.2
    bottom = 0.671765
    left = 0.5487688
    hspace = 0.2716

    # =================
    # States and inputs
    # =================
    figsize, pos = plot.posing(3, subsize, width, top, bottom, left, hspace)

    plt.figure(figsize=figsize)

    ax = plt.subplot(311, position=pos[0])
    lines = []
    lines += plt.plot(mrac.env["t"], mrac.env["c"][:, 0], **cmdstyle)
    lines += [plot.states_and_input(d, "x", 0)[0] for d in data]
    plt.ylabel(r"$x_1$")
    plt.ylim(-2, 2)
    plt.figlegend(
        lines,
        [line.get_label() for line in lines],
        bbox_to_anchor=(0.99, 0.78)
    )

    plt.subplot(312, sharex=ax, position=pos[1])
    [plot.states_and_input(d, "x", 1) for d in data]
    plt.ylabel(r"$x_2$")
    plt.ylim(-2, 2)

    plt.subplot(313, sharex=ax, position=pos[2])
    [plot.states_and_input(d, "u", 0) for d in data]
    plt.ylabel(r'$u$')
    plt.xlabel("Time, sec")
    plt.xlim(t_range)
    plt.ylim(-80, 80)

    for ax in plt.gcf().get_axes():
        ax.label_outer()

    # ====================
    # Parameter estimation
    # ====================
    figsize, pos = plot.posing(2, subsize, width, top, bottom, left, hspace)
    plt.figure(figsize=figsize)

    ax = plt.subplot(211, position=pos[0])
    # [plot.parameters(d, (0, 1, 8, 9, 10)) for d in data]
    plot.parameters(mrac)
    plt.ylabel(r"$W$")
    # plt.ylim(0, 0.6)
    plt.legend(loc='best')

    plt.subplot(212, sharex=ax, position=pos[1])
    # [plot.parameters(d, (2, 3, 4, 5, 6, 7)) for d in data]
    plot.parameters(hmrac)
    plt.ylabel(r"$W$")
    plt.legend(loc='best')
    plt.xlabel("Time, sec")
    plt.xlim(t_range)
    # plt.ylim(0, 85)

    # ===============================================
    # Tracking and parameter estimation errors (norm)
    # ===============================================
    figsize, pos = plot.posing(2, subsize, width, top, bottom, left, hspace)
    plt.figure(figsize=figsize)

    ax = plt.subplot(211, position=pos[0])
    [plot.tracking_error(d) for d in data]
    plt.ylabel(r"$||e||$")
    plt.ylim(0, 0.2)
    plt.legend(loc='best')

    plt.subplot(212, sharex=ax, position=pos[1])
    [plot.estimation_error(d) for d in data]
    plt.ylabel(r"$||\tilde{W}||$")
    plt.xlabel("Time, sec")
    plt.xlim(t_range)
    plt.ylim(0, 85)

    # =================
    # Performance index
    # =================
    figsize, pos = plot.posing(2, subsize, width, top, bottom, left, hspace)
    plt.figure(figsize=figsize)

    ax = plt.subplot(211, position=pos[0])
    [plot.performance_index(d) for d in data]
    plt.ylabel(r"J")
    plt.xlim(t_range)
    plt.legend(loc="best")

    plt.subplot(211, sharex=ax, position=pos[1])
    [plot.HJB_error(d) for d in data]
    plt.ylabel(r"$\epsilon_{\mathrm{HJB}}$")
    plt.xlabel("Time, sec")

    # Saving
    # basedir = Path("img")
    # basedir.mkdir(exist_ok=True)

    # plt.figure(1)
    # plt.savefig(Path(basedir, "figure_1.pdf"), bbox_inches="tight")

    # plt.figure(2)
    # plt.savefig(Path(basedir, "figure_2.pdf"), bbox_inches="tight")

    plt.show()


def exp2():
    basedir = Path("data/exp2")

    cfg = multirotor.cfg

    # multirotor.load_config()
    # cfg.dir = Path(basedir, "data00")
    # cfg.label = "MRAC"
    # run_simple(multirotor.MRACEnv())

    multirotor.load_config()
    cfg.dir = Path(basedir, "data01")
    cfg.label = "HMRAC"
    run_agent(multirotor.HMRACEnv(), multirotor.FECMRACAgent())


def exp2_plot():
    def get_data(datadir):
        data = SN()
        env, info = fym.logging.load(list(datadir.glob("*env.h5"))[0],
                                     with_info=True)
        data.env = env
        data.info = info
        agentlist = list(datadir.glob("*agent.h5"))
        if agentlist != []:
            data.agent = fym.logging.load(agentlist[0])
        data.style = dict(label=info["cfg"].label)
        return data

    cfg = multirotor.cfg
    multirotor.load_config()

    plt.rc("font", family="Times New Roman")
    plt.rc("text", usetex=True)
    plt.rc("lines", linewidth=1)
    plt.rc("axes", grid=True)
    plt.rc("grid", linestyle="--", alpha=0.8)

    datadir = Path("data", "exp2")
    mrac = get_data(Path(datadir, "data00"))
    hmrac = get_data(Path(datadir, "data01"))
    data = [mrac, hmrac]

    basestyle = dict(c="k", lw=0.7)
    cmdstyle = dict(basestyle, c="r", ls="--", label="Command")
    refstyle = dict(basestyle, c="k", ls="-", label="Ref. Model")
    mrac.style.update(basestyle, c="g", ls="-")
    hmrac.style.update(basestyle, c="b", ls="-")

    # Figure common setup
    t_range = (0, cfg.final_time)
    # t_range = (0, 15)

    # All in inches
    subsize = (4.05, 0.946)
    width = 4.94
    top = 0.2
    bottom = 0.671765
    left = 0.5487688
    hspace = 0.2716
    r2d = np.rad2deg(1)

    # =================
    # States and inputs
    # =================
    figsize, pos = plot.posing(3, subsize, width, top, bottom, left, hspace)

    plt.figure(figsize=figsize)

    ax = plt.subplot(311, position=pos[0])
    lines = []
    # lines += plt.plot(mrac.env["t"], mrac.env["c"][:, 0], **cmdstyle)
    lines += plot.vector_by_index(mrac, "c", 0, mult=r2d, style=cmdstyle)
    lines += plot.vector_by_index(mrac, "xr", 0, mult=r2d, style=refstyle)
    lines += [plot.vector_by_index(d, "x", 0, r2d)[0] for d in data]
    plt.ylabel(r"$p$ [deg/s]")
    # plt.ylim(-40, 40)
    plt.figlegend(
        lines,
        [line.get_label() for line in lines],
        bbox_to_anchor=(0.99, 0.78)
    )

    plt.subplot(312, sharex=ax, position=pos[1])
    plot.vector_by_index(mrac, "c", 1, mult=r2d, style=cmdstyle)
    plot.vector_by_index(mrac, "xr", 1, mult=r2d, style=refstyle)
    [plot.vector_by_index(d, "x", 1, r2d) for d in data]
    plt.ylabel(r"$q$ [deg/s]")
    # plt.ylim(-40, 40)

    plt.subplot(313, sharex=ax, position=pos[2])
    plot.vector_by_index(mrac, "c", 2, mult=r2d, style=cmdstyle)
    plot.vector_by_index(mrac, "xr", 2, mult=r2d, style=refstyle)
    [plot.vector_by_index(d, "x", 2, r2d) for d in data]
    plt.ylabel(r"$r$ [deg/s]")
    # plt.ylim(-40, 40)

    # plt.subplot(414, sharex=ax, position=pos[3])
    # [plot.all(d, "u") for d in data]
    # plt.ylabel(r'$u$')
    # plt.ylim(1.07, 1.47)

    plt.xlabel("Time, sec")
    plt.xlim(t_range)

    for ax in plt.gcf().get_axes():
        ax.label_outer()

    # =======================================
    # Tracking error and parameter estimation
    # =======================================
    figsize, pos = plot.posing(3, subsize, width, top, bottom, left, hspace)
    plt.figure(figsize=figsize)

    ax = plt.subplot(311, position=pos[0])
    [plot.tracking_error(d) for d in data]
    plt.ylabel(r"$||e||$")
    # plt.ylim(0, 0.2)
    plt.legend(loc='best')

    plt.subplot(312, sharex=ax, position=pos[1])
    [plot.all(d, "W") for d in data]
    plt.ylabel(r"$W$")
    # plt.ylim(0, 85)

    # plt.subplot(313, sharex=ax, position=pos[2])
    # plot.all(hmrac, "What")
    # plt.ylabel(r"$\hat{W}$")
    # plt.ylim(0, 85)

    plt.subplot(313, sharex=ax, position=pos[2])
    [plot.all(d, "u") for d in data]
    plt.ylabel(r'$u$')
    # plt.ylim(1.07, 1.47)

    plt.xlabel("Time, sec")
    plt.xlim(t_range)

    plt.show()


def exp3():
    basedir = Path("data/exp3")

    cfg = wingrock.cfg

    # wingrock.load_config()
    # cfg.dir = Path(basedir, "data00")
    # cfg.label = "MRAC"
    # run_simple(wingrock.MRACEnv())

    wingrock.load_config()
    cfg.dir = Path(basedir, "data01")
    cfg.label = "Value Learner"
    cfg.R = np.zeros((1, 1))
    run_with_agent(
        wingrock.ValueLearnerEnv(), wingrock.ValueLearnerAgent())

    wingrock.load_config()
    cfg.dir = Path(basedir, "data01")
    cfg.label = "Value Learner"
    cfg.R = np.zeros((1, 1))
    run_with_agent(
        wingrock.ValueLearnerEnv(), wingrock.ValueLearnerAgent())

    # wingrock.load_config()
    # cfg.dir = Path(basedir, "data02")
    # cfg.label = "Double-MRAC"
    # run_simple(wingrock.DoubleMRACEnv())


def exp3_plot():
    def get_data(datadir):
        data = SN()
        env, info = fym.logging.load(list(datadir.glob("*env.h5"))[0],
                                     with_info=True)
        data.env = env
        data.info = info
        agentlist = list(datadir.glob("*agent.h5"))
        if agentlist != []:
            data.agent = fym.logging.load(agentlist[0])
        data.style = dict(label=info["cfg"].label)
        return data

    plt.rc("font", family="Times New Roman")
    plt.rc("text", usetex=True)
    plt.rc("lines", linewidth=1)
    plt.rc("axes", grid=True)
    plt.rc("grid", linestyle="--", alpha=0.8)

    datadir = Path("data", "exp3")
    mrac = get_data(Path(datadir, "data00"))
    vlmrac = get_data(Path(datadir, "data01"))
    data = [mrac, vlmrac]

    basestyle = dict(c="k", lw=0.7)
    cmdstyle = dict(basestyle, c="r", ls="--", label="Command")
    mrac.style.update(basestyle, c="g", ls="-")
    vlmrac.style.update(basestyle, c="b", ls="-")

    # Figure common setup
    cfg = wingrock.cfg
    wingrock.load_config()
    t_range = (0, cfg.final_time)

    # All in inches
    subsize = (4.05, 0.946)
    width = 4.94
    top = 0.2
    bottom = 0.671765
    left = 0.5487688
    hspace = 0.2716

    # =================
    # States and inputs
    # =================
    figsize, pos = plot.posing(3, subsize, width, top, bottom, left, hspace)

    plt.figure(figsize=figsize)

    ax = plt.subplot(311, position=pos[0])
    lines = []
    lines += plt.plot(mrac.env["t"], mrac.env["c"][:, 0], **cmdstyle)
    lines += [plot.states_and_input(d, "x", 0)[0] for d in data]
    plt.ylabel(r"$x_1$")
    plt.ylim(-2, 2)
    plt.figlegend(
        lines,
        [line.get_label() for line in lines],
        bbox_to_anchor=(0.99, 0.78)
    )

    plt.subplot(312, sharex=ax, position=pos[1])
    [plot.states_and_input(d, "x", 1) for d in data]
    plt.ylabel(r"$x_2$")
    plt.ylim(-2, 2)

    plt.subplot(313, sharex=ax, position=pos[2])
    [plot.states_and_input(d, "u", 0) for d in data]
    plt.ylabel(r'$u$')
    plt.ylim(-80, 80)
    plt.xlabel("Time, sec")
    plt.xlim(t_range)

    for ax in plt.gcf().get_axes():
        ax.label_outer()

    # ====================
    # Parameter estimation
    # ====================
    figsize, pos = plot.posing(2, subsize, width, top, bottom, left, hspace)
    plt.figure(figsize=figsize)

    ax = plot.subplot(pos, 0)
    plot.all(mrac, "Wcirc", style=cmdstyle)
    plot.all(mrac, "W", style=dict(mrac.style, c="k"))
    plt.ylim(-70, 30)

    plot.subplot(pos, 1, sharex=ax)
    plot.all(mrac, "Wcirc", style=cmdstyle)
    plot.all(vlmrac, "W", style=dict(mrac.style, c="k"))
    plot.all(vlmrac, "F", style=dict(mrac.style, c="b"))
    plot.all(vlmrac, "What", is_agent=True, style=dict(mrac.style, c="g"))
    plt.ylim(-70, 30)

    plt.xlabel("Time, sec")
    plt.xlim(t_range)

    plt.show()


def exp4():
    basedir = Path("data/exp4")

    logs.set_logger(basedir, "train.log")

    cfg = linear.cfg

    # linear.load_config()
    # cfg.dir = Path(basedir, "data00")
    # cfg.label = "Z. P. Jiang (Init. Admm.)"
    # run_with_agent(linear.QLearnerEnv(), linear.ZLearnerAgent())

    linear.load_config()
    cfg.dir = Path(basedir, "data01")
    cfg.label = "Q Learner (Init. Admm.)"
    run_with_agent(linear.QLearnerEnv(), linear.QLearnerAgent())

    # linear.load_config()
    # cfg.dir = Path(basedir, "data02")
    # cfg.label = "Z. P. Jiang (Init. Non-Admm.)"
    # cfg.A = np.array([[1, 1, 0], [0, 2, 0], [1, 0, -3]])
    # linear.calc_config()
    # run_with_agent(linear.QLearnerEnv(), linear.ZLearnerAgent())

    # linear.load_config()
    # cfg.dir = Path(basedir, "data03")
    # cfg.label = "Q Learner (Init. Non-Admm.)"
    # cfg.A = np.array([[1, 1, 0], [0, 2, 0], [1, 0, 3]])
    # linear.calc_config()
    # run_with_agent(linear.QLearnerEnv(), linear.QLearnerAgent())


def exp4_plot():
    datadir = Path("data", "exp4")
    zlearner = plot.get_data(Path(datadir, "data00"))
    qlearner = plot.get_data(Path(datadir, "data01"))
    zlearner_na = plot.get_data(Path(datadir, "data02"))
    qlearner_na = plot.get_data(Path(datadir, "data03"))
    data = [zlearner, qlearner]
    data_na = [zlearner_na, qlearner_na]

    basestyle = dict(c="k", lw=0.7)
    refstyle = dict(basestyle, c="r", ls="--")
    zstyle = dict(basestyle, c="y", ls="-")
    qstyle = dict(basestyle, c="b", ls="-.")
    zlearner.style.update(zstyle)
    qlearner.style.update(qstyle)
    zlearner_na.style.update(zstyle)
    qlearner_na.style.update(qstyle)

    # Figure common setup
    cfg = linear.cfg
    linear.load_config()
    t_range = (0, cfg.final_time)

    # All in inches
    subsize = (4.05, 0.946)
    width = 4.94
    top = 0.2
    bottom = 0.671765
    left = 0.5487688
    hspace = 0.2716

    # =================
    # States and inputs
    # =================
    figsize, pos = plot.posing(5, subsize, width, top, bottom, left, hspace)
    plt.figure(figsize=figsize)

    ax = plot.subplot(pos, 0)
    [plot.vector_by_index(d, "x", 0)[0] for d in data]
    plt.ylabel(r"$x_1$")
    # plt.ylim(-2, 2)
    plt.legend()

    plot.subplot(pos, 1, sharex=ax)
    [plot.vector_by_index(d, "x", 1) for d in data]
    plt.ylabel(r"$x_2$")
    # plt.ylim(-2, 2)

    plot.subplot(pos, 2, sharex=ax)
    [plot.vector_by_index(d, "u", 0) for d in data]
    plt.ylabel(r'$u$')
    # plt.ylim(-80, 80)

    # ====================
    # Parameter estimation
    # ====================
    ax = plot.subplot(pos, 3)
    plot.all(qlearner, "K", style=dict(refstyle, label="True"))
    for d in data:
        plot.all(
            d, "K", is_agent=True,
            style=dict(marker="o", markersize=2)
        )
    plt.ylabel(r"$\hat{K}$")
    plt.legend()
    # plt.ylim(-70, 30)

    plot.subplot(pos, 4, sharex=ax)
    plot.all(qlearner, "P", style=dict(qlearner.style, c="r", ls="--"))
    for d in data:
        plot.all(
            d, "P", is_agent=True,
            style=dict(marker="o", markersize=2)
        )
    plt.ylabel(r"$\hat{P}$")
    # plt.ylim(-70, 30)

    plt.xlabel("Time, sec")
    plt.xlim(t_range)

    for ax in plt.gcf().get_axes():
        ax.label_outer()

    # ==================================
    # States and inputs (Non-Admissible)
    # ==================================
    figsize, pos = plot.posing(5, subsize, width, top, bottom, left, hspace)
    plt.figure(figsize=figsize)

    ax = plot.subplot(pos, 0)
    [plot.vector_by_index(d, "x", 0)[0] for d in data_na]
    plt.ylabel(r"$x_1$")
    # plt.ylim(-2, 2)
    plt.legend()

    plot.subplot(pos, 1, sharex=ax)
    [plot.vector_by_index(d, "x", 1) for d in data_na]
    plt.ylabel(r"$x_2$")
    # plt.ylim(-2, 2)

    plot.subplot(pos, 2, sharex=ax)
    [plot.vector_by_index(d, "u", 0) for d in data_na]
    plt.ylabel(r'$u$')
    # plt.ylim(-80, 80)

    # =====================================
    # Parameter estimation (Non-Admissible)
    # =====================================
    ax = plot.subplot(pos, 3)
    plot.all(qlearner_na, "K", style=dict(refstyle, label="True"))
    for d in data_na:
        plot.all(
            d, "K", is_agent=True,
            style=dict(marker="o", markersize=2)
        )
    plt.ylabel(r"$\hat{K}$")
    plt.legend()
    # plt.ylim(-70, 30)

    plot.subplot(pos, 4, sharex=ax)
    plot.all(qlearner_na, "P", style=refstyle)
    for d in data_na:
        plot.all(
            d, "P", is_agent=True,
            style=dict(marker="o", markersize=2)
        )
    plt.ylabel(r"$\hat{P}$")
    # plt.ylim(-70, 30)

    plt.xlabel("Time, sec")
    plt.xlim(t_range)

    for ax in plt.gcf().get_axes():
        ax.label_outer()

    imgdir = Path("img", datadir.relative_to("data"))
    imgdir.mkdir(exist_ok=True)

    plt.figure(1)
    plt.savefig(Path(imgdir, "figure_1.pdf"), bbox_inches="tight")

    plt.figure(2)
    plt.savefig(Path(imgdir, "figure_2.pdf"), bbox_inches="tight")

    plt.show()


def exp5():
    """
    This experiment compares our algorithms and the Kleinman algorithm.
    """
    basedir = Path("data", "exp5")

    # Setup
    np.random.seed(3000)
    v = np.random.randn(5, 5) * 3
    A = np.diag([2, 3, 4, 5, 6])
    A = v.dot(A).dot(np.linalg.inv(v))
    B = np.random.randn(5, 7) * 3
    Q = np.diag([100, 0, 0, 20, 30])
    R = np.diag([1, 3, 8, 10, 11, 12, 15])
    Kopt, Popt, *_ = LQR.clqr(A, B, Q, R)
    eps = 1e-16
    maxiter = 1000
    n, m = B.shape

    # Kleinman Iteration
    def kleinman(K, name):
        logger = fym.logging.Logger(path=Path(basedir, name))

        for i in itertools.count(0):
            P = scipy.linalg.solve_lyapunov(
                (A - B.dot(K)).T, -(Q + K.T.dot(R).dot(K)))
            next_K = np.linalg.inv(R).dot(B.T).dot(P)

            # print(np.linalg.eigvals(P))

            logger.record(
                i=i, P=P, K=K, next_K=next_K, Popt=Popt, Kopt=Kopt,
            )

            if ((K - next_K)**2).sum() < eps or i > maxiter:
                break

            K = next_K

        logger.close()

    # SQL Iteration
    def sql(K, name):
        F = - np.eye(m) * 1
        # f = np.random.rand(3, 3)
        # F = - f.T.dot(f)

        K0 = K
        # prev_H21 = None

        logger = fym.logging.Logger(path=Path(basedir, name))

        for i in itertools.count(0):
            blkA = np.block([[A - B.dot(K), B], [np.zeros_like(B.T), F]])
            blkK = np.block([[np.eye(n), np.zeros_like(B)], [-K, np.eye(m)]])
            blkQ = blkK.T.dot(scipy.linalg.block_diag(Q, R)).dot(blkK)
            blkH = scipy.linalg.solve_lyapunov(blkA.T, -blkQ)
            H11, H21, H22 = blkH[:n, :n], blkH[n:, :n], blkH[n:, n:]
            next_K = K + np.linalg.inv(H22).dot(H21)

            P = H11 - H21.T.dot(np.linalg.inv(H22)).dot(H21)

            next_H11 = P

            print("Ak: ", sorted(np.linalg.eigvals(A - B.dot(K)).real))
            print("-Dk: ", sorted(-np.linalg.eigvals(H21.T.dot(np.linalg.inv(H22)).dot(H21)).real))
            print("Hk: ", sorted(np.linalg.eigvals(H11).real))
            print("H21_n: ", np.linalg.norm(H21))

            Ak = A - B.dot(K)
            print(
                sorted(np.linalg.eigvals(
                    (Ak + np.eye(n)).dot(np.linalg.inv(Ak - np.eye(n)))).real)
            )
            # if prev_H21 is not None:
            #     print(np.linalg.norm(
            #         prev_H21.dot(A - B.dot(K) + np.eye(n))
            #         - H21.dot(A - B.dot(K) - np.eye(n)))
            #     )

            # print("Kkd: ",
            #       np.sum((K - np.linalg.inv(R).dot(B.T).dot(H11))**2))
            print("")

            if np.all(np.linalg.eigvals(A - B.dot(K)).real < 0):
                breakpoint()

            Kt = Kopt - K
            blkKt = np.block([[np.eye(n), np.zeros_like(B)], [-Kt, np.eye(m)]])
            blkA_s = blkKt.dot(np.block([[A - B.dot(K), B], [F.dot(Kt), F]]))
            blkH_s = scipy.linalg.solve_lyapunov(blkA_s.T, -blkQ)
            H11_s, H22_s = blkH_s[:n, :n], blkH_s[n:, n:]

            P_s = H11_s - Kt.T.dot(H22_s).dot(Kt)

            eigs = np.linalg.eigvals(P)
            Peig = [eigs.min().real, eigs.max().real]

            logger.record(
                i=i, P=P, K=K, next_K=next_K, Popt=Popt, Kopt=Kopt,
                P_s=P_s, Peig=Peig, K0=K0, H11=H11, next_H11=next_H11,
            )

            # if np.all(np.linalg.eigvals(A - B.dot(K)).real < 0):
            #     break

            if ((K - next_K)**2).sum() < eps or i > maxiter:
                break

            K = next_K

        logger.close()

    # Nonstabiliing initial gain
    K = np.zeros((m, n))
    kleinman(K, "kleinman-unstable.h5")
    sql(K, "sql-unstable.h5")

    # Stabiling initial gain
    K, *_ = LQR.clqr(A, B, np.eye(n), np.eye(m))
    kleinman(K, "kleinman-stable.h5")
    sql(K, "sql-stable.h5")


def exp5_plot():
    datadir = Path("data", "exp5")

    def get_data(name, label, style=dict()):
        data = SN()
        data.alg = fym.logging.load(Path(datadir, name))
        data.style = dict(label=label, **style)
        return data

    def error_plot(data, estkey, optkey, **style):
        style = dict(data.style, **style)
        plt.plot(
            data.alg["i"],
            np.sqrt(
                np.square(
                    data.alg[estkey] - data.alg[optkey]).sum(axis=(1, 2))),
            **style
        )
        plt.yscale("log")

    kleinman_style = dict(c="k", ls="--", marker="o", markersize=2)
    sql_style = dict(c="b", ls="-", marker="o", markersize=2)

    data_stable = [
        get_data(name, label, style) for name, label, style in
        (["kleinman-stable.h5", "Kleinman (stable)", kleinman_style],
         ["sql-stable.h5", "Proposed (stable)", sql_style])]

    data_unstable = [
        get_data(name, label, style) for name, label, style in
        (["kleinman-unstable.h5", "Kleinman (unstable)", kleinman_style],
         ["sql-unstable.h5", "Proposed (unstable)", sql_style])]

    subsize = (4.05, 0.946)
    width = 4.94
    top = 0.2
    bottom = 0.671765
    left = 0.5487688
    hspace = 0.2716

    # Figure 1 (stable)
    figsize, pos = plot.posing(2, subsize, width, top, bottom, left, hspace)
    plt.figure(figsize=figsize)

    ax = plot.subplot(pos, 0)
    [error_plot(d, "P", "Popt") for d in data_stable]
    error_plot(data_unstable[1], "P_s", "Popt", c="r")

    plt.ylabel(r"${P}$ error")
    plt.legend()

    plot.subplot(pos, 1, sharex=ax)
    [error_plot(d, "K", "Kopt") for d in data_stable]

    plt.ylabel(r"${K}$ error")
    plt.legend()

    plt.xlabel("Iteration")

    # Figure 2 (unstable)
    figsize, pos = plot.posing(4, subsize, width, top, bottom, left, hspace)
    plt.figure(figsize=figsize)

    ax = plot.subplot(pos, 0)
    [error_plot(d, "P", "Popt") for d in reversed(data_unstable)]
    # error_plot(data_unstable[1], "P_s", "Popt", c="r")

    plt.ylabel(r"${P}$ error")
    plt.legend()

    plot.subplot(pos, 1, sharex=ax)
    [error_plot(d, "K", "Kopt") for d in data_unstable]

    plt.ylabel(r"${K}$ error")
    plt.legend()

    plot.subplot(pos, 2, sharex=ax)
    plt.plot(data_unstable[1].alg["i"], data_unstable[1].alg["Peig"][:, 0])
    plt.plot(data_unstable[1].alg["i"], data_unstable[1].alg["Peig"][:, 1])

    plt.ylabel(r"Eigenvalues")

    plot.subplot(pos, 3, sharex=ax)
    [error_plot(d, "K", "next_K") for d in reversed(data_unstable)]

    plt.ylabel(r"$K_{k+1} - K_k$")

    plt.xlabel("Iteration")

    # Save
    imgdir = Path("img", datadir.relative_to("data"))
    imgdir.mkdir(exist_ok=True)

    plt.figure(1)
    plt.savefig(Path(imgdir, "figure_1.pdf"), bbox_inches="tight")

    plt.figure(2)
    plt.savefig(Path(imgdir, "figure_2.pdf"), bbox_inches="tight")

    plt.show()


def main():
    # exp1()
    # exp1_plot()

    # exp2()
    # exp2_plot()

    # exp3()
    # exp3_plot()

    # exp4()
    # exp4_plot()

    exp5()
    exp5_plot()


if __name__ == "__main__":
    main()
