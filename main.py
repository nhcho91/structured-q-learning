import numpy as np
import scipy
from types import SimpleNamespace as SN
from pathlib import Path

from fym.core import BaseEnv, BaseSystem
import fym.logging

import matplotlib
import matplotlib.pyplot as plt

import plot
import wingrock
import multirotor
import linear

# np.warnings.filterwarnings("error", category=np.VisibleDeprecationWarning)
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

    cfg = linear.cfg

    linear.load_config()
    cfg.dir = Path(basedir, "data00")
    cfg.label = "Q Learner"
    run_with_agent(linear.QLearnerEnv(), linear.QLearnerAgent())


def exp4_plot():
    datadir = Path("data", "exp4")
    qlearner = plot.get_data(Path(datadir, "data00"))
    data = [qlearner]

    basestyle = dict(c="k", lw=0.7)
    refstyle = dict(basestyle, c="r", ls="--")
    qlearner.style.update(basestyle, c="k", ls="-")

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
    figsize, pos = plot.posing(3, subsize, width, top, bottom, left, hspace)
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
    plot.all(qlearner, "K", style=refstyle)
    plot.all(qlearner, "Khat", style=dict(qlearner.style, c="k"))
    plt.ylabel(r"$\hat{K}$")
    # plt.ylim(-70, 30)

    plot.subplot(pos, 1, sharex=ax)
    plot.all(qlearner, "P", style=dict(qlearner.style, c="r", ls="--"))
    plot.all(qlearner, "Phat", style=dict(qlearner.style, c="k"))
    plt.ylabel(r"$\hat{P}$")
    # plt.ylim(-70, 30)

    plt.xlabel("Time, sec")
    plt.xlim(t_range)

    plt.show()


def main():
    # exp1()
    # exp1_plot()

    # exp2()
    # exp2_plot()

    # exp3()
    # exp3_plot()

    exp4()
    exp4_plot()


if __name__ == "__main__":
    main()
