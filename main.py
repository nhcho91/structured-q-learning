import numpy as np
import scipy
from types import SimpleNamespace as SN
from pathlib import Path
import itertools
import logging
import random

from fym.core import BaseEnv, BaseSystem
from fym.agents import LQR
from fym.utils.linearization import jacob_analytic
import fym.logging

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import plot
import wingrock
import multirotor
import linear
import logs
import agents

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


class LearningEnv(BaseEnv):
    def step(self, action):
        *_, done = self.update()
        next_obs = self.observation()
        return next_obs, None, done

    def behavior(self, t, x):
        return NotImplementedError("The behavior policy is not implemented")

    def observation(self):
        return NotImplementedError("The observation is not implemented")

    def reset(self):
        super().reset()
        return self.observation()

    def run(self, agent):
        obs = self.reset()
        logger = logging.getLogger("logs")

        while True:
            self.render()

            action = agent.get_action(obs)
            next_obs, reward, done = self.step(action)

            agent.update(obs, action, next_obs, reward, done)

            t = obs[0]

            if agent.is_train(t):
                agent.train(t)

                PRE = np.linalg.eigvals(agent.P).real
                CRE = np.linalg.eigvals(self.A - self.B.dot(agent.K)).real
                logger.info(
                    f"[{type(agent).__name__}] "
                    f"Time: {t:5.2f} sec | "
                    f"Max CRE: {CRE.max():5.2f} | "
                    f"Min PRE: {PRE.min():5.2f} | "
                )

            if agent.is_record(t):
                record = agent.logger_callback()
                agent.logger.record(t=t, **record)
                agent.last_t = t

            if done:
                break

            obs = next_obs

        self.close()
        agent.close()


def exp4():
    """This experimet uses a simple linear model to learn the optimal policy
    with arbitrary initial policy.
    """

    class Env(LearningEnv):
        def __init__(self):
            super().__init__(**vars(cfg.env_kwargs))
            wingrock.load_config()
            self.x = wingrock.System()
            self.x.unc = lambda t, x: 0

            self.A = wingrock.cfg.Am
            self.B = wingrock.cfg.B
            self.Kopt, self.Popt = LQR.clqr(self.A, self.B, cfg.Q, cfg.R)
            self.behave_K, _ = LQR.clqr(self.A - 3, self.B, cfg.Qb, cfg.Rb)

            self.logger = fym.logging.Logger(Path(cfg.dir, "env.h5"))
            self.logger.set_info(cfg=cfg)

        def behavior(self, t, x):
            un = - self.behave_K.dot(x)
            noise = np.sum([
                0.05 * (np.sin(t) + 1) * (np.cos(np.pi * t) + 1),
                - 1 * np.sin(3.1 * t + 2) + 1 * np.cos(t)**2,
            ]) * 0.1

            u = un + noise * np.exp(-0.8 * t / cfg.env_kwargs.max_t)
            return u

        def observation(self):
            t = self.clock.get()
            x = self.x.state
            u, c = self.deriv(t, x)
            xdot = self.x.deriv(t, x, u, c)
            return t, x, u, xdot

        def deriv(self, t, x):
            u = self.behavior(t, x)
            c = 0
            return u, c

        def set_dot(self, t):
            x = self.x.state
            u, c = self.deriv(t, x)
            self.x.dot = self.x.deriv(t, x, u, c)

        def logger_callback(self, i, t, y, *args):
            states = self.observe_dict(y)
            x = states["x"]
            u, c = self.deriv(t, x)
            return dict(t=t, u=u, c=c, K=self.Kopt, P=self.Popt, **states)

    def load_config():
        cfg.env_kwargs = SN()
        cfg.env_kwargs.dt = 0.01
        cfg.env_kwargs.max_t = 20

        agents.load_config()

        cfg.Agent = agents.cfg
        cfg.Agent.CommonAgent.memory_len = 2000
        cfg.Agent.CommonAgent.batch_size = 1000
        cfg.Agent.CommonAgent.train_epoch = 20
        cfg.Agent.CommonAgent.train_start = 10
        cfg.Agent.CommonAgent.train_period = 3

        cfg.Agent.SQLAgent = SN(**vars(cfg.Agent.CommonAgent))
        cfg.Agent.KLMAgent = SN(**vars(cfg.Agent.CommonAgent))

        cfg.Q = np.diag([10, 10, 1])
        cfg.R = np.diag([10])
        cfg.F = - 1 * np.eye(1)

        cfg.Qb = np.diag([1, 1, 1])
        cfg.Rb = np.diag([1])

        cfg.K_init = -5 * np.ones((1, 3))

    # Init the experiment
    expdir = Path("data/exp4")
    logs.set_logger(expdir, "train.log")
    cfg = SN()

    if False:
        """
        Data 001 ~ Data 002
        These data compares SQL and KLM for wingrock linear model
        without uncertainty but initial unstable gain.
        """
        # Data 001
        load_config()  # Load the experiment default configuration
        cfg.dir = Path(expdir, "data-001")
        cfg.label = "SQL"
        env = Env()
        agent = agents.SQLAgent(cfg.Q, cfg.R, cfg.F, K_init=cfg.K_init)
        agent.logger = fym.logging.Logger(Path(cfg.dir, "sql-agent.h5"))
        env.run(agent)

        # Data 002
        load_config()  # Load the experiment default configuration
        cfg.dir = Path(expdir, "data-002")
        cfg.label = "Kleinman"
        env = Env()
        agent = agents.KLMAgent(cfg.Q, cfg.R, K_init=cfg.K_init)
        agent.logger = fym.logging.Logger(Path(cfg.dir, "klm-agent.h5"))
        env.run(agent)

        """
        Data 003 ~ Data 004
        These data compares the two methods where
        uncertainty exists and an intinal stable gain is used.
        """
        unc = lambda t, x: 0.01 * x[0]**2 + 0.05 * x[1] * x[2]
        K_init = np.zeros((1, 3))

        # Data 003
        load_config()  # Load the experiment default configuration
        cfg.dir = Path(expdir, "data-003")
        cfg.label = "SQL"
        cfg.K_init = K_init
        env = Env()
        env.x.unc = unc
        agent = agents.SQLAgent(cfg.Q, cfg.R, cfg.F, K_init=cfg.K_init)
        agent.logger = fym.logging.Logger(Path(cfg.dir, "sql-agent.h5"))
        env.run(agent)

        # Data 004
        load_config()  # Load the experiment default configuration
        cfg.dir = Path(expdir, "data-004")
        cfg.label = "Kleinman"
        cfg.K_init = K_init
        env = Env()
        env.x.unc = unc
        agent = agents.KLMAgent(cfg.Q, cfg.R, K_init=cfg.K_init)
        agent.logger = fym.logging.Logger(Path(cfg.dir, "klm-agent.h5"))
        env.run(agent)

    """
    Data 005 ~ Data 006
    These data compares the two methods where
    uncertainty exists and an intinal unstable gain is used.
    """
    unc = lambda t, x: 0.1 * x[0] * np.abs(x[0]) - 0.5 * x[1] * x[2]
    # unc = lambda t, x: 0
    K_init = -np.vstack([3, 6, 7]).T

    # Data 005
    load_config()  # Load the experiment default configuration
    cfg.dir = Path(expdir, "data-005")
    cfg.label = "SQL"
    cfg.K_init = K_init
    cfg.env_kwargs.max_t = 50
    env = Env()
    env.x.unc = unc
    agent = agents.SQLAgent(cfg.Q, cfg.R, cfg.F, K_init=cfg.K_init)
    agent.logger = fym.logging.Logger(Path(cfg.dir, "sql-agent.h5"))
    env.run(agent)

    # Data 006
    load_config()  # Load the experiment default configuration
    cfg.dir = Path(expdir, "data-006")
    cfg.label = "Kleinman"
    cfg.K_init = K_init
    cfg.env_kwargs.max_t = 50
    env = Env()
    env.x.unc = unc
    agent = agents.KLMAgent(cfg.Q, cfg.R, K_init=cfg.K_init)
    agent.logger = fym.logging.Logger(Path(cfg.dir, "klm-agent.h5"))
    env.run(agent)


def exp4_plot():
    def comp(sqldir, klmdir):
        # Data 001 ~ Data 002
        sql = plot.get_data(Path(datadir, sqldir))
        klm = plot.get_data(Path(datadir, klmdir))
        data = [sql, klm]
        # data_na = []

        basestyle = dict(c="k", lw=0.7)
        refstyle = dict(basestyle, c="r", ls="--")
        klm_style = dict(basestyle, c="y", ls="-")
        sql_style = dict(basestyle, c="b", ls="-.")
        klm.style.update(klm_style)
        sql.style.update(sql_style)
        # zlearner_na.style.update(klm_style)
        # qlearner_na.style.update(sql_style)

        # Figure common setup
        t_range = (0, sql.info["cfg"].env_kwargs.max_t)

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
        [plot.vector_by_index(d, "x", 2) for d in data]
        plt.ylabel(r'$x_3$')
        # plt.ylim(-80, 80)

        plt.xlabel("Time, sec")
        plt.xlim(t_range)

        for ax in plt.gcf().get_axes():
            ax.label_outer()

        # ====================
        # Parameter estimation
        # ====================
        figsize, pos = plot.posing(3, subsize, width, top, bottom, left, hspace)
        plt.figure(figsize=figsize)

        ax = plot.subplot(pos, 0)
        [plot.vector_by_index(d, "u", 0) for d in data]
        plt.ylabel(r'$\delta_t$')

        plot.subplot(pos, 1, sharex=ax)
        plot.all(sql, "K", style=dict(refstyle, label="True"))
        for d in data:
            plot.all(
                d, "K", is_agent=True,
                style=dict(marker="o", markersize=2)
            )
        plt.ylabel(r"$\hat{K}$")
        plt.legend()
        # plt.ylim(-70, 30)

        plot.subplot(pos, 2, sharex=ax)
        plot.all(sql, "P", style=dict(sql.style, c="r", ls="--"))
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

    datadir = Path("data", "exp4")
    # comp("data-001", "data-002")
    # comp("data-003", "data-004")
    comp("data-005", "data-006")
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

            eigvals, eigvecs = np.linalg.eig(A - B.dot(K))
            # eigvec = eigvecs[:, -1]
            if np.linalg.eigvals(H22).min() > 0:
                # print(eigvec.T @ H11 @ eigvec)
                Binv = np.linalg.pinv(B)
                H11min = Q + (A - np.eye(n)).T @ Binv.T @ R @ Binv @ (A - np.eye(n))
                print(np.linalg.eigvals(H11).min())
                print(-np.linalg.eigvals(H11min).max())
                breakpoint()

            P = H11 - H21.T.dot(np.linalg.inv(H22)).dot(H21)

            next_H11 = P

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

            if ((K - next_K)**2).sum() < eps or i > maxiter:
                break

            K = next_K

        logger.close()

    # Nonstabiliing initial gain
    K = np.zeros((m, n))
    kleinman(K, "kleinman-unstable.h5")
    sql(K, "sql-unstable.h5")

    # Stabiling initial gain
    K, *_ = LQR.clqr(A, B, 2 * np.eye(n), 2 * np.eye(m))
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


def exp6():
    """This experiment uses a complex aircraft model to learn the optimal
    policy with arbitrary initial policy. The morphing aircraft in ``fym``
    is used.
    """
    from fym.models.aircraft import MorphingLon

    class Env(LearningEnv):
        def __init__(self):
            super().__init__(**vars(cfg.env_kwargs))
            self.x = MorphingLon()
            self.PI = BaseSystem()

            trims = self.x.get_trim()
            self.trim = {k: v for k, v in zip(["x", "u", "eta"], trims)}

            self.A = jacob_analytic(self.x.deriv, 0)(*trims)
            self.B = jacob_analytic(self.x.deriv, 1)(*trims)
            self.Kopt, self.Popt = LQR.clqr(self.A, self.B, cfg.Q, cfg.R)
            self.behave_K, _ = LQR.clqr(self.A, self.B, cfg.Qb, cfg.Rb)

            self.add_noise = True

        def behavior(self, t, x):
            # un = self.trim["u"] - self.behave_K.dot(x - self.trim["x"])

            if self.add_noise:
                un = self.trim["u"]
                noise = np.vstack([
                    0.2 * (np.sin(t) + 0.5) * (np.cos(np.pi * t) + 0.5),
                    - 1 * np.sin(0.31 * t + 2) + 1 * np.cos(2 * t),
                ]) * 0.02
                noise = noise * np.exp(-0.8 * t / cfg.env_kwargs.max_t)
            else:
                un = self.trim["u"] - self.behave_K.dot(x - self.trim["x"])
                noise = 0

            return un + noise

        def deriv(self, t, x):
            u = self.behavior(t, x)
            eta = self.trim["eta"]
            return u, eta

        def observation(self):
            t = self.clock.get()
            x = self.x.state
            u, eta = self.deriv(t, x)
            xdot = self.x.deriv(x, u, eta)
            dx = x - self.trim["x"]
            du = u - self.trim["u"]
            return t, dx, du, xdot

        def set_dot(self, t):
            x = self.x.state
            u, eta = self.deriv(t, x)
            self.x.dot = self.x.deriv(x, u, eta)
            self.PI.dot = self.get_cost(x, u)

        def get_cost(self, x, u):
            dx = x - self.trim["x"]
            du = u - self.trim["u"]
            return 0.5 * (dx.T @ cfg.Q @ dx + du.T @ cfg.R @ du)

        def logger_callback(self, i, t, y, *args):
            states = self.observe_dict(y)
            x = states["x"]
            u, eta = self.deriv(t, x)
            return dict(t=t, u=u, eta=eta, K=self.Kopt, P=self.Popt, **states)

    def load_config():
        cfg.env_kwargs = SN()
        cfg.env_kwargs.dt = 0.01
        cfg.env_kwargs.max_t = 40

        agents.load_config()
        agents.cfg.CommonAgent.memory_len = 4000
        agents.cfg.CommonAgent.batch_size = 2000
        agents.cfg.CommonAgent.train_epoch = 100
        agents.cfg.CommonAgent.train_start = 20
        agents.cfg.CommonAgent.train_period = 2

        agents.cfg.SQLAgent = SN(**vars(agents.cfg.CommonAgent))
        agents.cfg.KLMAgent = SN(**vars(agents.cfg.CommonAgent))

        cfg.Q = np.diag([10, 10, 1, 10])
        cfg.R = np.diag([1000, 1])
        cfg.F = - 1 * np.eye(2)

        cfg.Qb = np.diag([1, 1, 1, 10])
        cfg.Rb = np.diag([1000, 1])

        cfg.K_init = np.zeros((2, 4))

        cfg.test = SN()
        cfg.test.dataname_learnt = "test-learnt-env.h5"
        cfg.test.dataname_lqr = "test-lqr-env.h5"
        cfg.test.initial_perturb = np.vstack((3, 0.1, 0.1, 0.1))

    def test(env):
        env.reset()

        while True:
            env.render()

            *_, done = env.step(None)

            if done:
                break

        env.close()

    # Init the experiment
    expdir = Path("data/exp6")
    logs.set_logger(expdir, "train.log")
    cfg = SN()

    """
    Data 001 ~ Data 002
    This sub-experiment compares SQL and KLM for morphing aircraft
    using an initial unstable policy
    """
    # ------ Data 001 ------ #
    load_config()  # Load the experiment default configuration
    cfg.dir = Path(expdir, "data-001")
    cfg.label = "SQL"

    # ------ Train ------ #
    env = Env()
    agent = agents.SQLAgent(cfg.Q, cfg.R, cfg.F)
    # Set loggeres
    env.logger = fym.logging.Logger(Path(cfg.dir, "env.h5"))
    env.logger.set_info(cfg=cfg)
    agent.logger = fym.logging.Logger(Path(cfg.dir, "sql-agent.h5"), max_len=1)
    env.run(agent)

    # ------ Test ------ #
    env = Env()
    env.logger = fym.logging.Logger(Path(cfg.dir, cfg.test.dataname_learnt))
    env.behave_K = fym.logging.load(Path(cfg.dir, "sql-agent.h5"))["K"][-1]
    env.add_noise = False
    env.x.initial_state = env.trim["x"] + cfg.test.initial_perturb
    test(env)

    env = Env()
    env.logger = fym.logging.Logger(Path(cfg.dir, cfg.test.dataname_lqr))
    env.behave_K = env.Kopt
    env.add_noise = False
    env.x.initial_state = env.trim["x"] + cfg.test.initial_perturb
    test(env)

    # ------ Data 002 ------ #
    load_config()  # Load the experiment default configuration
    cfg.dir = Path(expdir, "data-002")
    cfg.label = "Kleinman"

    # ------ Train ------ #
    env = Env()
    agent = agents.KLMAgent(cfg.Q, cfg.R)
    # Set loggeres
    env.logger = fym.logging.Logger(Path(cfg.dir, "env.h5"))
    env.logger.set_info(cfg=cfg)
    agent.logger = fym.logging.Logger(Path(cfg.dir, "klm-agent.h5"), max_len=1)
    env.run(agent)

    # ------ Test ------ #
    env = Env()
    env.logger = fym.logging.Logger(Path(cfg.dir, cfg.test.dataname_learnt))
    env.behave_K = fym.logging.load(Path(cfg.dir, "klm-agent.h5"))["K"][-1]
    env.add_noise = False
    env.x.initial_state = env.trim["x"] + cfg.test.initial_perturb
    test(env)


def exp6_plot():
    def get_data(name, style=dict(), with_info=False):
        path = Path(datadir, name)
        style = datastyle | style

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

    # ------ Exp Setup ------ #
    expdir = Path("data", "exp6")
    basestyle = dict(c="k", lw=0.7)
    refstyle = basestyle | dict(c="r", ls="--")
    sql_style = basestyle | dict(c="b", ls="-.")
    klm_style = basestyle | dict(c="g", ls="-.")
    test_style = basestyle | dict(c="k", ls="--")

    # ------ Data 001 ------ #
    datadir = Path(expdir, "data-001")
    datastyle = sql_style
    sql_env = get_data("env.h5", with_info=True)
    sql_agent = get_data("sql-agent.h5", style=sql_env.style)
    sql_test = get_data("test-learnt-env.h5", style=sql_env.style)

    datastyle = test_style
    lqr_test = get_data("test-lqr-env.h5", style=dict(label="LQR"))

    # ------ Data 002 ------ #
    datadir = Path(expdir, "data-002")
    datastyle = klm_style
    klm_env = get_data("env.h5", with_info=True)
    klm_agent = get_data("klm-agent.h5", style=klm_env.style)
    klm_test = get_data("test-learnt-env.h5", style=klm_env.style)

    data_train = [sql_env, klm_env]
    data_agent = [sql_agent, klm_agent]
    data_test = [sql_test, klm_test, lqr_test]

    # Figure common setup
    t_range = (0, sql_env.info["cfg"].env_kwargs.max_t)

    # All in inches
    subsize = (4.05, 0.946)
    width, top, bottom, left = (4.94, 0.2, 0.671765, 0.5487688)
    hspace = 0.2716

    # ============================
    # States and inputs (Training)
    # ============================
    figsize, pos = plot.posing(6, subsize, width, top, bottom, left, hspace)
    plt.figure(figsize=figsize)

    ax = plot.subplot(pos, 0)
    [plot.vector_by_index(d, "x", 0)[0] for d in data_train]
    plt.ylabel(r"$V_T$, m/s")
    # plt.ylim(19, 23)
    plt.legend()

    plot.subplot(pos, 1, sharex=ax)
    [plot.vector_by_index(d, "x", 1, mult=np.rad2deg(1)) for d in data_train]
    plt.ylabel(r"$\alpha$, deg")
    # plt.ylim(-5, 8)

    plot.subplot(pos, 2, sharex=ax)
    [plot.vector_by_index(d, "x", 2, mult=np.rad2deg(1)) for d in data_train]
    plt.ylabel(r"$q$, deg/s")
    # plt.ylim(-50, 50)

    plot.subplot(pos, 3, sharex=ax)
    [plot.vector_by_index(d, "x", 3, mult=np.rad2deg(1)) for d in data_train]
    plt.ylabel(r"$\gamma$, deg")
    # plt.ylim(-5, 23)

    plot.subplot(pos, 4, sharex=ax)
    [plot.vector_by_index(d, "u", 0) for d in data_train]
    plt.ylabel(r"$\delta_t$")
    # plt.ylim(0, 0.2)

    plot.subplot(pos, 5, sharex=ax)
    [plot.vector_by_index(d, "u", 1, mult=np.rad2deg(1)) for d in data_train]
    plt.ylabel(r'$\delta_e$, deg')
    # plt.ylim(-15, 5)

    plt.xlabel("Time, sec")
    plt.xlim(t_range)

    for ax in plt.gcf().get_axes():
        ax.label_outer()

    # ===============================
    # Parameter estimation (Training)
    # ===============================
    figsize, pos = plot.posing(2, subsize, width, top, bottom, left, hspace)
    plt.figure(figsize=figsize)

    plot.subplot(pos, 0, sharex=ax)
    plot.all(sql_env, "K", style=dict(refstyle, label="True"))
    for d in data_agent:
        plot.all(
            d, "K", is_agent=True,
            style=dict(marker="o", markersize=2)
        )
    plt.ylabel(r"$\hat{K}$")
    plt.legend()
    plt.ylim(-15, 7)

    plot.subplot(pos, 1, sharex=ax)
    plot.all(sql_env, "P", style=dict(sql_env.style, c="r", ls="--"))
    for d in data_agent:
        plot.all(
            d, "P", is_agent=True,
            style=dict(marker="o", markersize=2)
        )
    plt.ylabel(r"$\hat{P}$")
    plt.ylim(-12, 28)

    plt.xlabel("Time, sec")
    plt.xlim(t_range)

    for ax in plt.gcf().get_axes():
        ax.label_outer()

    # ========================
    # States and inputs (Test)
    # ========================
    figsize, pos = plot.posing(6, subsize, width, top, bottom, left, hspace)
    plt.figure(figsize=figsize)

    ax = plot.subplot(pos, 0)
    [plot.vector_by_index(d, "x", 0)[0] for d in data_test]
    plt.ylabel(r"$V_T$, m/s")
    plt.ylim(19, 23)
    plt.legend()

    plot.subplot(pos, 1, sharex=ax)
    [plot.vector_by_index(d, "x", 1, mult=np.rad2deg(1)) for d in data_test]
    plt.ylabel(r"$\alpha$, deg")
    plt.ylim(-5, 8)

    plot.subplot(pos, 2, sharex=ax)
    [plot.vector_by_index(d, "x", 2, mult=np.rad2deg(1)) for d in data_test]
    plt.ylabel(r"$q$, deg/s")
    plt.ylim(-50, 50)

    plot.subplot(pos, 3, sharex=ax)
    [plot.vector_by_index(d, "x", 3, mult=np.rad2deg(1)) for d in data_test]
    plt.ylabel(r"$\gamma$, deg")
    plt.ylim(-5, 23)

    plot.subplot(pos, 4, sharex=ax)
    [plot.vector_by_index(d, "u", 0) for d in data_test]
    plt.ylabel(r"$\delta_t$")
    plt.ylim(0, 0.2)

    plot.subplot(pos, 5, sharex=ax)
    [plot.vector_by_index(d, "u", 1, mult=np.rad2deg(1)) for d in data_test]
    plt.ylabel(r'$\delta_e$, deg')
    plt.ylim(-15, 5)

    plt.xlabel("Time, sec")
    plt.xlim(0, 5)

    for ax in plt.gcf().get_axes():
        ax.label_outer()

    # =================
    # Performance Index
    # =================
    figsize, pos = plot.posing(1, subsize, width, top, bottom, left, hspace)
    plt.figure(figsize=figsize)

    plot.subplot(pos, 0)
    [plot.vector_by_index(d, "PI", 0)[0] for d in data_test]
    plt.ylabel(r"Performance Index")
    plt.ylim(-1, 20)
    plt.legend()

    plt.xlabel("Time, sec")
    plt.xlim(0, 5)

    for ax in plt.gcf().get_axes():
        ax.label_outer()

    # # ==================================
    # # States and inputs (Non-Admissible)
    # # ==================================
    # figsize, pos = plot.posing(5, subsize, width, top, bottom, left, hspace)
    # plt.figure(figsize=figsize)

    # ax = plot.subplot(pos, 0)
    # [plot.vector_by_index(d, "x", 0)[0] for d in data_na]
    # plt.ylabel(r"$x_1$")
    # # plt.ylim(-2, 2)
    # plt.legend()

    # plot.subplot(pos, 1, sharex=ax)
    # [plot.vector_by_index(d, "x", 1) for d in data_na]
    # plt.ylabel(r"$x_2$")
    # # plt.ylim(-2, 2)

    # plot.subplot(pos, 2, sharex=ax)
    # [plot.vector_by_index(d, "u", 0) for d in data_na]
    # plt.ylabel(r'$u$')
    # # plt.ylim(-80, 80)

    # # =====================================
    # # Parameter estimation (Non-Admissible)
    # # =====================================
    # ax = plot.subplot(pos, 3)
    # plot.all(qlearner_na, "K", style=dict(refstyle, label="True"))
    # for d in data_na:
    #     plot.all(
    #         d, "K", is_agent=True,
    #         style=dict(marker="o", markersize=2)
    #     )
    # plt.ylabel(r"$\hat{K}$")
    # plt.legend()
    # # plt.ylim(-70, 30)

    # plot.subplot(pos, 4, sharex=ax)
    # plot.all(qlearner_na, "P", style=refstyle)
    # for d in data_na:
    #     plot.all(
    #         d, "P", is_agent=True,
    #         style=dict(marker="o", markersize=2)
    #     )
    # plt.ylabel(r"$\hat{P}$")
    # # plt.ylim(-70, 30)

    # plt.xlabel("Time, sec")
    # plt.xlim(t_range)

    # for ax in plt.gcf().get_axes():
    #     ax.label_outer()

    imgdir = Path("img", datadir.relative_to("data"))
    imgdir.mkdir(exist_ok=True)

    plt.figure(1)
    plt.savefig(Path(imgdir, "figure_1.pdf"), bbox_inches="tight")

    plt.figure(2)
    plt.savefig(Path(imgdir, "figure_2.pdf"), bbox_inches="tight")

    plt.figure(3)
    plt.savefig(Path(imgdir, "figure_3.pdf"), bbox_inches="tight")

    plt.show()


def exp7():
    """Experiment 7
    This experiment demonstrates the performance of SQL
    for an unknown quadrotor model.

    The LQR controller is designed based on
    [Dydek et al., 2013](doi.org/10.1109/TCST.2012.2200104).
    The performance outputs are ``pos`` and ``psi``.
    The observed states are ``pos``, ``angle`` and their derivatives.
    where the controller takes angles and altitude and returns thrusts of
    each rotor.
    """
    from fym.models.quadrotor import Quadrotor

    class Env(BaseEnv):
        def __init__(self):
            super().__init__(**vars(cfg.env.kwargs))
            self.plant = Quadrotor(**vars(cfg.quad.init))

            # Get the linear model
            self.xtrim, self.utrim = self.get_trims(alt=1)
            self.A = jacob_analytic(self.deriv, 0)(self.xtrim, self.utrim)
            self.B = jacob_analytic(self.deriv, 1)(self.xtrim, self.utrim)

            # Get the optimal gain
            self.K, self.P = LQR.clqr(self.A, self.B, cfg.agent.Q, cfg.agent.R)

            # Base controller (returns 0)
            gain = np.zeros_like(self.B.T)
            self.controller = NoisyLQR(gain, self.xtrim, self.utrim)

        def set_random_behavior(self):
            A = self.A + 0 * np.random.randn(*self.A.shape)
            B = self.B + 0 * np.random.randn(*self.B.shape)
            Q = np.diag(np.random.rand(cfg.agent.Q.shape[0]))
            R = np.diag(np.random.rand(cfg.agent.R.shape[0]))
            # R = np.random.rand() * cfg.agent.R
            gain, _ = LQR.clqr(A, B, Q, R)
            amp, freq, phase = np.random.rand(3, 4, 1)
            noise = lambda t: 0.5 * amp * np.sin(freq * t + phase)
            self.controller = NoisyLQR(gain, self.xtrim, self.utrim, noise)

            cfg.behavior = (gain, amp, freq, phase)

        def step(self):
            *_, done = self.update()
            return done

        def set_dot(self, t):
            plant_states = self.plant.observe_list()
            *_, u = self._calc(t, plant_states)
            self.plant.set_dot(t, u)

        def get_trims(self, alt=1):
            pos = np.vstack((0, 0, -alt))
            vel = angle = omega = np.zeros((3, 1))
            u = np.vstack([self.plant.m * self.plant.g / 4] * 4)
            return np.vstack((pos, vel, angle, omega)), u

        def omega2dangle(self, omega, phi, theta):
            dangle = np.array([
                [1, np.sin(phi)*np.tan(theta), np.cos(phi)*np.tan(theta)],
                [0, np.cos(phi), -np.sin(phi)],
                [0, np.sin(phi)/np.cos(theta), np.cos(phi)/np.cos(theta)]
            ]) @ omega
            return dangle

        def deriv(self, x, u):
            pos, vel, angle, omega = x[:3], x[3:6], x[6:9], x[9:12]
            R = self.plant.angle2R(angle)
            dpos, dvel, _, domega = self.plant.deriv(pos, vel, R, omega, u)
            phi, theta, _ = angle.ravel()
            dangle = self.omega2dangle(omega, phi, theta)
            xdot = np.vstack((dpos, dvel, dangle, domega))
            return xdot

        def eager_stop(self, t_hist, ode_hist):
            done = False
            for i, (t, y) in enumerate(zip(t_hist, ode_hist)):
                plant_states = self.plant.observe_list(y[self.plant.flat_index])
                angle = np.vstack(self.plant.R2angle(plant_states[2]))
                if np.any(np.abs(angle) > cfg.quad.angle_lim):
                    t_hist = t_hist[:i+1]
                    ode_hist = ode_hist[:i+1]
                    done = True
                    breakpoint()
                    break
            return t_hist, ode_hist, done

        def logger_callback(self, t):
            state_dict = self.observe_dict()
            plant_states = self.plant.observe_list()
            dx, du, xdot, u = self._calc(t, plant_states)
            return dict(t=t, dx=dx, du=du, xdot=xdot, u=u, **state_dict)

        def _calc(self, t, plant_states):
            """states: list, returns (dx, du, xdot) for training"""
            pos, vel, R, omega = plant_states
            angle = np.vstack(self.plant.R2angle(R))
            x = np.vstack((pos, vel, angle, omega))
            u = self.controller(t, x)
            xdot = self.deriv(x, u)
            dx = x - self.xtrim
            du = u - self.utrim
            return dx, du, xdot, u

    class NoisyLQR:
        def __init__(self, gain, xtrim, utrim, noise=lambda t: 0):
            """gain: constant gain, noise: function of time"""
            self.gain = gain
            self.xtrim = xtrim
            self.utrim = utrim
            self.noise = noise

        def __call__(self, t, x):
            return self.utrim - self.gain @ (x - self.xtrim) + self.noise(t)

    def sample(env, naming_rule):
        for i in range(cfg.sample.n_trial):
            env.logger = fym.logging.Logger(
                Path(cfg.path.sample, naming_rule % i))

            env.set_random_behavior()

            env.logger.set_info(trial=i, cfg=cfg)
            env.reset()

            while True:
                env.render()
                done = env.step()

                if done:
                    break

            env.close()

    def train():
        logger = logging.getLogger("logs")

        agents.load_config()
        agent = agents.SQLAgent(
            Q=cfg.agent.Q, R=cfg.agent.R, F=-cfg.train.s*np.eye(cfg.agent.R.shape[0]))
        agent.logger = fym.logging.Logger(Path(cfg.path.train, "result.h5"))

        # Random sample N data
        memory = []
        for path in cfg.path.sample.glob("*.h5"):
            data, info = fym.logging.load(path, with_info=True)
            memory += [
                (dx, du, xdot)
                for dx, du, xdot
                in zip(data["dx"], data["du"], data["xdot"])]

        batch = random.sample(memory, int(len(memory) / 2))
        for i in range(cfg.train.n_step):
            info = agent.step(batch)
            agent.logger.record(i=i, **info)

            # P real eigenvalues
            PRE = np.linalg.eigvals(info["P"]).real
            # CRE = np.linalg.eigvals(self.A - self.B.dot(agent.K)).real
            logger.info(
                f"[{type(agent).__name__}] "
                f"Iter: {i} | "
                f"Min PRE: {PRE.min():.2e} | "
                f"no. PNE: {len(PRE[PRE < 0])} / {len(PRE)} | "
            )

        agent.logger.close()

    def test(train_path):
        def logger_callback(self, t):
            state_dict = self.observe_dict()
            plant_states = self.plant.observe_list()
            dx, du, xdot, u = self._calc(t, plant_states)
            return dict(t=t, dx=dx, du=du, xdot=xdot, u=u, **state_dict)

        cfg.quad.init = SN(**vars(cfg.quad.init) | vars(cfg.test.init))
        cfg.env.kwargs = SN(**vars(cfg.env.kwargs) | vars(cfg.test.kwargs))

        env = Env()
        env.logger = fym.logging.Logger(Path(cfg.path.test, "env.h5"))

        train_data = fym.logging.load(train_path)
        env.controller = NoisyLQR(train_data["K"][-1], env.xtrim, env.utrim)

        env.logger.set_info(cfg=cfg)
        env.reset()

        while True:
            env.render()
            done = env.step()

            if done:
                break

        env.close()

    def load_config():
        cfg.path = SN()
        cfg.path.exp = Path("data", "exp7")
        cfg.path.sample = Path(cfg.path.exp, "sampled_data")
        cfg.path.train = Path(cfg.path.exp, "train")
        cfg.path.test = Path(cfg.path.exp, "test")

        cfg.env = SN()
        cfg.env.kwargs = SN()
        cfg.env.kwargs.dt = 10
        cfg.env.kwargs.max_t = 20
        cfg.env.kwargs.solver = "odeint"
        cfg.env.kwargs.ode_step_len = 1000

        cfg.quad = SN()
        cfg.quad.init = SN()
        cfg.quad.init.pos = np.vstack((0, 0, -1))
        cfg.quad.init.vel = np.zeros((3, 1))
        cfg.quad.init.R = np.eye(3)
        cfg.quad.init.omega = np.zeros((3, 1))
        cfg.quad.angle_lim = np.deg2rad([30, 30, np.inf])

        cfg.sample = SN()
        cfg.sample.n_trial = 20

        cfg.agent = SN()
        # cfg.agent.Q = np.diag([1]*3 + [0]*3 + [1]*3 + [0]*3)
        cfg.agent.Q = np.diag([
            1, 1, 1,
            0, 0, 0,
            100, 100, 100,
            1, 1, 1
        ])
        # cfg.agent.Q = np.diag(np.ones(12))
        cfg.agent.R = np.diag([10, 10, 10, 10])

        cfg.train = SN()
        cfg.train.n_batch = 1000
        cfg.train.n_step = 100
        cfg.train.s = 1

        cfg.test = SN()
        cfg.test.init = SN()
        cfg.test.init.pos = np.vstack((0.1, 0.1, -0.9))
        cfg.test.init.omega = np.vstack((0.2, 0.2, -0.1))
        cfg.test.kwargs = SN()
        cfg.test.kwargs.dt = 10
        cfg.test.kwargs.max_t = 50
        cfg.test.kwargs.solver = "odeint"
        cfg.test.kwargs.ode_step_len = 1000

    cfg = SN()
    load_config()
    random.seed(0)
    np.random.seed(0)

    # # ------ Sampling ------#
    # cfg.env.kwargs.max_t = 20
    # env = Env()
    # sample(env, "sample_%02d.h5")

    # # ------ Traning ------#
    # logs.set_logger(cfg.path.exp, "train.log")
    # train()

    # # ------ Test ------#
    # test(train_path=Path(cfg.path.train, "result.h5"))


def exp7_plot():
    from matplotlib.patches import Circle
    import mpl_toolkits.mplot3d.art3d as art3d

    class Quadrotor:
        def __init__(self, ax):
            # Parameters
            d = 0.315
            r = 0.15

            # Body
            body_segs = np.array([
                [[d, 0, 0], [0, 0, 0]],
                [[-d, 0, 0], [0, 0, 0]],
                [[0, d, 0], [0, 0, 0]],
                [[0, -d, 0], [0, 0, 0]]
            ])
            colors = (
                (1, 0, 0, 1),
                (0, 0, 1, 1),
                (0, 0, 1, 1),
                (0, 0, 1, 1),
            )

            self.body = art3d.Line3DCollection(
                body_segs, colors=colors, linewidth=3)

            kwargs = dict(radius=r, ec="k", fc="k", alpha=0.3)
            self.rotors = [
                Circle((d, 0), **kwargs),
                Circle((0, d), **kwargs),
                Circle((-d, 0), **kwargs),
                Circle((0, -d), **kwargs),
            ]

            ax.add_collection3d(self.body)
            for rotor in self.rotors:
                ax.add_patch(rotor)
                art3d.pathpatch_2d_to_3d(rotor, z=0)

            self.body._base = self.body._segments3d
            for rotor in self.rotors:
                rotor._segment3d = np.array(rotor._segment3d)
                rotor._center = np.array(rotor._center + (0,))
                rotor._base = rotor._segment3d

        def set(self, pos, R=np.eye(3)):
            # Rotate
            self.body._segments3d = np.array([
                R @ point
                for point in self.body._base.reshape(-1, 3)
            ]).reshape(self.body._base.shape)

            for rotor in self.rotors:
                rotor._segment3d = np.array([
                    R @ point for point in rotor._base
                ])

            # Translate
            self.body._segments3d = self.body._segments3d + pos

            for rotor in self.rotors:
                rotor._segment3d = rotor._segment3d + pos

    class Animator:
        def __init__(self, fig, datalist):
            self.offsets = ['collections', 'patches', 'lines',
                            'texts', 'artists', 'images']
            self.fig = fig
            self.axes = fig.axes
            self.datalist = datalist

        def init(self):
            self.frame_artists = []

            for ax in self.axes:
                ax.quad = Quadrotor(ax)

                # set an axis
                ax.set_xlim3d(-0.5, 0.5)
                ax.set_ylim3d(-0.5, 0.5)
                ax.set_zlim3d(0.5, 1.5)
                ax.set_xticks((-0.5, 0.5))
                ax.set_yticks((-0.5, 0.5))
                ax.set_zticks((0.5, 1.5))

                ax.tick_params(labelsize=6)

                for name in self.offsets:
                    self.frame_artists += getattr(ax, name)

            self.fig.subplots_adjust(
                left=0.1, right=0.9, top=0.9, bottom=0.1,
                wspace=0.2, hspace=0.2)

            return self.frame_artists

        def get_sample(self):
            self.init()
            self.update(0)
            self.fig.show()

        def update(self, frame):
            for data, ax in zip(self.datalist, self.axes):
                pos = - data["plant"]["pos"][frame].squeeze()
                R = data["plant"]["R"][frame].squeeze()
                ax.quad.set(pos, R)

            return self.frame_artists

        def animate(self, *args, **kwargs):
            frames = range(0, len(self.datalist[0]["t"]), 10)

            self.anim = FuncAnimation(
                self.fig, self.update, init_func=self.init,
                frames=frames, interval=10, blit=True,
                *args, **kwargs)

        def save(self, fname, *args, **kwargs):
            self.anim.save(Path(imgdir, fname), writer="ffmpeg", fps=30,
                           *args, **kwargs)

    def plot_sampled_data():
        datalist = np.array([
            data | info
            for data, info in map(
                lambda x: fym.logging.load(x, with_info=True),
                sampledir.glob("*.h5"))
        ])

        fig, axes = plt.subplots(4, 5, subplot_kw=dict(projection="3d"))

        animator = Animator(fig, datalist)
        # animator.get_sample()
        animator.animate()
        animator.save("animation.mp4")

    def plot_train_data():
        # ------ Train ------ #
        data, info = fym.logging.load(trainpath, with_info=True)
        _, cfginfo = fym.logging.load(Path(expdir, "cfg.h5"), with_info=True)
        cfg = cfginfo["cfg"]

        shape = (len(data["i"]), -1)
        plt.figure()

        ax = plt.subplot(211)
        line, *_ = plt.plot(
            data["i"], data["P"].reshape(shape), "k-")
        for p in cfg.P.ravel():
            refline = plt.axhline(p, c="r", ls="--")
        plt.ylabel(r"$P$")
        plt.legend([refline, line], ["True", "Trained"])

        plt.subplot(212, sharex=ax)
        plt.plot(data["i"], data["K"].reshape(shape), "k-")
        for k in cfg.K.ravel():
            plt.axhline(k, c="r", ls="--")
        plt.ylabel(r"$K$")
        plt.xlabel("Iteration")
        plt.xlim(data["i"][[0, -1]])

        plt.savefig(Path(imgdir, "train.pdf"))
        plt.show()

    def plot_test_data():
        # ------ Test ------ #
        data, info = fym.logging.load(testpath, with_info=True)

        fig = plt.figure()
        plt.subplot(111, projection="3d")

        animator = Animator(fig, [data | info])
        # animator.get_sample()
        animator.animate()
        animator.save("test-animation.mp4")

#         fig, axes = plt.subplots()

#         # All states
#         plt.plot(data["t"], data["dx"].squeeze())
#         plt.show()

    # ------ Set paths ------ #
    expdir = Path("data", "exp7")
    sampledir = Path(expdir, "sampled_data")
    trainpath = Path(expdir, "train", "result.h5")
    testpath = Path(expdir, "test", "env.h5")

    imgdir = Path("img", expdir.relative_to("data"))
    imgdir.mkdir(exist_ok=True)

    # plot_sampled_data()
    plot_train_data()
    # plot_test_data()


def main():
    # exp1()
    # exp1_plot()

    # exp2()
    # exp2_plot()

    # exp3()
    # exp3_plot()

    # exp4()
    # exp4_plot()

    # exp5()
    # exp5_plot()

    # exp6()
    # exp6_plot()

    # exp7()
    exp7_plot()


if __name__ == "__main__":
    main()
