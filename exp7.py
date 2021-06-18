import numpy as np
from pathlib import Path
from types import SimpleNamespace as SN
import logging
import random
import time

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle
import mpl_toolkits.mplot3d.art3d as art3d


from fym.core import BaseEnv, BaseSystem
from fym.utils.linearization import jacob_analytic
from fym.agents import LQR
from fym.utils import parser
import fym.logging
from fym.models.quadrotor import Quadrotor

import agents
import logs


cfg = parser.parse()


def load_config():
    fym.parser.update(cfg, {"path.base": Path("data", "exp7")})
    fym.parser.update(cfg, {
        "path": {
            "sample": Path(cfg.path.base, "sampled_data"),
            "train.sql": Path(cfg.path.base, "train", "result-sql.h5"),
            "train.klm": Path(cfg.path.base, "train", "result-klm.h5"),
            "train.sql_stable": Path(cfg.path.base, "train", "result-sql-stable.h5"),
            "train.klm_stable": Path(cfg.path.base, "train", "result-klm-stable.h5"),
            "test.sql": Path(cfg.path.base, "test", "env-sql.h5"),
            "test.klm": Path(cfg.path.base, "test", "env-klm.h5"),
            "test.sql_stable": Path(cfg.path.base, "test", "env-sql-stable.h5"),
            "test.klm_stable": Path(cfg.path.base, "test", "env-klm-stable.h5"),
            "img": Path("img", cfg.path.base.relative_to("data")),
        },
        "env.kwargs": {
            "dt": 10,
            "max_t": 20,
            "solver": "odeint",
            "ode_step_len": 1000
        },
        "quad.init": {
            "pos": np.vstack((0, 0, -1)),
            "vel": np.zeros((3, 1)),
            "R": np.eye(3),
            "omega": np.zeros((3, 1))
        },
        "quad.angle_lim": np.deg2rad([70, 70, np.inf]),
        "sample.n_trial": 20,
        "agent": {
            "Q": np.diag([
                1, 1, 1,
                0, 0, 0,
                100, 100, 100,
                1, 1, 1
            ]),
            "R": np.diag([10, 10, 10, 10]),
        },
        "train": {
            "n_batch": 1000,
            "n_step": 100,
            "s": 1,
        },
        "test": {
            "init.pos": np.vstack((0.1, 0.1, -0.9)),
            "init.omega": np.vstack((0.2, 0.2, -0.1)),
            "kwargs": {
                "dt": 0.01,
                "max_t": 50,
                "solver": "rk4",
                "ode_step_len": 1,
            },
        },
    })


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

    def get_random_stable_gain(self):
        A = self.A + 0 * np.random.randn(*self.A.shape)
        B = self.B + 0 * np.random.randn(*self.B.shape)
        Q = np.diag(np.random.rand(cfg.agent.Q.shape[0]))
        R = np.diag(np.random.rand(cfg.agent.R.shape[0]))
        # R = np.random.rand() * cfg.agent.R
        gain, _ = LQR.clqr(A, B, Q, R)
        return gain

    def set_random_behavior(self):
        gain = self.get_random_stable_gain()
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

    def eager_stop(self, t):
        angle = np.vstack(self.plant.R2angle(self.plant.R.state))
        return np.any(np.abs(angle) > cfg.quad.angle_lim)

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


def train(agent):
    logger = logging.getLogger("logs")

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


def test(trainpath, savepath):
    def logger_callback(self, t):
        state_dict = self.observe_dict()
        plant_states = self.plant.observe_list()
        dx, du, xdot, u = self._calc(t, plant_states)
        return dict(t=t, dx=dx, du=du, xdot=xdot, u=u, **state_dict)

    cfg.quad.init = SN(**vars(cfg.quad.init) | vars(cfg.test.init))
    cfg.env.kwargs = SN(**vars(cfg.env.kwargs) | vars(cfg.test.kwargs))

    env = Env()
    env.logger = fym.logging.Logger(savepath)

    train_data = fym.logging.load(trainpath)
    env.controller = NoisyLQR(train_data["K"][-1], env.xtrim, env.utrim)

    env.logger.set_info(cfg=cfg)
    env.reset()

    while True:
        env.render()
        done = env.step()

        if done:
            break

    env.close()


def plot_sampled_data():
    datalist = np.array([
        data | info
        for data, info in map(
            lambda x: fym.logging.load(x, with_info=True),
            cfg.path.sample.glob("*.h5"))
    ])

    fig, axes = plt.subplots(4, 5, subplot_kw=dict(projection="3d"))

    animator = Animator(fig, datalist)
    # animator.get_sample()
    animator.animate()
    animator.save("animation.mp4")


def plot_train_data():
    import pandas as pd
    import seaborn as sns

    def plot_train_hist(initial="unstable"):

        if initial == "unstable":
            sqlpath = cfg.path.train.sql
            klmpath = cfg.path.train.klm
        elif initial == "stable":
            sqlpath = cfg.path.train.sql_stable
            klmpath = cfg.path.train.klm_stable

        # ------ Train ------ #
        sql = fym.parser.parse({
            "data": fym.parser.parse(fym.load(sqlpath)),
        })
        klm = fym.parser.parse({
            "data": fym.parser.parse(fym.load(klmpath)),
        })
        _, info = fym.logging.load(Path(cfg.path.base, "cfg.h5"), with_info=True)
        ref = fym.parser.parse({
            "data": {
                "P": info["cfg"].P,
                "K": info["cfg"].K,
            },
        })

        color_palette = sns.color_palette()
        sql.style = dict(color=color_palette[0], lw=1)
        klm.style = dict(color=color_palette[2], lw=1)
        ref.style = dict(color=color_palette[3], ls="--")

        shape = (len(sql.data.i), -1)

        # ------ Figure 1 ------ #
        plt.rc("axes", grid=True)

        def plot_rolling(exp, x, window=5, alpha=0.2):
            rolling = pd.DataFrame(x).rolling(window=window)
            mean = rolling.mean().to_numpy().ravel()
            std = rolling.std().to_numpy().ravel()
            line, = plt.plot(exp.data.i, mean, **exp.style)
            plt.fill_between(exp.data.i, mean - std, mean + std,
                             **exp.style | dict(alpha=alpha, lw=0))
            return line

        fig, axes = plt.subplots(3, 2, sharex=True)

        # SQL
        plt.axes(axes[0, 0])
        [plt.axhline(p, **ref.style) for p in ref.data.P.ravel()]
        sql_line, *_ = [plot_rolling(sql, p) for p in sql.data.P.reshape(shape).T]
        plt.ylabel(r"$P$")
        plt.ylim(-100, 35000)
        plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

        plt.axes(axes[1, 0])
        [plt.axhline(p, **ref.style) for p in ref.data.K.ravel()]
        [plot_rolling(sql, k) for k in sql.data.K.reshape(shape).T]
        plt.ylabel(r"$K$")
        plt.ylim(-30, 30)

        plt.axes(axes[2, 0])
        error = np.linalg.norm(sql.data.P - ref.data.P, axis=(1, 2))
        plot_rolling(sql, error, alpha=0.3)
        plt.ylabel(r"$\|\|\tilde{P}\|\|$")
        if initial == "unstable":
            plt.ylim(0, 5e4)
        elif initial == "stable":
            plt.ylim(0, 5e3)
        plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

        plt.xlabel("Iteration")
        plt.xlim(sql.data.i[[0, -1]])

        # Kleinman
        plt.axes(axes[0, 1])
        [plt.axhline(p, **ref.style) for p in ref.data.P.ravel()]
        klm_line, *_ = [plot_rolling(klm, p) for p in klm.data.P.reshape(shape).T]
        plt.ylim(-100, 35000)
        plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

        plt.axes(axes[1, 1])
        [plt.axhline(p, **ref.style) for p in ref.data.K.ravel()]
        [plot_rolling(klm, k) for k in klm.data.K.reshape(shape).T]
        plt.ylim(-30, 30)

        plt.axes(axes[2, 1])
        error = np.linalg.norm(klm.data.P - ref.data.P, axis=(1, 2))
        plot_rolling(klm, error, alpha=0.3)
        if initial == "unstable":
            plt.ylim(0, 5e4)
        elif initial == "stable":
            plt.ylim(0, 5e3)
        plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

        plt.xlabel("Iteration")
        plt.xlim(klm.data.i[[0, -1]])

        fig.set_size_inches(8.78, 4.3)
        fig.align_ylabels(axes)
        fig.tight_layout()

        # def when_resize(event):
        #     print(fig.get_size_inches(), end="\r")

        # fig.canvas.mpl_connect("resize_event", when_resize)

        fig.legend([sql_line, klm_line], ["SQL", "Kleinman"],
                   bbox_to_anchor=[0.5, 0.91],
                   loc="lower center", ncol=2)
        fig.subplots_adjust(top=0.88)

        if initial == "unstable":
            fig.savefig(Path(cfg.path.img, "train.pdf"))
        elif initial == "stable":
            fig.savefig(Path(cfg.path.img, "train-stable.pdf"))

    # ------ Figure 2 ------ #

    # plot_train_hist(initial="unstable")
    plot_train_hist(initial="stable")

    plt.show()


class Quadrotor3D:
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
    def __init__(self, fig, datalist, verbose=False):
        self.offsets = ['collections', 'patches', 'lines',
                        'texts', 'artists', 'images']
        self.fig = fig
        self.axes = fig.axes
        self.datalist = datalist
        self.verbose = verbose
        self.sa_kwargs = dict(left=0.1, right=0.9, top=0.9, bottom=0.1,
                              wspace=0.2, hspace=0.2)

    def init(self):
        self.frame_artists = []

        for ax in self.axes:
            # Quadrotor
            ax.quad = Quadrotor3D(ax)

            # set an axis
            ax.set_xlim3d(-0.5, 0.5)
            ax.set_ylim3d(-0.5, 0.5)
            ax.set_zlim3d(0.5, 1.5)
            ax.set_xticks((-0.5, 0.5))
            ax.set_yticks((-0.5, 0.5))
            ax.set_zticks((0.5, 1.5))

            ax.tick_params(labelsize=6)

            # Verbose
            if self.verbose:
                ax.text(-1.6, -0.5, 0.3, s="Text", fontfamily="monospace")

            for name in self.offsets:
                self.frame_artists += getattr(ax, name)

        self.fig.subplots_adjust(**self.sa_kwargs)

        return self.frame_artists

    def subplots_adjust(self, **kwargs):
        dict.update(self.sa_kwargs, kwargs)

    def get_sample(self):
        self.init()
        self.update(0)
        plt.show()

    def update(self, frame):
        for data, ax in zip(self.datalist, self.axes):
            pos = -data.plant.pos[frame].squeeze()
            R = data.plant.R[frame].squeeze()
            ax.quad.set(pos, R)

            if self.verbose:
                t = data.t[frame]
                vel = data.plant.vel[frame].squeeze()
                angle = np.rad2deg(Quadrotor.R2angle(0, R))
                omega = np.rad2deg(data.plant.omega[frame].squeeze())
                ax.texts[0].set_text(
                    "\n".join([
                        f"{'t:':<7} {t:6.3f} {'[s]':>7}",
                        f"{'x:':<7} {pos[0]:6.3f} {'[m]':>7}",
                        f"{'y:':<7} {pos[1]:6.3f} {'[m]':>7}",
                        f"{'z:':<7} {pos[2]:6.3f} {'[m]':>7}",
                        f"{'vx:':<7} {vel[0]:6.3f} {'[m/s]':>7}",
                        f"{'vy:':<7} {vel[1]:6.3f} {'[m/s]':>7}",
                        f"{'vz:':<7} {vel[2]:6.3f} {'[m/s]':>7}",
                        f"{'roll:':<7} {angle[0]:6.3f} {'[deg]':>7}",
                        f"{'pitch:':<7} {angle[1]:6.3f} {'[deg]':>7}",
                        f"{'yaw:':<7} {angle[2]:6.3f} {'[deg]':>7}",
                        f"{'p:':<7} {omega[0]:6.3f} {'[deg/s]':>7}",
                        f"{'q:':<7} {omega[1]:6.3f} {'[deg/s]':>7}",
                        f"{'r:':<7} {omega[2]:6.3f} {'[deg/s]':>7}",
                    ])
                )

        return self.frame_artists

    def animate(self, frame_step=10, *args, **kwargs):
        t = self.datalist[0].t
        frames = range(0, len(t), frame_step)

        self.anim = FuncAnimation(
            self.fig, self.update, init_func=self.init,
            frames=frames, interval=10, blit=True,
            *args, **kwargs)

    def save(self, fname, *args, **kwargs):
        self.anim.save(Path(cfg.path.img, fname), writer="ffmpeg", fps=30,
                       *args, **kwargs)


def plot_test_data():
    # ------ Test ------ #
    sql = fym.parser.parse({
        "data": fym.logging.load(cfg.path.test.sql)
    })
    klm = fym.parser.parse({
        "data": fym.logging.load(cfg.path.test.klm)
    })

    sa_kwargs = dict(left=0.3, right=0.95, top=0.95, bottom=0.05)

    # SQL
    fig = plt.figure()
    plt.subplot(111, projection="3d")
    animator = Animator(fig, [sql.data], verbose=True)
    animator.subplots_adjust(**sa_kwargs)
    # animator.get_sample()
    animator.animate()
    animator.save("test-sql.mp4")

    # Kleinman
    fig = plt.figure()
    plt.subplot(111, projection="3d")
    animator = Animator(fig, [klm.data], verbose=True)
    animator.subplots_adjust(**sa_kwargs)
    # animator.get_sample()
    animator.animate(frame_step=1)
    animator.save("test-klm.mp4")


def main():
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
    load_config()
    random.seed(0)
    np.random.seed(0)

    # t0 = time.time()

    # ------ Sampling ------ #

    # cfg.env.kwargs.max_t = 20
    # env = Env()
    # sample(env, "sample_%02d.h5")

    # ------ Training ------#

    logs.set_logger(cfg.path.base, "train.log")

    # ------ Training Unstable: SQL ------#
    # agents.load_config()
    # agent = agents.SQLAgent(
    #     Q=cfg.agent.Q,
    #     R=cfg.agent.R,
    #     F=-cfg.train.s*np.eye(cfg.agent.R.shape[0])
    # )
    # agent.logger = fym.logging.Logger(cfg.path.train.sql)
    # train(agent=agent)

    # ------ Training Unstable: KLM ------#
    # agents.load_config()
    # agent = agents.KLMAgent(Q=cfg.agent.Q, R=cfg.agent.R)
    # agent.logger = fym.Logger(cfg.path.train.klm)
    # train(agent=agent)

    # ------ Training Stable ------#
    env = Env()
    K = env.get_random_stable_gain()

    # ------ Training Stable: SQL ------#
    agents.load_config()
    agent = agents.SQLAgent(
        Q=cfg.agent.Q,
        R=cfg.agent.R,
        F=-cfg.train.s*np.eye(cfg.agent.R.shape[0]),
        K_init=K
    )
    agent.logger = fym.logging.Logger(cfg.path.train.sql_stable)
    train(agent=agent)

    # ------ Training Stable: KLM ------#
    agents.load_config()
    agent = agents.KLMAgent(Q=cfg.agent.Q, R=cfg.agent.R, K_init=K)
    agent.logger = fym.Logger(cfg.path.train.klm_stable)
    train(agent=agent)

    # ------ Test ------ #

    # # SQLAgent
    # test(trainpath=Path(cfg.path.train.sql),
    #      savepath=Path(cfg.path.test.sql))

    # # KLMAgent
    # test(trainpath=Path(cfg.path.train.klm),
    #      savepath=Path(cfg.path.test.klm))

    # print(f"Elapsed time is {time.time() - t0} seconds.")


def plot():
    load_config()

    # ------ Set paths ------ #
    cfg.path.img.mkdir(exist_ok=True)

    # plot_sampled_data()
    plot_train_data()
    # plot_test_data()


if __name__ == "__main__":
    # main()
    plot()
