import numpy as np
import scipy.signal
import scipy.optimize
from types import SimpleNamespace as SN
from pathlib import Path
import itertools

from fym.core import BaseEnv, BaseSystem
import fym.logging

import matplotlib
import matplotlib.pyplot as plt
from cycler import cycler

import plot

# np.warnings.filterwarnings("error", category=np.VisibleDeprecationWarning)

cfg = SN()

style = SN()
style.base = dict(c="k", lw=0.7)


def load_config():
    cfg.Am = np.array([[0, 1, 0], [-15.8, -5.6, -17.3], [1, 0, 0]])
    cfg.B = np.array([[0, 1, 0]]).T
    cfg.Br = np.array([[0, 0, -1]]).T
    cfg.x_init = np.vstack((0.3, 0, 0))
    cfg.Q_lyap = np.eye(3)
    cfg.P = scipy.linalg.solve_lyapunov(cfg.Am.T, -cfg.Q_lyap)
    cfg.R = np.diag([0.1])
    cfg.final_time = 50

    cfg.Wcirc = np.vstack((-18.59521, 15.162375, -62.45153, 9.54708, 21.45291))

    cfg.vareps = SN()
    cfg.vareps.freq = 5
    cfg.vareps.amp = 2  # 2
    cfg.vareps.offset = 0

    cfg.tauf = 1e-3
    cfg.Gamma1 = 1e4
    cfg.threshold = 1e-10

    cfg.bF = 5000
    cfg.bh = 1500
    cfg.LF_speed = 0.05
    cfg.Lh_speed = 0.05
    cfg.LF_init = 10
    cfg.Lh_init = 10

    cfg.dir = "data"

    # MRAC
    cfg.MRAC = SN()
    cfg.MRAC.env_kwargs = dict(
        solver="odeint", dt=20, max_t=cfg.final_time, ode_step_len=int(20/0.01))

    # H-Modification MRAC
    cfg.HMRAC = SN()
    cfg.HMRAC.env_kwargs = dict(
        solver="odeint", dt=20, max_t=cfg.final_time, ode_step_len=int(20/0.01))


def get_eig(A, threshold=0):
    eigs, eigv = np.linalg.eig(A)
    sort = np.argsort(np.real(eigs))  # sort in ascending order
    eigs = np.real(eigs[sort])
    eigv = np.real(eigv[:, sort])
    eigs[eigs < threshold] = 0
    return eigs, eigv


class System(BaseSystem):
    def __init__(self):
        super().__init__(cfg.x_init)
        self.unc = Uncertainty()

    def set_dot(self, t, u, c):
        x = self.state
        self.dot = (
            cfg.Am.dot(x)
            + cfg.B.dot(u + self.unc(t, x))
            + cfg.Br.dot(c)
        )


class ReferenceSystem(BaseSystem):
    def __init__(self):
        super().__init__(cfg.x_init)
        self.cmd = Command()

    def set_dot(self, c):
        xr = self.state
        self.dot = cfg.Am.dot(xr) + cfg.Br.dot(c)


class PerformanceIndex(BaseSystem):
    def set_dot(self, x, u):
        self.dot = 1/2 * (x.T.dot(cfg.Q_lyap).dot(x) + u.T.dot(cfg.R).dot(u))


class Uncertainty():
    def __init__(self):
        self.Wcirc = cfg.Wcirc

    def basis(self, x):
        return np.vstack((x[:2], np.abs(x[:2])*x[1], x[0]**3))

    def parameter(self, t):
        return self.Wcirc

    def vareps(self, t, x):
        # vareps = cfg.vareps.amp * np.sin(cfg.vareps.freq * t) + cfg.vareps.offset
        # vareps = np.tanh(np.sum(np.abs(x[:2]) * x[0]) + x[1]**3)
        vareps = np.tanh(x[1])
        vareps += np.exp(-t/10) * np.sin(5 * t)
        return cfg.vareps.amp * vareps

    def __call__(self, t, x):
        Wcirc = self.parameter(t)
        return Wcirc.T.dot(self.basis(x)) + self.vareps(t, x)


class Command():
    def __call__(self, t):
        if t > 10:
            c = scipy.signal.square([(t - 10) * 2*np.pi / 20])
        else:
            c = 0
        # return np.atleast_2d(c)
        return np.atleast_2d(np.sin(t))


class CMRAC(BaseSystem):
    def __init__(self):
        super().__init__(shape=cfg.Wcirc.shape)

    def set_dot(self, e, phi, composite_term):
        self.dot = (
            cfg.Gamma1 * phi.dot(e.T).dot(cfg.P).dot(cfg.B)
            + composite_term
        )


class MRACEnv(BaseEnv):
    def __init__(self):
        super().__init__(**cfg.MRAC.env_kwargs)
        self.x = System()
        self.xr = ReferenceSystem()
        self.W = CMRAC()
        self.J = PerformanceIndex()

        self.basis = self.x.unc.basis
        self.cmd = Command()

        self.P = scipy.linalg.solve_continuous_are(cfg.Am, cfg.B, cfg.Q_lyap, cfg.R)

        self.BTBinv = np.linalg.inv(cfg.B.T.dot(cfg.B))

        self.logger = fym.logging.Logger(Path(cfg.dir, "mrac-env.h5"))
        self.logger.set_info(cfg=cfg)

    def step(self):
        *_, done = self.update()

        return done

    def set_dot(self, t):
        x = self.x.state
        xr = self.xr.state
        W = self.W.state
        phi = self.basis(x)
        c = self.cmd(t)

        e = x - xr
        u = self.get_input(e, W, phi)

        self.x.set_dot(t, u, c)
        self.xr.set_dot(c)
        self.W.set_dot(e, phi, 0)
        self.J.set_dot(e, u)

    def get_input(self, e, W, phi):
        un = -np.linalg.inv(cfg.R).dot(cfg.B.T).dot(self.P).dot(e)
        ua = -W.T.dot(phi)
        return un + ua

    def logger_callback(self, i, t, y, t_hist, ode_hist):
        x, xr, W, J = self.observe_list(y)
        phi = self.basis(x)
        c = self.cmd(t)

        Wcirc = self.x.unc.parameter(t)

        e = x - xr
        u = self.get_input(e, W, phi)

        e_HJB = self.get_HJB_error(t, x, xr, W)

        return dict(t=t, x=x, xr=xr, W=W, Wcirc=Wcirc, e=e, c=c, u=u, J=J,
                    e_HJB=e_HJB)

    def get_HJB_error(self, t, x, xr, W):
        e = x - xr
        Wcirc = self.x.unc.parameter(t)
        phi = self.basis(x)

        B, R = cfg.B, cfg.R

        grad_V_tilde = - B.dot(self.BTBinv).dot(R).dot(W.T).dot(phi)
        u_tilde = W.T.dot(phi)

        return np.sum([
            e.T.dot(self.P).dot(cfg.B).dot(W.T.dot(phi) - Wcirc.T.dot(phi)),
            grad_V_tilde.T.dot(cfg.Am).dot(e),
            u_tilde.T.dot(R).dot(Wcirc.T).dot(phi),
            -0.5 * u_tilde.T.dot(R).dot(u_tilde)
        ])


class HMRACEnv(BaseEnv):
    def __init__(self):
        super().__init__(**cfg.HMRAC.env_kwargs)
        self.x = System()
        self.xr = ReferenceSystem()
        self.W = CMRAC()
        self.J = PerformanceIndex()

        self.basis = self.x.unc.basis
        self.cmd = Command()

        self.P = scipy.linalg.solve_continuous_are(cfg.Am, cfg.B, cfg.Q_lyap, cfg.R)

        self.BTBinv = np.linalg.inv(cfg.B.T.dot(cfg.B))

        self.logger = fym.logging.Logger(Path(cfg.dir, "mrac-env.h5"))
        self.logger.set_info(cfg=cfg)

    def step(self):
        *_, done = self.update()

        return done

    def set_dot(self, t):
        x = self.x.state
        xr = self.xr.state
        W = self.W.state
        phi = self.basis(x)
        c = self.cmd(t)

        e = x - xr
        u = self.get_input(e, W, phi)

        B = cfg.B
        composite_term = cfg.Gamma1 * (
            phi.dot(
                - e.T.dot(cfg.Am.T).dot(B).dot(np.linalg.inv(B.T.dot(B)))
                + phi.T.dot(cfg.Wcirc).dot(cfg.R)
                - phi.T.dot(W).dot(cfg.R)
            )
        )

        self.x.set_dot(t, u, c)
        self.xr.set_dot(c)
        self.W.set_dot(e, phi, composite_term)
        self.J.set_dot(e, u)

    def get_input(self, e, W, phi):
        un = -np.linalg.inv(cfg.R).dot(cfg.B.T).dot(self.P).dot(e)
        ua = -W.T.dot(phi)
        return un + ua

    def logger_callback(self, i, t, y, t_hist, ode_hist):
        x, xr, W, J = self.observe_list(y)
        phi = self.basis(x)
        c = self.cmd(t)

        Wcirc = self.x.unc.parameter(t)

        e = x - xr
        u = - W.T.dot(phi)

        e_HJB = self.get_HJB_error(t, x, xr, W)

        return dict(t=t, x=x, xr=xr, W=W, Wcirc=Wcirc, e=e, c=c, u=u, J=J,
                    e_HJB=e_HJB)

    def get_HJB_error(self, t, x, xr, W):
        e = x - xr
        Wcirc = self.x.unc.parameter(t)
        phi = self.basis(x)

        B, R = cfg.B, cfg.R

        grad_V_tilde = - B.dot(self.BTBinv).dot(R).dot(W.T).dot(phi)
        u_tilde = W.T.dot(phi)

        return np.sum([
            e.T.dot(self.P).dot(cfg.B).dot(W.T.dot(phi) - Wcirc.T.dot(phi)),
            grad_V_tilde.T.dot(cfg.Am).dot(e),
            u_tilde.T.dot(R).dot(Wcirc.T).dot(phi),
            -0.5 * u_tilde.T.dot(R).dot(u_tilde)
        ])


def run_mrac():
    env = MRACEnv()
    env.reset()

    while True:
        env.render()

        done = env.step()

        if done:
            break

    env.close()


def run_hmrac():
    env = HMRACEnv()
    env.reset()

    while True:
        env.render()

        done = env.step()

        if done:
            break

    env.close()


def exp1():
    basedir = Path("data/exp1")

    load_config()
    cfg.dir = Path(basedir, "data00")
    cfg.label = "MRAC"
    run_mrac()

    load_config()
    cfg.dir = Path(basedir, "data01")
    cfg.label = "HMRAC"
    run_hmrac()


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
    cmdstyle = dict(basestyle, ls="--", label="Command")
    mrac.style.update(basestyle, c="g", ls="-")

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

    # ===============================================
    # Tracking and parameter estimation errors (norm)
    # ===============================================
    figsize, pos = plot.posing(2, subsize, width, top, bottom, left, hspace)
    plt.figure(figsize=figsize)

    ax = plt.subplot(211, position=pos[0])
    [plot.tracking_error(d) for d in data]
    plt.ylabel(r"$||e||$")
    plt.ylim(0, 0.6)
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


def main():
    exp1()
    exp1_plot()


if __name__ == "__main__":
    main()
