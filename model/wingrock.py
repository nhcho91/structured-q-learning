import numpy as np
import scipy
from types import SimpleNamespace as SN
from pathlib import Path
from collections import deque
import random

import fym
from fym.core import BaseEnv, BaseSystem


cfg = SN()


def load_config():
    cfg.dir = "data"
    cfg.final_time = 30

    cfg.Am = np.array([[0, 1, 0], [-15.8, -5.6, -17.3], [1, 0, 0]])
    cfg.B = np.array([[0, 1, 0]]).T
    cfg.Br = np.array([[0, 0, -1]]).T
    cfg.x_init = np.vstack((0.3, 0, 0))
    cfg.Q_lyap = np.eye(3)
    cfg.R = np.diag([0.1])
    cfg.Rinv = np.linalg.inv(cfg.R)

    cfg.P = scipy.linalg.solve_lyapunov(cfg.Am.T, -cfg.Q_lyap)

    cfg.Wcirc = np.vstack((
        -18.59521, 15.162375, -62.45153, 9.54708, 21.45291,
    ))
    cfg.Winit = np.zeros_like(cfg.Wcirc)

    cfg.vareps = SN()
    cfg.vareps.freq = 5
    cfg.vareps.amp = 0  # 2
    cfg.vareps.offset = 0

    cfg.Gamma1 = 1e4

    # MRAC
    cfg.MRAC = SN()
    cfg.MRAC.env_kwargs = dict(
        max_t=cfg.final_time,
        solver="odeint", dt=20, ode_step_len=int(20/0.01),
    )

    # H-Modification MRAC
    cfg.HMRAC = SN()
    cfg.HMRAC.env_kwargs = dict(
        max_t=cfg.final_time,
        solver="odeint", dt=20, ode_step_len=int(20/0.01),
    )

    # Value Learner
    cfg.ValueLearner = SN()
    cfg.ValueLearner.env_kwargs = dict(
        max_t=cfg.final_time,
        solver="rk4", dt=1e-3)
    cfg.ValueLearner.switching_time = 30
    cfg.ValueLearner.Gamma = 1e2
    cfg.ValueLearner.kL = 0.1
    cfg.ValueLearner.kU = 10
    cfg.ValueLearner.theta = 0.1
    cfg.ValueLearner.tauf = 1e-3
    cfg.ValueLearner.threshold = 1e-10

    # Double MRAC
    cfg.DoubleMRAC = SN()
    cfg.DoubleMRAC.env_kwargs = dict(
        max_t=cfg.final_time,
        solver="odeint", dt=20, ode_step_len=int(20/0.01),
    )
    cfg.DoubleMRAC.GammaF = 1e-4


def sharp_jump(t, t0, dy=1):
    jump = dy * ((t - t0) > 0)
    return jump


class System(BaseSystem):
    def __init__(self):
        super().__init__(cfg.x_init)
        self.unc = Uncertainty()

    def deriv(self, t, x, u, c):
        return (
            cfg.Am.dot(x)
            + cfg.B.dot(u + self.unc(t, x))
            + cfg.Br.dot(c)
        )

    def set_dot(self, t, u, c):
        x = self.state
        self.dot = self.deriv(t, x, u, c)


class ReferenceSystem(BaseSystem):
    def __init__(self):
        super().__init__(cfg.x_init)

    def deriv(self, xr, c):
        return cfg.Am.dot(xr) + cfg.Br.dot(c)

    def set_dot(self, c):
        xr = self.state
        self.dot = self.deriv(xr, c)


class PerformanceIndex(BaseSystem):
    def set_dot(self, x, u):
        self.dot = 1/2 * (x.T.dot(cfg.Q_lyap).dot(x) + u.T.dot(cfg.R).dot(u))


class Uncertainty():
    def __init__(self):
        self.Wcirc = cfg.Wcirc

    def basis(self, x, xr, c):
        return np.vstack((x[:2], np.abs(x[:2])*x[1], x[0]**3))

    def parameter(self, t):
        return self.Wcirc

    def vareps(self, t, x):
        vareps = np.tanh(x[1])
        vareps += np.exp(-t/10) * np.sin(5 * t)
        return cfg.vareps.amp * vareps

    def __call__(self, t, x):
        Wcirc = self.parameter(t)
        return (
            Wcirc.T.dot(self.basis(x, np.zeros((3, 1)), np.zeros((1, 1))))
            + self.vareps(t, x))


class Command():
    def get(self, t):
        if t > 10:
            c = scipy.signal.square([(t - 10) * 2*np.pi / 20])
        else:
            c = 0
        return np.atleast_2d(c)


class CMRAC(BaseSystem):
    def __init__(self, P, B):
        super().__init__(cfg.Winit)
        self.P = P
        self.B = B

    def set_dot(self, e, phi, composite_term):
        self.dot = (
            cfg.Gamma1 * phi.dot(e.T).dot(self.P).dot(self.B)
            + composite_term
        )


class MRACEnv(BaseEnv):
    def __init__(self):
        super().__init__(**cfg.MRAC.env_kwargs)
        self.x = System()
        self.xr = ReferenceSystem()

        self.P = cfg.P
        self.W = CMRAC(self.P, cfg.B)
        self.J = PerformanceIndex()
        self.cmd = Command()

        self.basis = self.x.unc.basis

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

        c = self.cmd.get(t)
        phi = self.basis(x, xr, c)

        e = x - xr
        u = self.get_input(e, W, phi)

        self.x.set_dot(t, u, c)
        self.xr.set_dot(c)
        self.W.set_dot(e, phi, 0)
        self.J.set_dot(e, u)

    def get_input(self, e, W, phi):
        un = -cfg.Rinv.dot(cfg.B.T).dot(self.P).dot(e)
        ua = -W.T.dot(phi)
        return un + ua

    def get_HJB_error(self, t, x, xr, W, c):
        e = x - xr
        Wcirc = self.x.unc.parameter(t)
        phi = self.basis(x, xr, c)

        B, R = cfg.B, cfg.R

        grad_V_tilde = - B.dot(self.BTBinv).dot(R).dot(W.T).dot(phi)
        u_tilde = W.T.dot(phi)

        return np.sum([
            e.T.dot(self.P).dot(cfg.B).dot(W.T.dot(phi) - Wcirc.T.dot(phi)),
            grad_V_tilde.T.dot(cfg.Am).dot(e),
            u_tilde.T.dot(R).dot(Wcirc.T).dot(phi),
            -0.5 * u_tilde.T.dot(R).dot(u_tilde)
        ])

    def logger_callback(self, i, t, y, t_hist, ode_hist):
        x, xr, W, J = self.observe_list(y)
        c = self.cmd.get(t)
        phi = self.basis(x, xr, c)

        Wcirc = self.x.unc.parameter(t)

        e = x - xr
        u = self.get_input(e, W, phi)

        e_HJB = self.get_HJB_error(t, x, xr, W, c)

        return dict(t=t, x=x, xr=xr, W=W, J=J,
                    e=e, c=c, Wcirc=Wcirc, u=u, e_HJB=e_HJB)


class ValueLearnerAgent():
    def __init__(self):
        n, m = cfg.Winit.shape
        self.F = np.zeros((n, n))
        self.G = np.zeros((n, m))
        self.What = np.zeros_like(cfg.Wcirc)

        self.best_F = self.F.copy()
        self.best_G = self.G.copy()

        self.logger = fym.logging.Logger(
            Path(cfg.dir, "value-learner-agent.h5"))
        self.logger.set_info(cfg=cfg)

    def get_action(self, obs):
        F, _ = self.best_F, self.best_G

        t, *_ = obs
        eigs, _ = self.get_eig(F, cfg.ValueLearner.threshold)
        self.logger.record(
            t=t,
            eigs=cfg.ValueLearner.Gamma * eigs,
            What=self.What
        )

        return self.What

    def update(self, obs):
        t, y, xi, phi = obs

        if t >= cfg.ValueLearner.switching_time:
            return

        What = self.What

        p, q = self._get_pq(obs)

        self.F = q * self.F + p * xi.dot(xi.T)
        self.G = q * self.G + p * xi.dot(y.T)

        if self.get_eig(self.F)[0][0] > self.get_eig(self.best_F)[0][0]:
            self.best_F = self.F
            self.best_G = self.G

        FWtilde = self.best_F.dot(What) - self.best_G
        dt = cfg.ValueLearner.env_kwargs["dt"]
        self.What = What - cfg.ValueLearner.Gamma * dt * FWtilde

    def close(self):
        self.logger.close()

    def _get_pq(self, obs):
        t, y, xi, phi = obs

        p = cfg.ValueLearner.env_kwargs["dt"]
        xidot = - (xi - phi) / cfg.ValueLearner.tauf
        nxidot = np.linalg.norm(xidot)
        k = self._get_k(nxidot)
        q = 1 - k * p

        return p, q

    def _get_k(self, nxidot):
        return (cfg.ValueLearner.kL
                + ((cfg.ValueLearner.kU - cfg.ValueLearner.kL)
                   * np.tanh(cfg.ValueLearner.theta * nxidot)))

    def get_eig(self, A, threshold=1e-10):
        eigs, eigv = np.linalg.eig(A)
        sort = np.argsort(np.real(eigs))  # sort in ascending order
        eigs = np.real(eigs[sort])
        eigv = np.real(eigv[:, sort])
        eigs[eigs < threshold] = 0
        return eigs, eigv


class ResidualFilter(BaseEnv):
    def __init__(self, basis, B):
        super().__init__()
        n, m = cfg.Winit.shape
        self.xi = BaseSystem(shape=(n, 1))
        self.z = BaseSystem(shape=(m, 1))

        self.Bdagger = np.linalg.pinv(B)
        self.tauf = cfg.ValueLearner.tauf

    def set_dot(self, e, phi, u):
        xi = self.xi.state
        z = self.z.state

        self.xi.dot = -1 / self.tauf * (xi - phi)
        self.z.dot = (
            1 / self.tauf * (self.Bdagger.dot(e) - z)
            + self.Bdagger.dot(cfg.Am).dot(e)
            + u
        )

    def get_y(self, e, z):
        return 1 / self.tauf * (self.Bdagger.dot(e) - z)


class ValueLearnerEnv(MRACEnv, BaseEnv):
    def __init__(self):
        BaseEnv.__init__(self, **cfg.ValueLearner.env_kwargs)
        self.x = System()
        self.xr = ReferenceSystem()

        self.P = cfg.P
        self.W = CMRAC(self.P, cfg.B)

        self.F = BaseSystem(np.zeros_like(cfg.Wcirc.dot(cfg.Wcirc.T)))
        self.J = PerformanceIndex()

        self.basis = self.x.unc.basis
        self.filter = ResidualFilter(basis=self.basis, B=cfg.B)

        self.cmd = Command()

        self.BTBinv = np.linalg.inv(cfg.B.T.dot(cfg.B))

        self.logger = fym.logging.Logger(
            Path(cfg.dir, "value-learner-env.h5"))
        self.logger.set_info(cfg=cfg)

    def step(self, action):
        J = self.J.state
        *_, done = self.update(What=action)
        next_obs = self.observation()
        reward = self.J.state - J
        return next_obs, reward, done

    def reset(self):
        super().reset()
        return self.observation()

    def observation(self):
        t = self.clock.get()
        x, xr, W, F, J, (xi, z) = self.observe_list()
        e = x - xr
        y = self.filter.get_y(e, z)
        c = self.cmd.get(t)
        phi = self.basis(x, xr, c)
        return t, y, xi, phi

    def set_dot(self, t, What):
        x = self.x.state
        xr = self.xr.state
        W = self.W.state
        F = self.F.state

        c = self.cmd.get(t)
        phi = self.basis(x, xr, c)

        e = x - xr
        u = self.get_input(e, W, phi)

        if t > cfg.ValueLearner.switching_time:
            Wtilde = W - What
            composite_term = - cfg.Gamma1 * F.dot(Wtilde)
            Fdot = cfg.DoubleMRAC.GammaF * Wtilde.dot(Wtilde.T)
        else:
            composite_term = 0
            Fdot = np.zeros_like(F)

        self.x.set_dot(t, u, c)
        self.xr.set_dot(c)
        self.W.set_dot(e, phi, composite_term)
        self.F.dot = Fdot
        self.J.set_dot(e, u)
        self.filter.set_dot(e, phi, u)

    def logger_callback(self, i, t, y, t_hist, ode_hist):
        x, xr, W, F, J, (xi, z) = self.observe_list(y)
        c = self.cmd.get(t)
        phi = self.basis(x, xr, c)

        Wcirc = self.x.unc.parameter(t)

        e = x - xr
        u = self.get_input(e, W, phi)

        e_HJB = self.get_HJB_error(t, x, xr, W, c)

        return dict(t=t, x=x, xr=xr, W=W, F=F, J=J, xi=xi, z=z,
                    e=e, c=c, Wcirc=Wcirc, u=u, e_HJB=e_HJB)


class DoubleMRACEnv(MRACEnv, BaseEnv):
    def __init__(self):
        BaseEnv.__init__(self, **cfg.DoubleMRAC.env_kwargs)
        self.x = System()
        self.xr = ReferenceSystem()
        self.P = cfg.P
        self.W = CMRAC(self.P, cfg.B)
        self.F = BaseSystem(np.zeros_like(cfg.Wcirc.dot(cfg.Wcirc.T)))
        self.J = PerformanceIndex()
        self.cmd = Command()

        self.basis = self.x.unc.basis

        self.BTBinv = np.linalg.inv(cfg.B.T.dot(cfg.B))

        self.logger = fym.logging.Logger(Path(cfg.dir, "double-mrac-env.h5"))
        self.logger.set_info(cfg=cfg)

    def set_dot(self, t):
        x = self.x.state
        xr = self.xr.state
        W = self.W.state
        F = self.F.state

        c = self.cmd.get(t)
        phi = self.basis(x, xr, c)

        e = x - xr
        u = self.get_input(e, W, phi)

        Wtilde = W - cfg.Wcirc
        composite_term = - cfg.Gamma1 * F.dot(Wtilde)

        self.x.set_dot(t, u, c)
        self.xr.set_dot(c)
        self.W.set_dot(e, phi, composite_term)
        self.J.set_dot(e, u)
        self.F.dot = cfg.DoubleMRAC.GammaF * Wtilde.dot(Wtilde.T)

    def logger_callback(self, i, t, y, t_hist, ode_hist):
        x, xr, W, F, J = self.observe_list(y)
        c = self.cmd.get(t)
        phi = self.basis(x, xr, c)

        Wcirc = self.x.unc.parameter(t)

        e = x - xr
        u = self.get_input(e, W, phi)

        e_HJB = self.get_HJB_error(t, x, xr, W, c)

        return dict(t=t, x=x, xr=xr, W=W, F=F, J=J,
                    e=e, c=c, Wcirc=Wcirc, u=u, e_HJB=e_HJB)


class HMRACEnv(MRACEnv, BaseEnv):
    def __init__(self):
        BaseEnv.__init__(self, **cfg.HMRAC.env_kwargs)
        self.x = System()
        self.xr = ReferenceSystem()

        self.P = scipy.linalg.solve_continuous_are(
            cfg.Am, cfg.B, cfg.Q_lyap, cfg.R)

        self.W = CMRAC(self.P, cfg.B)
        self.J = PerformanceIndex()
        self.cmd = Command()

        self.basis = self.x.unc.basis

        self.BTBinv = np.linalg.inv(cfg.B.T.dot(cfg.B))

        self.logger = fym.logging.Logger(Path(cfg.dir, "hmrac-env.h5"))
        self.logger.set_info(cfg=cfg)

    def set_dot(self, t):
        x = self.x.state
        xr = self.xr.state
        W = self.W.state

        c = self.cmd.get(t)
        phi = self.basis(x, xr, c)

        e = x - xr
        u = self.get_input(e, W, phi)

        B = cfg.B
        composite_term = cfg.Gamma1 * (
            phi.dot(
                - e.T.dot(cfg.Am.T).dot(B).dot(self.BTBinv)
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
