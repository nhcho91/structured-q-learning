import numpy as np
import scipy
from types import SimpleNamespace as SN
from pathlib import Path

import fym
from fym.core import BaseEnv, BaseSystem


cfg = SN()


def load_config():
    cfg.dir = "data"
    cfg.final_time = 50

    cfg.Am = np.array([[0, 1, 0], [-15.8, -5.6, -17.3], [1, 0, 0]])
    cfg.B = np.array([[0, 1, 0]]).T
    cfg.Br = np.array([[0, 0, -1]]).T
    cfg.x_init = np.vstack((0.3, 0, 0))
    cfg.Q_lyap = np.eye(3)
    cfg.R = np.diag([0.1])
    cfg.Rinv = np.linalg.inv(cfg.R)

    cfg.Wcirc = np.vstack((
        -18.59521, 15.162375, 0,
        np.zeros((3, 1)), np.zeros((1, 1)),
        -62.45153, 9.54708, 21.45291,
    ))

    cfg.vareps = SN()
    cfg.vareps.freq = 5
    cfg.vareps.amp = 0  # 2
    cfg.vareps.offset = 0

    cfg.Gamma1 = 1e4

    # MRAC
    cfg.MRAC = SN()
    cfg.MRAC.env_kwargs = dict(
        solver="odeint", dt=20, max_t=cfg.final_time, ode_step_len=int(20/0.01))

    # H-Modification MRAC
    cfg.HMRAC = SN()
    cfg.HMRAC.env_kwargs = dict(
        solver="odeint", dt=20, max_t=cfg.final_time, ode_step_len=int(20/0.01))


def sharp_jump(t, t0, dy=1):
    jump = dy * ((t - t0) > 0)
    return jump


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

    def set_dot(self, c):
        xr = self.state
        self.dot = cfg.Am.dot(xr) + cfg.Br.dot(c)


class PerformanceIndex(BaseSystem):
    def set_dot(self, x, u):
        self.dot = 1/2 * (x.T.dot(cfg.Q_lyap).dot(x) + u.T.dot(cfg.R).dot(u))


class Uncertainty():
    def __init__(self):
        self.Wcirc = cfg.Wcirc

    def basis(self, x, xr, c):
        xa = np.vstack((x, xr, c))
        return np.vstack((xa, np.abs(x[:2])*x[1], x[0]**3))

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


# class Command(BaseSystem):
#     def __init__(self):
#         super().__init__(np.vstack((1, 0)))

#         self.A = np.array([
#             [0, 0.1],
#             [-1, 0],
#         ])
#         self.C = np.array([[1, 0]])

#     def set_dot(self, x):
#         self.dot = self.A.dot(x)

#     def get(self, x):
#         return self.C.dot(x)


class CMRAC(BaseSystem):
    def __init__(self, P, B):
        super().__init__(shape=cfg.Wcirc.shape)
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

        self.P = scipy.linalg.solve_continuous_are(
            cfg.Am, cfg.B, cfg.Q_lyap, cfg.R)

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
