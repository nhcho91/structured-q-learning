import numpy as np
import scipy
from types import SimpleNamespace as SN
from pathlib import Path
from collections import deque
import random
import logging


import fym
from fym.agents import LQR
from fym.core import BaseEnv, BaseSystem


cfg = SN()


def load_config():
    cfg.dir = "data"
    cfg.final_time = 150

    cfg.A = np.array([[0, 1, 0], [0, 0, 0], [1, 0, 0]])
    cfg.B = np.array([[0, 1, 0]]).T
    cfg.Q = np.diag([1, 10, 10])
    cfg.R = np.diag([10])

    cfg.K, cfg.P, *_ = LQR.clqr(cfg.A, cfg.B, cfg.Q, cfg.R)

    cfg.x_init = np.vstack((0.3, 0, 0))

    cfg.QLearner = SN()
    cfg.QLearner.env_kwargs = dict(
        max_t=cfg.final_time,
        # solver="odeint", dt=20, ode_step_len=int(20/0.01),
        solver="rk4", dt=0.001,
    )
    cfg.QLearner.Khat_init = np.zeros_like(cfg.K)
    cfg.QLearner.Phat_init = np.zeros_like(cfg.P)
    cfg.QLearner.Khat_gamma = 1e-2
    cfg.QLearner.Phat_gamma = 1e-2
    cfg.QLearner.memory_len = 10000
    cfg.QLearner.batch_size = 128
    cfg.QLearner.train_epoch = 100


class LinearSystem(BaseSystem):
    def __init__(self):
        super().__init__(cfg.x_init)

    def deriv(self, t, x, u):
        return cfg.A.dot(x) + cfg.B.dot(u)

    def set_dot(self, t, u):
        x = self.state
        self.dot = self.deriv(t, x, u)


class QLearnerAgent():
    def __init__(self):
        self.memory = deque(maxlen=cfg.QLearner.memory_len)
        self.Phat = cfg.QLearner.Phat_init
        self.Khat = cfg.QLearner.Khat_init

        self.gK = cfg.QLearner.Khat_gamma
        self.gP = cfg.QLearner.Phat_gamma
        self.N = cfg.QLearner.batch_size

        self.logger = fym.logging.Logger(
            Path(cfg.dir, "qlearner-agent.h5"))
        self.logger.set_info(cfg=cfg)

    def get_action(self, obs):
        return None

    def update(self, obs, action, next_obs, reward, done):
        t, x, u, xdot = obs
        self.memory.append((x, u, xdot))

        if len(self.memory) >= self.memory.maxlen:
            self.train(t)

    def train(self, t):
        for i in range(cfg.QLearner.train_epoch):
            Phat = self.Phat
            Khat = self.Khat

            gradP, gradK = 0, 0
            loss = 0
            batch = random.sample(self.memory, self.N)
            for b in batch:
                x, u, xdot = b

                e = np.sum([
                    x.T.dot(Phat).dot(xdot),
                    -u.T.dot(cfg.R).dot(Khat).dot(x),
                    -0.5 * x.T.dot(Khat.T).dot(cfg.R).dot(Khat).dot(x),
                    0.5 * x.T.dot(cfg.Q).dot(x)
                ])
                gradP_e = 0.5 * x.dot(xdot.T) + 0.5 * xdot.dot(x.T)
                gradK_e = cfg.R.dot(
                    u.dot(xdot.T)
                    - 0.5 * (u + Khat.dot(x + 2*xdot)).dot(x.T)
                )

                gradP = gradP + e * gradP_e
                gradK = gradK + e * gradK_e

                loss += 0.5 * e**2

            logging.debug(
                f"Time {t:5.2f}/{cfg.final_time:5.2f} | "
                f"Epoch {i+1:03d}/{cfg.QLearner.train_epoch:03d} | "
                f"Loss: {loss:07.4f}")

            factor = 1 - i / cfg.QLearner.train_epoch
            self.Phat = Phat - factor * self.gP * e * gradP / self.N
            self.Khat = Khat - factor * self.gK * e * gradK / self.N

    def close(self):
        self.logger.close()


class QLearnerEnv(BaseEnv):
    def __init__(self):
        super().__init__(**cfg.QLearner.env_kwargs)
        self.x = LinearSystem()
        self.Khat = BaseSystem(cfg.QLearner.Khat_init)
        self.Phat = BaseSystem(cfg.QLearner.Phat_init)
        self.gK = cfg.QLearner.Khat_gamma
        self.gP = cfg.QLearner.Phat_gamma

        self.logger = fym.logging.Logger(Path(cfg.dir, "qlearner-env.h5"))
        self.logger.set_info(cfg=cfg)

    def step(self, action):
        *_, done = self.update(action=action)
        next_obs = self.observation()
        return next_obs, None, done

    def reset(self):
        super().reset()
        return self.observation()

    def observation(self):
        t = self.clock.get()
        x, Khat, Phat = self.observe_list()

        u = self.get_input(t, x)
        xdot = self.x.deriv(t, x, u)
        return t, x, u, xdot

    def set_dot(self, t, action):
        x, Khat, Phat = self.observe_list()

        u = self.get_input(t, x)
        xdot = self.x.deriv(t, x, u)
        udot = -0.5 * u - 0.5 * cfg.K.dot(x + 2 * xdot)

        gradxV = Phat.dot(x) + Khat.T.dot(cfg.R).dot(u + Khat.dot(x))
        graduV = cfg.R.dot(u + Khat.dot(x))

        e = np.sum([
            gradxV.T.dot(xdot),
            graduV.T.dot(udot),
            0.5 * x.T.dot(cfg.Q).dot(x),
            0.5 * u.T.dot(cfg.R).dot(u)
        ])

        if e > 0:
            e = 0

        self.x.set_dot(t, u)
        self.Khat.dot = -self.gK * e * cfg.R.dot(
            Khat.dot(x).dot(xdot.T)
            + Khat.dot(xdot).dot(x.T)
            + u.dot(xdot.T)
            + udot.dot(x.T)
        )
        self.Phat.dot = -self.gP * e * 0.5 * (
            x.dot(xdot.T)
            + xdot.dot(x.T)
        )

    def get_input(self, t, x):
        un = - cfg.K.dot(x)
        noise = np.sum([
            0.3 * np.sin(t) * np.cos(np.pi * t),
            0.5 * np.sin(0.2 * t + 1)
        ])

        return un + noise * np.exp(-0.8 * t / cfg.final_time)

    def logger_callback(self, i, t, y, t_hist, ode_hist):
        states = self.observe_dict(y)
        x, Khat, Phat = self.observe_list(y)
        u = self.get_input(t, x)

        return dict(t=t, u=u, K=cfg.K, P=cfg.P, **states)
