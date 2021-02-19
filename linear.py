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
    cfg.final_time = 40

    # cfg.A = np.array([[0, 1, 0], [0, 0, 0], [1, 0, 0]])
    cfg.A = np.array([[-2, 1, 0], [0, -2, -1], [0.5, 0, -1]])
    # cfg.A = np.array([[2, 1, 0], [0, 1, 0], [1, 0, 3]])
    cfg.B = np.array([
        [0, 0],
        [1, 0],
        [0, 1],
    ])
    Fp = np.array([[-1, 1], [0, 1]])
    Kf, *_ = LQR.clqr(Fp, np.eye((2)), np.eye(2), np.eye(2))
    cfg.F = Fp - Kf
    cfg.Q = np.array([
        [1, 0, 0],
        [0, 10, 0],
        [0, 0, 10]
    ])
    cfg.R = np.array([
        [1, 0],
        [0, 10]
    ])

    cfg.x_init = np.vstack((0.3, 0, 0))

    cfg.QLearner = SN()
    cfg.QLearner.env_kwargs = dict(
        max_t=cfg.final_time,
        # solver="odeint", dt=20, ode_step_len=int(20/0.01),
        solver="rk4", dt=0.001,
    )
    cfg.QLearner.memory_len = 10000
    cfg.QLearner.batch_size = 100
    cfg.QLearner.train_epoch = 10

    calc_config()


def calc_config():
    cfg.K, cfg.P, *_ = LQR.clqr(cfg.A, cfg.B, cfg.Q, cfg.R)

    cfg.QLearner.K_init = np.zeros_like(cfg.K)
    cfg.QLearner.W1_init = np.zeros_like(cfg.P)
    cfg.QLearner.W2_init = np.zeros_like(cfg.K)
    cfg.QLearner.W3_init = np.zeros_like(cfg.R)

    K = cfg.QLearner.K_init
    print(np.linalg.eigvals(cfg.A - K.T.dot(cfg.R).dot(K)))


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
        self.W1 = cfg.QLearner.W1_init
        self.W2 = cfg.QLearner.W2_init
        self.W3 = cfg.QLearner.W3_init
        self.K = cfg.QLearner.K_init

        self.M = cfg.QLearner.batch_size
        self.N = cfg.QLearner.train_epoch

        self.n, self.m = cfg.B.shape

        self.train_time = 0

        self.logger = fym.logging.Logger(
            Path(cfg.dir, "qlearner-agent.h5"), max_len=1)
        self.logger.set_info(cfg=cfg)

        P = self.W1 - self.K.T.dot(cfg.R).dot(self.K)
        self.logger.record(
            t=0, W1=self.W1, W2=self.W2, W3=self.W3, K=self.K, P=P,
        )

    def get_action(self, obs):
        return None

    def update(self, obs, action, next_obs, reward, done):
        t, x, u, xdot = obs
        self.memory.append((x, u, xdot))

        if len(self.memory) >= self.memory.maxlen:
            if t - self.train_time > 3:
                self.train(t)

    def train(self, t):
        W1_his = np.zeros((self.N, self.n, self.n))
        W2_his = np.zeros((self.N, self.m, self.n))
        W3_his = np.zeros((self.N, self.m, self.m))

        # Initiallize
        K = self.K

        for i in range(self.N):
            y_stack = []
            phi_stack = []
            loss = 0
            batch = random.sample(self.memory, self.M)
            for b in batch:
                x, u, xdot = b

                lxu = 0.5 * (x.T.dot(cfg.Q).dot(x) + u.T.dot(cfg.R).dot(u))
                # e = (
                #     xdot.T.dot(W2.dot(cfg.R).dot(u) + W1.dot(x))
                #     - 0.5 * (u + W2.dot(x)).T.dot(u + K.dot(x + 2 * xdot))
                #     + lxu
                # )

                udot = cfg.F.dot(u + K.dot(x)) - K.dot(xdot)
                phi1 = 0.5 * np.kron(x, xdot) + 0.5 * np.kron(xdot, x)
                phi2 = np.kron(xdot, u) + np.kron(x, udot)
                phi3 = 0.5 * (np.kron(u, udot) + np.kron(udot, u))
                phi = np.vstack((phi1, phi2, phi3))

                y = - lxu

                y_stack.append(y)
                phi_stack.append(phi.T)

            Y = np.vstack(y_stack)
            Phi = np.vstack(phi_stack)

            w1s = self.n * self.n
            w2s = self.n * self.m
            w = np.linalg.pinv(Phi).dot(Y)
            w1, w2, w3 = w[:w1s], w[w1s:w1s + w2s], w[w1s + w2s:]

            W1 = w1.reshape(self.n, self.n, order="F")
            W2 = w2.reshape(self.m, self.n, order="F")
            W3 = w3.reshape(self.m, self.m, order="F")

            W1_his[i] = W1
            W2_his[i] = W2
            W3_his[i] = W3

            next_K = np.linalg.inv(W3).dot(W2)

            loss = ((Y - Phi.dot(w))**2).sum()
            error = ((next_K - K)**2).sum()

            logging.debug(
                f"Time: {t:5.2f}/{cfg.final_time:5.2f} | "
                f"Epoch: {i+1:03d}/{self.N:03d} | "
                f"Loss: {loss:07.4f} | "
                f"Error: {error:07.4f}"
            )

            # Policy Improvement
            K = next_K

        P = W1 - K.T.dot(W3).dot(K)
        P_loss = ((cfg.P - P)**2).sum()
        K_loss = ((cfg.K - K)**2).sum()
        R_loss = ((cfg.R + cfg.F.T.dot(W3) + W3.dot(cfg.F))**2).sum()

        logging.info(
            f"[Finished] Time: {t:5.2f} | "
            f"P Loss: {P_loss:07.4f} | "
            f"K Loss: {K_loss:07.4f} | "
            f"R Loss: {R_loss:07.4f} | "
        )

        self.W1 = W1
        self.W2 = W2
        self.W3 = W3
        self.K = K

        self.logger.record(
            t=t, W1=W1, W2=W2, W3=W3, K=K, P=P,
            W1_his=W1_his, W2_his=W2_his, W3_his=W3_his,
        )

        self.train_time = t

    def close(self):
        self.logger.close()


class ZLearnerAgent():
    def __init__(self):
        self.memory = deque(maxlen=cfg.QLearner.memory_len)
        self.W1 = cfg.QLearner.W1_init
        self.W2 = cfg.QLearner.W2_init
        self.K = cfg.QLearner.K_init

        self.M = cfg.QLearner.batch_size
        self.N = cfg.QLearner.train_epoch

        self.n, self.m = cfg.B.shape

        self.train_time = 0

        self.logger = fym.logging.Logger(
            Path(cfg.dir, "qlearner-agent.h5"), max_len=1)
        self.logger.set_info(cfg=cfg)

        P = self.W1 - self.K.T.dot(cfg.R).dot(self.K)
        self.logger.record(
            t=0, W1=self.W1, W2=self.W2, K=self.K, P=P,
        )

    def get_action(self, obs):
        return None

    def update(self, obs, action, next_obs, reward, done):
        t, x, u, xdot = obs
        self.memory.append((x, u, xdot))

        if len(self.memory) >= self.memory.maxlen:
            if t - self.train_time > 3:
                self.train(t)

    def train(self, t):
        W1_his = np.zeros((self.N, self.n, self.n))
        W2_his = np.zeros((self.N, self.m, self.n))

        # Initiallize
        K = self.K

        for i in range(self.N):
            y_stack = []
            phi_stack = []
            loss = 0
            batch = random.sample(self.memory, self.M)
            for b in batch:
                x, u, xdot = b

                lxu = 0.5 * (x.T.dot(cfg.Q + K.T.dot(cfg.R).dot(K)).dot(x))

                phi1 = 0.5 * np.kron(x, xdot) + 0.5 * np.kron(xdot, x)
                phi2 = - np.kron(cfg.R.dot(u + K.dot(x)), x)
                phi = np.vstack((phi1, phi2))

                y = - lxu

                y_stack.append(y)
                phi_stack.append(phi.T)

            Y = np.vstack(y_stack)
            Phi = np.vstack(phi_stack)

            w1s = self.n * self.n
            w2s = self.n * self.m
            w = np.linalg.pinv(Phi).dot(Y)
            w1, w2 = w[:w1s], w[w1s:w1s + w2s]

            W1 = w1.reshape(self.n, self.n, order="F")
            W2 = w2.reshape(self.m, self.n, order="F")

            W1_his[i] = W1
            W2_his[i] = W2

            next_K = W2

            loss = ((Y - Phi.dot(w))**2).sum()
            error = ((next_K - K)**2).sum()

            logging.info(
                f"Time: {t:5.2f}/{cfg.final_time:5.2f} | "
                f"Epoch: {i+1:03d}/{self.N:03d} | "
                f"Loss: {loss:07.4f} | "
                f"Error: {error:07.4f}"
            )

            # Policy Improvement
            K = next_K

        P = W1
        P_loss = ((cfg.P - P)**2).sum()
        K_loss = ((cfg.K - K)**2).sum()

        logging.info(
            f"[Finished] Time: {t:5.2f} | "
            f"P Loss: {P_loss:07.4f} | "
            f"K Loss: {K_loss:07.4f} | "
        )

        self.W1 = W1
        self.W2 = W2
        self.K = K

        self.logger.record(
            t=t, W1=W1, W2=W2, K=K, P=P,
            W1_his=W1_his, W2_his=W2_his,
        )

        self.train_time = t

    def close(self):
        self.logger.close()


class QLearnerEnv(BaseEnv):
    def __init__(self):
        super().__init__(**cfg.QLearner.env_kwargs)
        self.x = LinearSystem()

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
        x = self.x.state
        u = self.get_input(t, x)
        xdot = self.x.deriv(t, x, u)
        return t, x, u, xdot

    def set_dot(self, t, action):
        x = self.x.state
        u = self.get_input(t, x)
        self.x.set_dot(t, u)

    def get_input(self, t, x):
        un = - cfg.K.dot(x)
        noise = np.vstack([
            0.3 * np.sin(t) * np.cos(np.pi * t),
            0.5 * np.sin(0.2 * t + 1)
        ])

        return un + noise * np.exp(-0.8 * t / cfg.final_time)

    def logger_callback(self, i, t, y, t_hist, ode_hist):
        states = self.observe_dict(y)
        x = states["x"]
        u = self.get_input(t, x)

        return dict(t=t, u=u, K=cfg.K, P=cfg.P, **states)
