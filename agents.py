import numpy as np
from types import SimpleNamespace as SN
from pathlib import Path
from collections import deque
import logging
import random

import fym
from fym.agents import LQR

logger = logging.getLogger("logs")

cfg = SN()


def load_config():
    cfg.SQLAgent = SN()
    cfg.SQLAgent.memory_len = 2000
    cfg.SQLAgent.batch_size = 400
    cfg.SQLAgent.train_epoch = 100
    cfg.SQLAgent.train_period = 3

    cfg.KLMAgent = SN()
    cfg.KLMAgent.memory_len = 2000
    cfg.KLMAgent.batch_size = 400
    cfg.KLMAgent.train_epoch = 100
    cfg.KLMAgent.train_period = 3


class SQLAgent():
    def __init__(self, Q, R, F,
                 W1_init=None, W2_init=None, W3_init=None, K_init=None):
        self.Q, self.R, self.F = Q, R, F
        self.memory = deque(maxlen=cfg.SQLAgent.memory_len)

        n, m = Q.shape[0], R.shape[0]
        self.W1 = W1_init or np.zeros((n, n))
        self.W2 = W2_init or np.zeros((m, n))
        self.W3 = W3_init or np.zeros((m, m))
        self.K = K_init or np.zeros((m, n))

        self.M = cfg.SQLAgent.batch_size
        self.N = cfg.SQLAgent.train_epoch

        self.n, self.m = n, m

        self.W1_his = np.zeros((self.N, self.n, self.n))
        self.W2_his = np.zeros((self.N, self.m, self.n))
        self.W3_his = np.zeros((self.N, self.m, self.m))

        self.last_t = 0

    def logger_callback(self):
        P = self.W1 - self.K.T.dot(self.W3).dot(self.K)
        return dict(
            W1=self.W1, W2=self.W2, W3=self.W3, K=self.K,
            W1_his=self.W1_his, W2_his=self.W2_his, W3_his=self.W3_his,
            P=P,
        )

    def get_action(self, obs):
        return None

    def update(self, obs, action, next_obs, reward, done):
        t, x, u, xdot = obs
        self.memory.append((x, u, xdot))

    def is_train(self, t):
        return len(self.memory) >= self.memory.maxlen and self.is_record(t)

    def is_record(self, t):
        return t - self.last_t > cfg.SQLAgent.train_period

    def train(self, t):
        for i in range(self.N):
            y_stack = []
            phi_stack = []
            loss = 0
            batch = random.sample(self.memory, self.M)

            for b in batch:
                x, u, xdot = b

                lxu = 0.5 * (x.T.dot(self.Q).dot(x) + u.T.dot(self.R).dot(u))

                udot = self.F.dot(u + self.K.dot(x)) - self.K.dot(xdot)
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
            W = np.diag(-1/Y.ravel())
            w = np.linalg.pinv(Phi.T.dot(W).dot(Phi)).dot(Phi.T).dot(W).dot(Y)
            # w2 = np.linalg.pinv(Phi).dot(Y)
            w1, w2, w3 = w[:w1s], w[w1s:w1s + w2s], w[w1s + w2s:]

            self.W1 = w1.reshape(self.n, self.n, order="F")
            self.W2 = w2.reshape(self.m, self.n, order="F")
            self.W3 = w3.reshape(self.m, self.m, order="F")

            self.W1_his[i] = self.W1
            self.W2_his[i] = self.W2
            self.W3_his[i] = self.W3

            next_K = np.linalg.inv(self.W3).dot(self.W2)

            loss = ((Y - Phi.dot(w))**2).sum()
            error = ((next_K - self.K)**2).sum()

            logger.debug(
                f"Time: {t:5.2f} sec | "
                f"Epoch: {i+1:03d}/{self.N:03d} | "
                f"Loss: {loss:07.4f} | "
                f"Error: {error:07.4f}"
            )

            # Policy Improvement
            self.K = next_K

    def close(self):
        self.logger.close()


class KLMAgent():
    def __init__(self, Q, R,
                 W1_init=None, W2_init=None, K_init=None):
        self.Q, self.R = Q, R
        self.memory = deque(maxlen=cfg.KLMAgent.memory_len)

        n, m = Q.shape[0], R.shape[0]
        self.W1 = W1_init or np.zeros((n, n))
        self.W2 = W2_init or np.zeros((m, n))
        self.K = K_init or np.zeros((m, n))

        self.M = cfg.KLMAgent.batch_size
        self.N = cfg.KLMAgent.train_epoch

        self.n, self.m = n, m

        self.W1_his = np.zeros((self.N, self.n, self.n))
        self.W2_his = np.zeros((self.N, self.m, self.n))

        self.last_t = 0

    def logger_callback(self):
        P = self.W1 - self.K.T.dot(self.R).dot(self.K)
        return dict(
            W1=self.W1, W2=self.W2, K=self.K,
            W1_his=self.W1_his, W2_his=self.W2_his,
            P=P,
        )

    def get_action(self, obs):
        return None

    def update(self, obs, action, next_obs, reward, done):
        t, x, u, xdot = obs
        self.memory.append((x, u, xdot))

    def is_train(self, t):
        return len(self.memory) >= self.memory.maxlen and self.is_record(t)

    def is_record(self, t):
        return t - self.last_t > cfg.KLMAgent.train_period

    def train(self, t):
        for i in range(self.N):
            y_stack = []
            phi_stack = []
            loss = 0
            batch = random.sample(self.memory, self.M)

            for b in batch:
                x, u, xdot = b

                lxu = 0.5 * (
                    x.T.dot(self.Q + self.K.T.dot(self.R).dot(self.K)).dot(x))

                # phi1 = np.kron(xdot, x)
                phi1 = 0.5 * np.kron(x, xdot) + 0.5 * np.kron(xdot, x)
                phi2 = - np.kron(x, self.R.dot(u + self.K.dot(x)))
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

            self.W1 = w1.reshape(self.n, self.n, order="F")
            self.W2 = w2.reshape(self.m, self.n, order="F")

            self.W1_his[i] = self.W1
            self.W2_his[i] = self.W2

            next_K = self.W2

            loss = ((Y - Phi.dot(w))**2).sum()
            error = ((next_K - self.K)**2).sum()

            logger.debug(
                f"Time: {t:5.2f} sec | "
                f"Epoch: {i+1:03d}/{self.N:03d} | "
                f"Loss: {loss:07.4f} | "
                f"Error: {error:07.4f}"
            )

            # Policy Improvement
            self.K = next_K

    def close(self):
        self.logger.close()
