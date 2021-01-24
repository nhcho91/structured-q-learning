import numpy as np
import scipy
from scipy.signal import lti
import math
from types import SimpleNamespace as SN
from pathlib import Path

import fym
from fym.core import BaseEnv, BaseSystem
import fym.utils.rot


cfg = SN()


def load_config():
    # Simulation Setting
    cfg.dir = "data"
    cfg.final_time = 30
    cfg.multirotor_config = "quadrotor"

    # Reference model
    cfg.xm_init = np.zeros((3, 1))
    cfg.Am = np.array([
        [-7, 0, 0],
        [0, -7, 0],
        [0, 0, -4],
    ])
    cfg.Bm = np.array([
        [7, 0, 0],
        [0, 7, 0],
        [0, 0, 4],
    ])

    cfg.Q_lyap = np.eye(3)
    cfg.R = 0.1 * np.eye(4)
    cfg.Rinv = np.linalg.inv(cfg.R)

    cfg.W_init = np.zeros((12, 4))

    # MRAC
    cfg.MRAC = SN()
    cfg.MRAC.env_kwargs = dict(
        # solver="odeint", dt=5, ode_step_len=int(5/0.01),
        solver="rk4", dt=1e-3,
        max_t=cfg.final_time,
    )
    # cfg.MRAC.Gamma = 8e1
    cfg.MRAC.Gamma = 1e1

    # H-Modification MRAC
    cfg.HMRAC = SN()
    cfg.HMRAC.env_kwargs = dict(
        # solver="odeint", dt=5, ode_step_len=int(5/0.01),
        solver="rk4", dt=1e-3,
        max_t=cfg.final_time,
    )
    cfg.HMRAC.Gamma = 1e1

    # FECMRAC
    cfg.FECMRAC = SN()
    cfg.FECMRAC.Gamma = 1e0
    cfg.FECMRAC.kL = 0.1
    cfg.FECMRAC.kU = 10
    cfg.FECMRAC.theta = 0.1
    cfg.FECMRAC.tauf = 1e-3
    cfg.FECMRAC.threshold = 1e-10


def smooth_jump(t, t0, dt, dy, half=False):
    if not half:
        jump = dy * 0.5 * (np.tanh((12 * (t - t0) / dt - 6)) + 1)
    else:
        jump = dy * max(0, np.tanh(6 * (t - t0) / dt))
    return jump


def sharp_jump(t, t0, dy=1):
    jump = dy * ((t - t0) > 0)
    return jump


class MultiRotor(BaseEnv):
    """
    Reference:
        - Baldini et al., 2020
        - P. Pounds et al., 2010

    Description:
        - `+z` direction is upward.
        - pos and vel are resolved in the inertial frame,
        whereas Rot and omega are resolved in the body frame
    """
    g = 9.81  # Gravitaional acceleration [m/s^2]
    rho = 1.225  # Air density [kg/m^3]

    e3 = np.vstack((0, 0, 1))

    # Physical parameteres from Baldini et al., 2020
    m = 0.5  # Total system mass [kg]
    J = np.diag([5.9e-3, 5.9e-3, 1.16e-2])  # Inertia [kg*m^2]
    Jinv = np.linalg.inv(J)
    l = 0.225  # Distance between c.m. and motors [m]
    kr = 1e-3 * np.eye(3)  # Rotational friction coefficient (avg.) [N*s*m/rad]
    Jr = 6e-5  # Rotor inertia [N*m]
    R = 0.15  # Rotor radius [m]
    c = 0.04  # Propeller chord [m]
    b = 3.13e-5  # Thrust coefficient [N*s^2]
    d = 7.5e-7  # Drag coefficient [N*m*s^2]
    CdA = 0.08  # Flat plate area [m^2]
    a0 = 6  # Slope of the lift curve per radian [-]

    # Parameters from P. Pounds et al., 2010
    sigma = 0.054  # Solidity ratio [-]
    thetat = np.deg2rad(4.4)  # Blade tip angle [rad]
    CT = 0.0047  # Thrust coefficient [-]

    name = "multirotor"

    def __init__(self,
                 pos=np.zeros((3, 1)),
                 vel=np.zeros((3, 1)),
                 angle=np.zeros((3, 1)),
                 omega=np.zeros((3, 1)),
                 config="quadrotor",
                 ):
        super().__init__()
        self.pos = BaseSystem(pos)
        self.vel = BaseSystem(vel)
        self.angle = BaseSystem(angle)
        self.omega = BaseSystem(omega)

        # u_factor * U = M
        self.u_factor = np.diag([self.l, self.l, self.d / self.b])

        # The variables defined below relate to
        # specific configuration of the multirotor
        if config == "quadrotor":
            # Allocation matrix
            self.B = np.array([
                [1, 1, 1, 1],
                [0, -1, 0, 1],
                [-1, 0, 1, 0],
                [1, -1, 1, -1]
            ])
            self.b_gyro = np.vstack((-1, 1, -1, 1))

            # Distance vector to each rotor (P. Pounds et al., 2010)
            self.d_rotor = np.array([
                [self.l, 0, 0],
                [0, -self.l, 0],
                [-self.l, 0, 0],
                [0, self.l, 0],
            ])

            # Number of rotors
            self.n = 4

        elif config == "hexarotor":
            s2 = 1/2
            s3 = np.sqrt(3)/2
            self.B = np.array([
                [1, 1, 1, 1, 1, 1],
                [0, -s3, -s3, 0, s3, s3],
                [-1, -s2, s2, 1, s2, -s2],
                [1, -1, 1, -1, 1, -1]
            ])
            self.b_gyro = np.vstack((1, -1, 1, -1, 1, -1))

            l = self.l
            self.d_rotor = np.array([
                [l, 0, 0],
                [l*s2, -l*s3, 0],
                [-l*s2, -l*s3, 0],
                [-l, 0, 0],
                [-l*s2, l*s3, 0],
                [l*s2, l*s3, 0],
            ])

            self.n = 6
        else:
            raise ValueError

        self.fmax = self.m * self.g * 2 / self.n

    def deriv(self, pos, vel, angle, omega, F, M, F_wind):
        m, g, J, Jinv, e3 = self.m, self.g, self.J, self.Jinv, self.e3

        phi, theta, psi = angle.ravel()
        dcm = fym.utils.rot.angle2dcm(psi, theta, phi)
        Rot = dcm.T

        dpos = vel
        dvel = -g * e3 + F * Rot.dot(e3) / m + F_wind / m
        dangle = np.array([
            [1, np.sin(phi) * np.tan(theta), np.cos(phi) * np.tan(theta)],
            [0, np.cos(phi), -np.sin(phi)],
            [0, np.sin(phi) / np.cos(theta), np.cos(phi) / np.cos(theta)]
        ]).dot(omega)
        domega = Jinv.dot(
            M - np.cross(omega, J.dot(omega), axis=0))
        return dpos, dvel, dangle, domega

    def get_Omega(self, f):
        f = np.clip(f, 0, self.fmax)
        Omega = self.b_gyro.T.dot(np.sqrt(f / self.b))
        return Omega

    def get_FM_wind(self, f, vel, omega, windvel):
        relvel = windvel - vel

        f = np.clip(f, 0, self.fmax)

        # Frame drag
        F_drag = 1/2 * self.rho * self.CdA * np.linalg.norm(relvel) * relvel

        # Blade Flapping
        F_blade = np.zeros((3, 1))
        M_blade = np.zeros((3, 1))
        for fi, di in zip(f, self.d_rotor):
            di = di[:, None]
            if fi != 0:
                Omegai = np.sqrt(fi / self.b)
                vr = relvel + np.cross(omega, di, axis=0)
                mur = np.linalg.norm(vr[:2]) / (Omegai * self.R)
                psir = np.arctan2(vr[1, 0], vr[0, 0])
                lambdah = np.sqrt(self.CT / 2)
                gamma = self.rho * self.a0 * self.c * self.R**4 / self.Jr
                v1s = 1 / (1 + mur**2 / 2) * 4 / 3 * (
                    self.CT / self.sigma * 2 / 3 * mur * gamma / self.a0 + mur)
                u1s = 1 / (1 - mur**2 / 2) * mur * (
                    4 * self.thetat - 2 * lambdah**2)
                alpha1s, beta1s = np.array([
                    [np.cos(psir), -np.sin(psir)],
                    [np.sin(psir), np.cos(psir)]
                ]).dot(np.vstack((u1s, v1s)))

                ab = np.vstack((
                    -np.sin(alpha1s),
                    -np.cos(alpha1s) * np.sin(beta1s),
                    np.cos(alpha1s) * np.cos(beta1s) - 1))

                F_blade += self.b * Omegai**2 * ab
                M_blade += np.cross(di, self.b * Omegai**2 * ab, axis=0)

        F_wind = F_blade + F_drag
        M_wind = M_blade

        return F_wind, M_wind

    def get_W(self, t):
        W = np.eye(self.n)
        # W[0, 0] += np.sum([
        #     smooth_jump(t, 10, 5, -0.4, half=True),
        #     smooth_jump(t, 20, 5, -0.2, half=True),
        # ])
        # W[1, 1] += smooth_jump(t, 15, 5, -0.2, half=True)

        # W = np.diag([0.8, 0.9, 1, 1])

        return np.clip(W, 0, 1)

    def get_M_gyro(self, f, omega):
        Omega = self.get_Omega(f)
        M_gyro = -self.Jr * Omega * np.cross(omega, self.e3, axis=0)
        return M_gyro

    def get_M_drag(self, omega):
        M_drag = -self.kr.dot(omega)
        return M_drag

    def set_dot(self, t, f, windvel, windpqr):
        pos, vel, angle, omega = self.observe_list()

        f = np.clip(f, 0, self.fmax)

        W = self.get_W(t)
        u = self.B.dot(W).dot(f)
        F, U = u[:1], u[1:]
        M = self.u_factor.dot(U)

        relomega = omega - windpqr

        # Gyro
        M_gyro = self.get_M_gyro(f, omega)

        # Angular drag
        M_drag = self.get_M_drag(relomega)

        F_wind, M_wind = self.get_FM_wind(f, vel, relomega, windvel)

        M = M + M_gyro + M_drag + M_wind

        dots = self.deriv(pos, vel, angle, omega, F, M, F_wind)

        self.pos.dot, self.vel.dot, self.angle.dot, self.omega.dot = dots

    def hat(self, v):
        v1, v2, v3 = v.squeeze()
        return np.array([
            [0, -v3, v2],
            [v3, 0, -v1],
            [-v2, v1, 0]
        ])


class Wind():
    def __init__(self, dt=0.01):
        # Dryden turbulence approximated by sinuosidals
        dryden_model = DrydenGustModel(dt)
        dryden_model.reset()
        length = cfg.final_time / dt + 1
        dryden_model.simulate(int(length))
        self.dryden_lin = scipy.interpolate.interp1d(
            dryden_model.t, dryden_model.vel_lin)
        self.dryden_ang = scipy.interpolate.interp1d(
            dryden_model.t, dryden_model.vel_ang)

    def get(self, t):
        turblin, turbang = self.turb(t)
        return self.mean() + self.gust(t) + turblin, turbang

    def mean(self):
        """Mean wind velocity resolved in the earth frame"""
        return np.vstack((4, 4, 0))

    def gust(self, t):
        vx = np.sum([
            smooth_jump(t, 5, 8, 2),
            smooth_jump(t, 10, 10.5, 4),
            smooth_jump(t, 14.5, 15, -4),
        ])
        vy = np.sum([
            smooth_jump(t, 10, 10.5, 4),
            smooth_jump(t, 14.5, 15, -4)
        ])
        vz = 0
        return np.vstack((vx, vy, vz))

    def turb(self, t):
        """Dryden model"""
        return self.dryden_lin(t)[:, None], self.dryden_ang(t)[:, None]


"""
Dryden Model:
    https://github.com/eivindeb/pyfly/blob/master/pyfly/dryden.py
"""


class Filter:
    def __init__(self, num, den):
        """
        Wrapper for the scipy LTI system class.
        :param num: numerator of transfer function
        :param den: denominator of transfer function
        """
        self.filter = lti(num, den)
        self.x = None
        self.y = None
        self.t = None

    def simulate(self, u, t):
        """
        Simulate filter
        :param u: filter input
        :param t: time steps for which to simulate
        :return: filter output
        """
        if self.x is None:
            x_0 = None
        else:
            x_0 = self.x[-1]

        self.t, self.y, self.x = self.filter.output(U=u, T=t, X0=x_0)

        return self.y

    def reset(self):
        """
        Reset filter
        :return:
        """
        self.x = None
        self.y = None
        self.t = None


class DrydenGustModel:
    def __init__(self, dt, b=1, h=10, V_a=10, intensity=None):
        """
        Python realization of the continuous Dryden Turbulence Model
            (MIL-F-8785C).
        :param dt: (float) band-limited white noise input sampling time.
        :param b: (float) wingspan of aircraft
        :param h: (float) Altitude of aircraft
        :param V_a: (float) Airspeed of aircraft
        :param intensity: (str) Intensity of turbulence, one of
            ["light", "moderate", "severe"]
        """
        # For fixed (nominal) altitude and airspeed
        h = h  # altitude [m]
        V_a = V_a  # airspeed [m/s]

        # Conversion factors
        # 1 meter = 3.281 feet
        meters2feet = 3.281
        feet2meters = 1 / meters2feet
        # 1 knot = 0.5144 m/s
        knots2mpers = 0.5144

        if intensity is None:
            W_20 = 15 * knots2mpers  # light turbulence
        elif intensity == "light":
            W_20 = 15 * knots2mpers  # light turbulence
        elif intensity == "moderate":
            W_20 = 30 * knots2mpers  # moderate turbulence
        elif intensity == "severe":
            W_20 = 45 * knots2mpers  # severe turbulence
        else:
            raise Exception("Unsupported intensity type")

        # Convert meters to feet and follow MIL-F-8785C spec
        h = h * meters2feet
        b = b * meters2feet
        V_a = V_a * meters2feet
        W_20 = W_20 * meters2feet

        # Turbulence intensities
        sigma_w = 0.1 * W_20
        sigma_u = sigma_w / (0.177 + 0.000823 * h) ** 0.4
        sigma_v = sigma_u

        # Turbulence length scales
        L_u = h / (0.177 + 0.000823 * h) ** 1.2
        L_v = L_u
        L_w = h

        K_u = sigma_u * math.sqrt((2 * L_u) / (math.pi * V_a))
        K_v = sigma_v * math.sqrt((L_v) / (math.pi * V_a))
        K_w = sigma_w * math.sqrt((L_w) / (math.pi * V_a))

        T_u = L_u / V_a
        T_v1 = math.sqrt(3.0) * L_v / V_a
        T_v2 = L_v / V_a
        T_w1 = math.sqrt(3.0) * L_w / V_a
        T_w2 = L_w / V_a

        K_p = sigma_w * math.sqrt(0.8 / V_a) * (
            ((math.pi / (4 * b)) ** (1 / 6)) / ((L_w) ** (1 / 3)))
        K_q = 1 / V_a
        K_r = K_q

        T_p = 4 * b / (math.pi * V_a)
        T_q = T_p
        T_r = 3 * b / (math.pi * V_a)

        self.filters = {
            "H_u": Filter(feet2meters * K_u, [T_u, 1]),
            "H_v": Filter([feet2meters * K_v * T_v1, feet2meters * K_v],
                          [T_v2 ** 2, 2 * T_v2, 1]),
            "H_w": Filter([feet2meters * K_w * T_w1, feet2meters * K_w],
                          [T_w2 ** 2, 2 * T_w2, 1]),
            "H_p": Filter(K_p, [T_p, 1]),
            "H_q": Filter([-K_w * K_q * T_w1, -K_w * K_q, 0],
                          [T_q * T_w2 ** 2,
                           T_w2 ** 2 + 2 * T_q * T_w2,
                           T_q + 2 * T_w2,
                           1]),
            "H_r": Filter([K_v * K_r * T_v1, K_v * K_r, 0],
                          [T_r * T_v2 ** 2,
                           T_v2 ** 2 + 2 * T_r * T_v2,
                           T_r + 2 * T_v2,
                           1]),
        }

        self.np_random = None
        self.seed()

        self.dt = dt
        self.sim_length = None

        self.noise = None

        self.vel_lin = None
        self.vel_ang = None

    def seed(self, seed=None):
        """
        Seed the random number generator.
        :param seed: (int) seed.
        :return:
        """
        self.np_random = np.random.RandomState(seed)

    def _generate_noise(self, size):
        return np.sqrt(np.pi / self.dt) * (
            self.np_random.standard_normal(size=(4, size)))

    def reset(self, noise=None):
        """
        Reset model.
        :param noise: (np.array) Input to filters, should be four sequences of
            Gaussianly distributed numbers.
        :return:
        """
        self.vel_lin = None
        self.vel_ang = None
        self.sim_length = 0

        if noise is not None:
            assert len(noise.shape) == 2
            assert noise.shape[0] == 4
            noise = noise * math.sqrt(math.pi / self.dt)
        self.noise = noise

        for filter in self.filters.values():
            filter.reset()

    def simulate(self, length):
        """
        Simulate turbulence by passing band-limited Gaussian noise of length
            through the shaping filters.
        :param length: (int) the number of steps to simulate.
        :return:
        """
        t_span = [self.sim_length, self.sim_length + length]

        t = np.linspace(t_span[0] * self.dt, t_span[1] * self.dt, length)

        if self.noise is None:
            noise = self._generate_noise(t.shape[0])
        else:
            if self.noise.shape[-1] >= t_span[1]:
                noise = self.noise[:, t_span[0]:t_span[1]]
            else:
                nstart_i = t_span[0] % self.noise.shape[-1]
                remaining_nlen = self.noise.shape[-1] - nstart_i
                if remaining_nlen >= length:
                    noise = self.noise[:, nstart_i:nstart_i + length]
                else:
                    if length - remaining_nlen > self.noise.shape[-1]:
                        n22 = length - remaining_nlen - self.noise.shape[-1]
                        concat_noise = np.pad(
                            self.noise,
                            ((0, 0),
                             (0, n22)),
                            mode="wrap")
                    else:
                        concat_noise = self.noise[:, :length - remaining_nlen]
                    noise = np.concatenate(
                        (self.noise[:, nstart_i:], concat_noise), axis=-1)

        vel_lin = np.array([self.filters["H_u"].simulate(noise[0], t),
                            self.filters["H_v"].simulate(noise[1], t),
                            self.filters["H_w"].simulate(noise[2], t)])

        vel_ang = np.array([self.filters["H_p"].simulate(noise[3], t),
                            self.filters["H_q"].simulate(noise[1], t),
                            self.filters["H_r"].simulate(noise[2], t)])

        if self.vel_lin is None:
            self.vel_lin = vel_lin
            self.vel_ang = vel_ang
        else:
            self.vel_lin = np.concatenate((self.vel_lin, vel_lin), axis=1)
            self.vel_ang = np.concatenate((self.vel_ang, vel_ang), axis=1)

        self.sim_length += length
        self.t = t


class ReferenceSystem(BaseSystem):
    def __init__(self):
        super().__init__(cfg.xm_init)
        self.cmd = Command()

    def set_dot(self, c):
        # dim(c) = 3
        xr = self.state
        self.dot = cfg.Am.dot(xr) + cfg.Bm.dot(c)


class Command():
    def get(self, t):
        p = np.sum([
            sharp_jump(t, 5, 1),
            sharp_jump(t, 10, -2),
            sharp_jump(t, 15, 2),
            sharp_jump(t, 20, -2),
            sharp_jump(t, 25, 1),
        ])
        q = np.sum([
            sharp_jump(t, 3, 1),
            sharp_jump(t, 8, -2),
            sharp_jump(t, 13, 2),
            sharp_jump(t, 18, -2),
            sharp_jump(t, 23, 1),
        ])
        r = 0
        return np.vstack((p, q, r)) * np.deg2rad(35)


class CMRAC(BaseSystem):
    def __init__(self, P, B, Gamma):
        super().__init__(shape=cfg.W_init.shape)
        self.P = P
        self.B = B
        self.Gamma = Gamma

    def set_dot(self, e, phi, composite_term):
        self.dot = self.Gamma * (
            phi.dot(e.T).dot(self.P).dot(self.B)
            + composite_term
        )


class PerformanceIndex(BaseSystem):
    def set_dot(self, x, u):
        self.dot = 1/2 * (x.T.dot(cfg.Q_lyap).dot(x) + u.T.dot(cfg.R).dot(u))


class MRACEnv(BaseEnv):
    def __init__(self):
        super().__init__(**cfg.MRAC.env_kwargs)
        self.multirotor = MultiRotor(config=cfg.multirotor_config)
        self.xr = ReferenceSystem()

        self.B = self.multirotor.Jinv.dot(
            self.multirotor.u_factor.dot(self.multirotor.B[1:]))
        self.P = scipy.linalg.solve_continuous_are(
            cfg.Am, self.B, cfg.Q_lyap, cfg.R)

        self.W = CMRAC(self.P, self.B, cfg.MRAC.Gamma)
        self.J = PerformanceIndex()

        # Not BaseEnv or BaseSystem
        self.cmd = Command()
        self.wind = Wind()

        self.ut = self.multirotor.m * self.multirotor.g / self.multirotor.n

        self.logger = fym.logging.Logger(Path(cfg.dir, "mrac-env.h5"))
        self.logger.set_info(cfg=cfg)

    def step(self):
        *_, done = self.update()

        return done

    def set_dot(self, t):
        mult_x, xr, W, J = self.observe_list()

        x = self.get_x(mult_x)
        c = self.cmd.get(t)

        xa = self.multirotor.observe_flat()
        phi = self.basis(xa, xr, c)

        windvel, windpqr = self.wind.get(t)

        e = x - xr

        u = self.get_input(e, W, phi)

        self.multirotor.set_dot(t, u, windvel, windpqr)
        self.xr.set_dot(c)
        self.W.set_dot(e, phi, 0)
        self.J.set_dot(e, u)

    def get_x(self, mult_x):
        return mult_x[-1]

    def get_input(self, e, W, phi):
        ut = self.ut
        un = -cfg.Rinv.dot(self.B.T).dot(self.P).dot(e)
        ua = -W.T.dot(phi)
        return ut + un + ua

    def basis(self, x, xr, c):
        # xa = np.vstack((x, x - xr))
        x = x[3:, None]
        return np.vstack((x, np.roll(x[-3:], 1) * np.roll(x[-3:], 2)))

    def logger_callback(self, i, t, y, t_hist, ode_hist):
        mult_x, xr, W, J = self.observe_list(y)

        x = self.get_x(mult_x)
        c = self.cmd.get(t)
        xa = y[self.multirotor.flat_index]
        phi = self.basis(xa, xr, c)

        e = x - xr
        u = self.get_input(e, W, phi)

        # e_HJB = self.get_HJB_error(t, x, xr, W, c)

        return dict(
            t=t, x=x, xr=xr, W=W, J=J,
            e=e, c=c, u=u,
            # e_HJB=e_HJB,
        )

#     def get_HJB_error(self, t, x, xr, W, c):
#         e = x - xr
#         Wcirc = self.x.unc.parameter(t)
#         phi = self.basis(x, xr, c)

#         B, R = cfg.B, cfg.R

#         grad_V_tilde = - B.dot(self.BBTinv).dot(R).dot(W.T).dot(phi)
#         u_tilde = W.T.dot(phi)

#         return np.sum([
#             e.T.dot(self.P).dot(cfg.B).dot(W.T.dot(phi) - Wcirc.T.dot(phi)),
#             grad_V_tilde.T.dot(cfg.Am).dot(e),
#             u_tilde.T.dot(R).dot(Wcirc.T).dot(phi),
#             -0.5 * u_tilde.T.dot(R).dot(u_tilde)
#         ])


class ResidualFilter(BaseEnv):
    def __init__(self, basis, B):
        super().__init__()
        n, m = cfg.W_init.shape
        self.xi = BaseSystem(shape=(n, 1))
        self.z = BaseSystem(shape=(m, 1))

        self.Bdagger = np.linalg.pinv(B)
        self.tauf = cfg.FECMRAC.tauf

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
        return - 1 / self.tauf * (self.Bdagger.dot(e) - z)


class FECMRACAgent():
    def __init__(self):
        n, m = cfg.W_init.shape
        self.F = np.zeros((n, n))
        self.G = np.zeros((n, m))

        self.best_F = self.F.copy()
        self.best_G = self.G.copy()

        self.logger = fym.logging.Logger(Path(cfg.dir, "fecmrac-agent.h5"))

    def close(self):
        self.logger.close()

    def get_action(self, obs):
        self.update(obs)

        F, G = self.best_F, self.best_G

        t, *_ = obs
        eigs, _ = self.get_eig(F, cfg.FECMRAC.threshold)
        self.logger.record(
            t=t,
            eigs=cfg.FECMRAC.Gamma * eigs,
        )

        return F, G

    def update(self, obs):
        t, y, xi, phi = obs

        p, q = self._get_pq(obs)

        self.F = q * self.F + p * xi.dot(xi.T)
        self.G = q * self.G + p * xi.dot(y.T)

        if self.get_eig(self.F)[0][0] > self.get_eig(self.best_F)[0][0]:
            self.best_F = self.F
            self.best_G = self.G

    def _get_pq(self, obs):
        t, y, xi, phi = obs

        p = cfg.HMRAC.env_kwargs["dt"]
        xidot = - (xi - phi) / cfg.FECMRAC.tauf
        nxidot = np.linalg.norm(xidot)
        k = self._get_k(nxidot)
        q = 1 - k * p

        return p, q

    def _get_k(self, nxidot):
        return (cfg.FECMRAC.kL
                + ((cfg.FECMRAC.kU - cfg.FECMRAC.kL)
                   * np.tanh(cfg.FECMRAC.theta * nxidot)))

    def get_eig(self, A, threshold=1e-10):
        eigs, eigv = np.linalg.eig(A)
        sort = np.argsort(np.real(eigs))  # sort in ascending order
        eigs = np.real(eigs[sort])
        eigv = np.real(eigv[:, sort])
        eigs[eigs < threshold] = 0
        return eigs, eigv


class HMRACEnv(MRACEnv, BaseEnv):
    def __init__(self):
        BaseEnv.__init__(self, **cfg.HMRAC.env_kwargs)
        self.multirotor = MultiRotor(config=cfg.multirotor_config)
        self.xr = ReferenceSystem()

        self.B = self.multirotor.Jinv.dot(
            self.multirotor.u_factor.dot(self.multirotor.B[1:]))
        self.P = scipy.linalg.solve_continuous_are(
            cfg.Am, self.B, cfg.Q_lyap, cfg.R)

        self.W = CMRAC(self.P, self.B, cfg.HMRAC.Gamma)
        self.What = BaseSystem(cfg.W_init)
        self.J = PerformanceIndex()
        self.filter = ResidualFilter(basis=self.basis, B=self.B)

        # Not BaseEnv or BaseSystem
        self.cmd = Command()
        self.wind = Wind()

        self.ut = self.multirotor.m * self.multirotor.g / self.multirotor.n
        self.BBTinv = np.linalg.inv(self.B.dot(self.B.T))

        self.logger = fym.logging.Logger(Path(cfg.dir, "hmrac-env.h5"))
        self.logger.set_info(cfg=cfg)

    def reset(self):
        super().reset()
        return self.observation()

    def step(self, action):
        F, G = action
        *_, done = self.update(F=F, G=G)
        return self.observation(), done

    def observation(self):
        mult_x, xr, W, What, J, (xi, z) = self.observe_list()

        x = self.get_x(mult_x)
        e = x - xr
        y = self.filter.get_y(e, z)
        t = self.clock.get()
        c = self.cmd.get(t)
        xa = self.multirotor.observe_flat()
        phi = self.basis(xa, xr, c)
        return t, y, xi, phi

    def set_dot(self, t, F, G):
        mult_x, xr, W, What, J, (xi, z) = self.observe_list()

        x = self.get_x(mult_x)
        c = self.cmd.get(t)
        xa = self.multirotor.observe_flat()
        phi = self.basis(xa, xr, c)

        windvel, windpqr = self.wind.get(t)

        e = x - xr

        u = self.get_input(e, W, phi)

        B = self.B
        composite_term = phi.dot(
            - e.T.dot(cfg.Am.T).dot(self.BBTinv).dot(B)
            + phi.T.dot(What).dot(cfg.R)
            - phi.T.dot(W).dot(cfg.R)
        )

        self.multirotor.set_dot(t, u, windvel, windpqr)
        self.xr.set_dot(c)
        self.W.set_dot(e, phi, composite_term)
        self.What.dot = - cfg.FECMRAC.Gamma * (F.dot(What) - G)
        self.J.set_dot(e, u)
        self.filter.set_dot(e, phi, u)

    def logger_callback(self, i, t, y, t_hist, ode_hist):
        mult_x, xr, W, What, J, (xi, z) = self.observe_list(y)

        x = self.get_x(mult_x)
        c = self.cmd.get(t)
        xa = y[self.multirotor.flat_index]
        phi = self.basis(xa, xr, c)

        e = x - xr
        u = self.get_input(e, W, phi)

        # e_HJB = self.get_HJB_error(t, x, xr, W, c)

        return dict(
            t=t, x=x, xr=xr, W=W, What=What, J=J, xi=xi, z=z,
            e=e, c=c, u=u,
            # e_HJB=e_HJB,
        )
