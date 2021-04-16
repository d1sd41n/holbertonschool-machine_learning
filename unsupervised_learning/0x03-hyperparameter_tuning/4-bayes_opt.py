#!/usr/bin/env python3
"""[summary]
"""
import numpy as np
GP = __import__('2-gp').GaussianProcess


class BayesianOptimization:
    """[summary]
    """

    def __init__(self, f, X_init, Y_init, bounds, ac_samples,
                 l=1, sigma_f=1, xsi=0.01, minimize=True):
        """[summary]

        Args:
            f ([type]): [description]
            X_init ([type]): [description]
            Y_init ([type]): [description]
            bounds ([type]): [description]
            ac_samples ([type]): [description]
            l (int, optional): [description]. Defaults to 1.
            sigma_f (int, optional): [description]. Defaults to 1.
            xsi (float, optional): [description]. Defaults to 0.01.
            minimize (bool, optional): [description]. Defaults to True.
        """
        self.f = f
        self.gp = GP(X_init,
                     Y_init, l,
                     sigma_f
                     )

        t_min, t_max = bounds
        self.X_s = np.linspace(t_min, t_max,
                               ac_samples).reshape(-1, 1)

        self.xsi = xsi

        self.minimize = minimize

    def acquisition(self):
        """[summary]

        Returns:
            [type]: [description]
        """
        X = self.gp.X
        mu_sample, _ = self.gp.predict(X)
        mu, sigma = self.gp.predict(self.X_s)
        sigma = sigma.reshape(-1, 1)
        with np.errstate(divide='warn'):
            if self.minimize is True:
                mu_sample_opt = np.amin(self.gp.Y)
                imp = (mu_sample_opt - mu - self.xsi).reshape(-1, 1)
            else:
                mu_sample_opt = np.amax(self.gp.Y)
                imp = (mu - mu_sample_opt - self.xsi).reshape(-1, 1)

            Z = imp / sigma
            EI = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            EI[sigma == 0.0] = 0.0
        X_next = self.X_s[np.argmax(EI)]
        return X_next, EI.reshape(-1)
