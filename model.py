#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 12 18:11:16 2025

@author: akshayjacobthomas
"""

from scipy.stats import qmc
import pandas as pd
from math import ceil, floor, log2
import os
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from typing import Optional, Sequence, Union, List, Dict
from scipy.stats import qmc, norm
from pathlib import Path

ArrayLike = Union[float, int, Sequence[float], np.ndarray]

class BayesianFLow:
    """
    
    class that handles data creation and surrogate modeling
    """

    def __init__(self, dim: int,n_train_samples: int, n_test_samples: int): # xcould add a boudna file next time
        
        self.dim = dim
        self.n_train_samples = n_train_samples
        self.n_test_samples = n_test_samples
        
        self.lb = np.array([0, 1e3, 1e-3, 10, 1.0e6])
        self.ub = np.array([1, 1e7, 100, 1e3, 5.0e6])
        
      
    def _sobol_exact(self, n: int, d: int, scramble: bool = True) -> np.ndarray:
        """Return exactly `n` Sobol points in [0,1]^d."""
        if n <= 0:
            raise ValueError("n must be positive.")
        sampler = qmc.Sobol(d=d, scramble=scramble)
        m = int(ceil(log2(n)))
        X = sampler.random_base2(m=m)  # 2**m points
        
        return X[:n]
        
    def create_training_set(self, scramble_=True) -> np.ndarray:
        """
        create the training set
        """
        X_= self._sobol_exact(self.n_train_samples, self.dim, scramble=scramble_)
        X = qmc.scale(X_, self.lb, self.ub)
        
        
        df = pd.DataFrame(X, columns=[f"x{i+1}" for i in range(self.dim)])
        df.to_csv("train_set.csv", index=False)
        print(f"Training set saved to train_set.csv with shape {df.shape}")

    def create_test_set(self, scramble_=True) -> np.ndarray:

        X_ = self._sobol_exact(self.n_test_samples, self.dim, scramble=scramble_)
        X = qmc.scale(X_, self.lb, self.ub)
        
        df = pd.DataFrame(X, columns=[f"x{i+1}" for i in range(self.dim)])
        df.to_csv("test_set.csv", index=False)
        print(f"Test set saved to test_set.csv with shape {df.shape}")
        
def main():
    # Instantiate and run
    flow = BayesianFLow(dim=5, n_train_samples=1024, n_test_samples=256)
    flow.create_training_set()
    flow.create_test_set()


if __name__ == "__main__":
    main()