"""Consensus Representation


@author: Soufiane Mourragui

This module computes the consensus representation between two datasets, by:
- Computing the domain-specific factors.
- Computing the principal vectors from source and target.
- Interpolating between the sets of principal vectors.
- Using KS statistics, finds the intermediate point where source and target are best balanced.


Example
-------
    Examples are given in the vignettes.

Notes
-------
	Examples are given in the vignette
	
References
-------
	[1] Mourragui, S., Loog, M., Reinders, M.J.T., Wessels, L.F.A. (2019)
    PRECISE: A domain adaptation approach to transfer predictors of drug response
    from pre-clinical models to tumors
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.externals.joblib import Parallel, delayed
from scipy.stats import ks_2samp

from precise.principal_vectors import PVComputation
from precise.intermediate_factors import IntermediateFactors

class ConsensusRepresentation:

    def __init__(self, n_factors,
                n_pv,
                n_representations=100,
                dim_reduction='pca',
                dim_reduction_target=None,
                total_variance=10**3,
                n_jobs=1):
        """
        Parameters
        -------
        n_factors: int
            Number of domain-specific factors.

        n_pv: int
            Number of principal vectors.

        n_representations: int, optional, default to 100
            Number of interpolated features between source and target principal vectors.

        dim_reduction : str, default to 'pca' 
            Dimensionality reduction method for the source data,
            i.e. 'pca', 'ica', 'nmf', 'fa', 'sparsepca', pls'.

        dim_reduction_target : str, default to None 
            Dimensionality reduction method for the target data.

        total_variance: float, default to 10^3
            Total variance in both source and target after total variance normalization.

        n_jobs: int (optional, default to 1)
            Number of jobs for computation.
        """
        self.n_factors = n_factors
        self.n_pv = n_pv
        self.n_representations = n_representations
        self.dim_reduction = dim_reduction
        self.dim_reduction_target = dim_reduction_target
        self.total_variance = total_variance

        self.source_data = None
        self.target_data = None
        self.source_components_ = None
        self.target_components_ = None

        self.intermediate_factors_ = None

        self.consensus_components_ = None

        self.n_jobs = 1

    def fit(self, source_data, target_data):
        """
        Compute the consensus representation between two set of data.

        IMPORTANT: Same genes have to be given for source and target, and in same order

        Parameters
        -------
        source_data : np.ndarray, shape (n_components, n_genes)
            Source dataset

        target_data : np.ndarray, shape (n_components, n_genes)
            Target dataset

        Return values
        -------
        self: returns an instance of self.
        """
        # Low-rank representation
        Ps = self.dim_reduction_source.fit(X_source, y_source).components_
        self.source_components_ = scipy.linalg.orth(Ps.transpose()).transpose()

        Pt = self.dim_reduction_target.fit(X_target, y_source).components_
        self.target_components_ = scipy.linalg.orth(Pt.transpose()).transpose()

        # Compute intermediate factors
        self.intermediate_factors_ = IntermediateFactors(self.n_representations)\
                                    .sample_flow(self.source_components_, self.target_components_)
        self.intermediate_factors_ = self.intermediate_factors_.transpose(1,0,2)

        # Normalize for total variance
        target_total_variance = np.sqrt(np.sum(np.var(target_data, 0)))
        normalized_target_data = target_data / target_total_variance
        normalized_target_data *= self.total_variance

        source_total_variance = np.sqrt(np.sum(np.var(source_data, 0)))
        normalized_source_data = source_data / source_total_variance
        normalized_source_data *= self.total_variance

        # Compute consensus representation
        self.consensus_components_ = []

        for i in range(self.n_pv):
            source_projected = intermediate_factors_[i].dot(normalized_source_data.transpose())
            target_projected = intermediate_factors_[i].dot(normalized_target_data.transpose())

            ks_stats = [
                ks_2samp(s,t)[0]
                for (s,t) in zip(source_projected, target_projected)
            ]

            self.consensus_components_.append(intermediate_factors_[i, np.argmin(ks_stats)])

        self.consensus_components_ = np.array(self.consensus_components_).transpose()

        return self.consensus_components_