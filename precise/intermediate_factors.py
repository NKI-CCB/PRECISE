""" Intermediate Factors

@author: Soufiane Mourragui

This module computes the interpolated features between the principal vectors -- the one
linking source to target following the geodesics on the Grassmannian. We use the
equivalent formulation derived in [1] and represent this geodesics for each pair
of principal components.

Example
-------
    Examples are given in the vignettes.

Notes
-------
	Examples are given in the vignette
	
References
-------
	[1] Mourragui, S., Loog, M., Reinders, M.J.T., Wessels, L.F.A., "TO CHANGE"
"""

import numpy as np
import pandas as pd
from pathlib import Path
from joblib import Parallel, delayed

from precise.principal_vectors import PVComputation

class IntermediateFactors:
    """
    Handle the intermediate representations between 

    Attributes
    -------
    source_components_ : numpy.ndarray, shape (n_components, n_features)
        Loadings of the source factors, be them already aligned to target or not.
    
    target_components : numpy.ndarray, shape (n_components, n_features)
    	Loadings of the target factors, be them already aligned to source or not.
    
    intermediate_factors_ : numpy.ndarray, shape (n_representations, n_components, n_features)
        Loadings of intermediate factors along the geodesic path. Components are ordered
        by similarity, i.e. first components correspond to path between first PVs, etc.
    
    n_representations: int
        Number of representations along the geodesic path. If -1, means that the Geodesic Flow Kernel
        has been used instead.
    
    geodesic_matrix_: numpy.ndarray, shape (n_features, n_features)
        Geodesic Matrix for geodesic flow kernel.
    
    geodesic_flow_: method float:numpy.array
        Method that computes geodesic flow at a certain position.
    """

    def __init__(self, n_representations, n_jobs=1):
        """
        Parameters
        -------
        n_representations : int
            Number of representations to pick between source and target.
        n_jobs: int (optional, default to 1)
        	Number of jobs for computation.
        """
        self.n_representations = n_representations

        self.intermediate_factors_ = None
        self.source_components_ = None
        self.target_components_ = None

        self.n_jobs = 1

    def _compute_principal_vectors(self):
        n_pv = np.min([self.source_components_.shape[0],
                    self.target_components_.shape[0]])
        n_factors = {
            'source': self.source_components_.shape[0],
            'target': self.target_components_.shape[0]
        }

        self.principal_vectors_ = PVComputation(n_factors, n_pv)
        self.principal_vectors_.compute_principal_vectors(self.source_components_,
                                                        self.target_components_)

    def _compute_flow_time(t, principal_vectors):
        Pi = np.sin( (1-t) * principal_vectors.angles_)\
            /np.sin(principal_vectors.angles_)
        Pi[np.isnan(Pi)] = 1-t # Asymptotic value of sin/sin in 0

        Xi = np.sin( t * principal_vectors.angles_)\
            / np.sin(principal_vectors.angles_)
        Xi[np.isnan(Xi)] = t # Asymptotic value of sin/sin in 0

        return (principal_vectors.source_components_.T*Pi \
            + principal_vectors.target_components_.T*Xi).T

    def sample_flow(self, source_components, target_components, already_aligned=False):
        """
        Sample intermediate subspaces (i.e. set of factors) uniformely along the geodesic flow.

        IMPORTANT: Same genes have to be given for source and target, and in same order

        Parameters
        -------
        source_components : np.ndarray, shape (n_components, n_features)
            Source factors
        
        target_components : np.ndarray, shape (n_components, n_features)
            Target factors
        
        already_aligned : boolean (optional, default to False)
            Whether the components are already aligned (i.e. are they PV or not).

        Return values
        -------
        Intermediate subspace, numpy.ndarray of shape (n_representations + 1, n_components, n_features).
        """
        self.source_components_ = source_components
        self.target_components_ = target_components

        # Compute the principal vectors
        if not already_aligned:
            self._compute_principal_vectors()
        else:
            self.principal_vectors_.source_components_ = self.source_components_
            self.principal_vectors_.target_components = self.target_components_
        
        # Sample at different uniformly distributed time points
        if self.n_representations == -1:
            t_sample = np.array([1])
        else:
            t_sample = np.linspace(0, 1, self.n_representations + 1)

        if self.n_jobs >= 2:
            return np.array(
                Parallel(n_jobs=self.n_jobs)\
                        (delayed(IntermediateFactors._compute_flow_time)(t, self.principal_vectors_)\
            			     for t in t_sample)
            )
        else:
            return np.array([IntermediateFactors._compute_flow_time(t, self.principal_vectors_) for t in t_sample])

    def compute_geodesic_matrix(self, source_components, target_components):
        """
        Return method for computing the domain-invariant kernel of Geodesic Flow Kernel.

        Parameters
        -------
        source_components : np.ndarray, shape (n_components, n_features)
            Source factors
        
        target_components : np.ndarray, shape (n_components, n_features)
            Target factors

        Return values
        -------
        Method that takes two p-dimensional vector and returns their domain-invariant
        scalar product.
        """
        self.source_components_ = source_components
        self.target_components_ = target_components

        self._compute_principal_vectors()

        diag_term = (self.principal_vectors_.angles_ - np.cos(self.principal_vectors_.angles_)*np.sin(self.principal_vectors_.angles_)) \
        				/ 2 / self.principal_vectors_.angles_ / np.power(np.sin(self.principal_vectors_.angles_), 2)
        off_diag_term = (np.sin(self.principal_vectors_.angles_) - np.cos(self.principal_vectors_.angles_)*self.principal_vectors_.angles_) \
        				/ 2 / np.power(np.sin(self.principal_vectors_.angles_),2) / self.principal_vectors_.angles_
        # Correct for extreme case when theta = 0
        diag_term[np.isnan(diag_term)] = 1./3.
        diag_term[np.isinf(diag_term)] = 1./3.
        off_diag_term[np.isnan(off_diag_term)] = 1./6.
        off_diag_term[np.isinf(off_diag_term)] = 1./6.
        diag_term = np.diag(diag_term)
        off_diag_term = np.diag(off_diag_term)

        self.G_matrix = np.block([
        	[diag_term, off_diag_term],
        	[off_diag_term, diag_term]
        ])

        self.projection = np.block([self.principal_vectors_.source_components_.transpose(), self.principal_vectors_.target_components_.transpose()])

        return self.G_matrix
        #return lambda x,y: IntermediateFactors._compute_domain_invariant_scalar_product(x, y, self.projection, self.G_matrix)

    def _compute_domain_invariant_scalar_product(x, y, projection, G_matrix):
        x_p = x.dot(projection)
        y_p = y.dot(projection)

        return x_p.dot(G_matrix).dot(y_p.transpose())

