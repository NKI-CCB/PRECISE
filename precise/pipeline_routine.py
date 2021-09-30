"""Pipeline Routine

@author: Soufiane Mourragui

Different routines used for training PRECISE. These correspond to the domain adaptation
step prior to regression.

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
from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler
from scipy.stats import ks_2samp

from precise.intermediate_factors import IntermediateFactors
from precise.principal_vectors import PVComputation


class FlowProjector(BaseEstimator):
    """Project on the geodesic.

    Given source and target data, computes the domain-specific factors, aligns them
    to get the principal vectors and finally interpolates between source PVs and
    target PVs. Data can then be projected on all these intermediate features.

    Attributes
    -------
    """

    def __init__(self, source_data, target_data, n_factors, n_pv,
                dim_reduction='pca',
                dim_reduction_target=None,
                n_representations=100,
                use_data=False,
                mean_center=False,
                std_unit=False):
        """
        Parameters
        -------
        source_data: np.ndarray (n_samples, n_genes)
            Data use as source, e.g. cell line or PDX transcriptome read outs.

        target_data: np.ndarray (n_samples, n_genes)
            Data use as target, e.g. tumor transcriptome read outs.

        n_factors: int
            Number of domain-invariant factors.

        n_pv: int
            Number of principal vectors.

        dim_reduction : str, default to 'pca' 
            Dimensionality reduction method for the source data,
            i.e. 'pca', 'ica', 'nmf', 'fa', 'sparsepca', pls'.

        dim_reduction_target : str, default to None 
            Dimensionality reduction method for the target data.

        n_representations: int, optional default to 100
            Number of interpolated features.

        use_data: bool, optional, default to False
            Whether data given additionally in fit should be used in domain-adaptation.

        mean_center : bool, optional, default to False
            Whether X_source features (i.e. genes) should be mean-centered.

        std_unit : bool, optional, default to False
            Whether X_source features (i.e. genes) should be standardized.
        """
        self.source_data = source_data
        self.target_data = target_data

        self.n_factors = n_factors
        self.n_pv = n_pv
        self.dim_reduction = dim_reduction
        self.dim_reduction_target = dim_reduction_target
        self.n_representations = n_representations
        self.use_data = use_data

        self.standard_scaler_input_ = StandardScaler(with_mean=mean_center, with_std=std_unit)
        self.standard_scaler_source_ = StandardScaler(with_mean=mean_center, with_std=std_unit)
        self.standard_scaler_target_ = StandardScaler(with_mean=mean_center, with_std=std_unit)

        self.pv_computation = PVComputation(
            n_factors = self.n_factors,
            n_pv = self.n_pv,
            dim_reduction = self.dim_reduction,
            dim_reduction_target = self.dim_reduction_target,
        )

        self.intermediate_factors = IntermediateFactors(
            n_representations = self.n_representations
        )

    def fit(self, X, y=None):
        """
        Computes the intermediate features between the pairs of principal vectors.

        Parameters
        -------
        X: numpy.ndarray, shape (n_samples, n_genes)
            Genomics data to consider

        y: numpy.ndarray, shape(n_samples, 1), optional, default to None
            Response data (optional, just for compliance with BaseEstimator)

        Returned Values
        -------
        self: returns an instance of self.
        """

        # Add X to source data if use_data set to True
        if self.use_data:
            if self.source_data is None or self.source_data.shape[0] == 0:
                self.source_data = X
            else:
                self.source_data = np.concatenate([self.source_data, X])

        # Standardize data
        self.standard_scaler_input_.fit(X)
        self.source_data = self.standard_scaler_source_.fit_transform(self.source_data)
        self.target_data = self.standard_scaler_target_.fit_transform(self.target_data)

        # Compute principal vectors
        self.pv_computation.fit(self.source_data, self.target_data, y)

        # Compute intermediate factors.
        self.flow = self.intermediate_factors.sample_flow(
                self.pv_computation.source_components_, 
                self.pv_computation.target_components_
        )

        # Concatenate feature representations before projection
        self.flow = np.concatenate(self.flow).transpose()

        return self

    def transform(self, X, y=None):
        """
        Project data along the geodesic path. 

        Parameters
        -------
        X: numpy.ndarray, shape (n_components, n_features)
            Genomics data use for prediction.

        Returned values
        -------
        X_projected: numpy.ndarray, shape (n_components, n_pv * n_representations)
            Genomics data projected along the flow.
        """

        return self.standard_scaler_input_.fit_transform(X).dot(self.flow)


class GeodesicMatrixComputer(BaseEstimator):
    """ Geodesic Flow Kernel computation.

    Compute the geodesic flow kernel matrix. We use the equivalent definition
    derived in [1] to make it faster. Principal vectors are therefore first
    computed to project onto them.

    Attributes
    -------
    """
    def __init__(self, source_data, target_data, n_factors, n_pv,
                dim_reduction='pca',
                dim_reduction_target=None,
                n_representations=1000,
                use_data=False,
                mean_center=False,
                std_unit=False):
        """
        Parameters
        -------
        source_data: np.ndarray (n_samples, n_genes)
            Data use as source, e.g. cell line or PDX transcriptome read outs.

        target_data: np.ndarray (n_samples, n_genes)
            Data use as target, e.g. tumor transcriptome read outs.

        n_factors: int
            Number of domain-invariant factors.

        n_pv: int
            Number of principal vectors.

        dim_reduction : str, default to 'pca' 
            Dimensionality reduction method for the source data,
            i.e. 'pca', 'ica', 'nmf', 'fa', 'sparsepca', pls'.

        dim_reduction_target : str, default to None 
            Dimensionality reduction method for the target data.

        n_representations: int, optional default to 100
            Number of interpolated features.

        use_data: bool, optional, default to False
            Whether data given additionally in fit should be used in domain-adaptation.

        mean_center : bool, optional, default to False
            Whether X_source features (i.e. genes) should be mean-centered.

        std_unit : bool, optional, default to False
            Whether X_source features (i.e. genes) should be standardized.
        """
        self.source_data = source_data
        self.target_data = target_data

        self.n_factors = n_factors
        self.n_pv = n_pv
        self.dim_reduction = dim_reduction
        self.dim_reduction_target = dim_reduction_target
        self.n_representations = n_representations
        self.use_data = use_data

        self.standard_scaler_input_ = StandardScaler(with_mean=mean_center, with_std=std_unit)
        self.standard_scaler_source_ = StandardScaler(with_mean=mean_center, with_std=std_unit)
        self.standard_scaler_target_ = StandardScaler(with_mean=mean_center, with_std=std_unit)
        
        self.pv_computation_ = PVComputation(
            n_factors=self.n_factors,
            n_pv=self.n_pv,
            dim_reduction=self.dim_reduction,
            dim_reduction_target=self.dim_reduction_target,
        )

        self.intermediate_factors = IntermediateFactors(
            n_representations=self.n_representations
        )


    def fit(self, X, y=None):
        """
        Computes the geodesic flow kernel matrix used in kernel ridge.

        Parameters
        -------
        X: numpy.ndarray, shape (n_samples, n_genes)
            Genomics data to consider

        y: numpy.ndarray, shape(n_samples, 1), optional, default to None
            Response data (optional, just for compliance with BaseEstimator)

        Returned Values
        -------
        self: returns an instance of self.
        """

        # Add X to source data if use_data set to True
        if self.use_data:
            if self.source_data is None or self.source_data.shape[0] == 0:
                self.source_data = X
            else:
                self.source_data = np.concatenate([self.source_data, X])

        # Standardize data
        self.standard_scaler_input_.fit(X)
        self.source_data = self.standard_scaler_source_.fit_transform(self.source_data)
        self.target_data = self.standard_scaler_target_.fit_transform(self.target_data)
        self.training_data = self.standard_scaler_input_.transform(X)

        # Compute principal vectors
        self.pv_computation_.fit(self.source_data, self.target_data, y)

        # Compute G, kernel matrix
        self.G_ = self.intermediate_factors.compute_geodesic_matrix(
                self.pv_computation_.source_components_, 
                self.pv_computation_.target_components_
        )

        # Compute projector
        self.projector_ = np.block([self.pv_computation_.source_components_.transpose(), self.pv_computation_.target_components_.transpose()])

        return self

    def _compute_kernel_matrix(self, X1, X2=None):
        X1_projected = X1.dot(self.projector_)
        if X2 is None:
            X2_projected = X1_projected
        else:
            X2_projected = X2.dot(self.projector_)

        return X1_projected.dot(self.G_).dot(X2_projected.transpose())

    def transform(self, X, y=None):
        """
        Compute the domain-invariant kernel matrix

        Parameters
        -------
        X: numpy.ndarray, shape (n_components, n_features)
            Genomics data use for prediction.

        Returned values
        -------
        X_projected: numpy.ndarray, shape (n_components, n_representations)
            Kernel matrix with source data (fed in fit method).
        """
        return self._compute_kernel_matrix(self.standard_scaler_input_.fit_transform(X), self.training_data)

class ConsensusRepresentation(BaseEstimator):
    """Consensus Representation computation.

    Compute the geodesic flow kernel matrix. We use the equivalent definition
    derived in [1] to make it faster. Principal vectors are therefore first
    computed to project onto them.

    Attributes
    -------
    """
    def __init__(self, source_data, target_data, n_factors, n_pv,
                dim_reduction='pca',
                dim_reduction_target=None,
                n_representations=1000,
                use_data=False,
                mean_center=False,
                std_unit=False):
        """
        Parameters
        -------
        source_data: np.ndarray (n_samples, n_genes)
            Data use as source, e.g. cell line or PDX transcriptome read outs.

        target_data: np.ndarray (n_samples, n_genes)
            Data use as target, e.g. tumor transcriptome read outs.

        n_factors: int
            Number of domain-invariant factors.

        n_pv: int
            Number of principal vectors.

        dim_reduction : str, default to 'pca' 
            Dimensionality reduction method for the source data,
            i.e. 'pca', 'ica', 'nmf', 'fa', 'sparsepca', pls'.

        dim_reduction_target : str, default to None 
            Dimensionality reduction method for the target data.

        n_representations: int, optional default to 100
            Number of interpolated features.

        use_data: bool, optional, default to False
            Whether data given additionally in fit should be used in domain-adaptation.

        mean_center : bool, optional, default to False
            Whether X_source features (i.e. genes) should be mean-centered.

        std_unit : bool, optional, default to False
            Whether X_source features (i.e. genes) should be standardized.
        """
        self.source_data = source_data
        self.target_data = target_data

        self.n_factors = n_factors
        self.n_pv = n_pv
        self.dim_reduction = dim_reduction
        self.dim_reduction_target = dim_reduction_target
        self.n_representations = n_representations
        self.use_data = use_data

        self.mean_center = mean_center
        self.std_unit = std_unit
        self.standard_scaler_input_ = StandardScaler(with_mean=mean_center, with_std=std_unit)
        self.standard_scaler_source_ = StandardScaler(with_mean=mean_center, with_std=std_unit)
        self.standard_scaler_target_ = StandardScaler(with_mean=mean_center, with_std=std_unit)

        self.pv_computation = PVComputation(
            n_factors = self.n_factors,
            n_pv = self.n_pv,
            dim_reduction = self.dim_reduction,
            dim_reduction_target = self.dim_reduction_target,
        )

        self.intermediate_factors = IntermediateFactors(
            n_representations = self.n_representations
        )

    def _find_common_representation(self):
        flow_vectors = self.flow.transpose(1,0,2)
        self.consensus_representation = []

        for i in range(self.n_pv):
            source_projected = flow_vectors[i].dot(self.source_data.transpose())
            target_projected = flow_vectors[i].dot(self.target_data.transpose())

            ks_stats = [
                ks_2samp(s,t)[0]
                for (s,t) in zip(source_projected, target_projected)
            ]

            self.consensus_representation.append(flow_vectors[i, np.argmin(ks_stats)])

        self.consensus_representation = np.array(self.consensus_representation).transpose()

        return self.consensus_representation


    def fit(self, X, y=None):
        """
        Computes the principal vectors, interpolates between them, projects source and target
        data, and finally computes, by comparing for each pair source and target projected data, 
        the point where these two quantities are comparable (using KS statistics).

        Parameters
        -------
        X: numpy.ndarray, shape (n_samples, n_genes)
            Source data to consider

        y: numpy.ndarray, shape(n_samples, 1), optional, default to None
            Response data (optional, just for compliance with BaseEstimator)

        Returned Values
        -------
        self: returns an instance of self.
        """

        # Add X to source data if use_data set to True
        if self.use_data:
            if self.source_data is None or self.source_data.shape[0] == 0:
                self.source_data = X
            else:
                self.source_data = np.concatenate([self.source_data, X])

        # Standardize data
        self.standard_scaler_input_.fit(X)
        self.source_data = self.standard_scaler_source_.fit_transform(self.source_data)
        self.target_data = self.standard_scaler_target_.fit_transform(self.target_data)

        # Compute principal vectors
        self.pv_computation.fit(self.source_data, self.target_data, y)

        # Compute intermediate features
        self.flow = self.intermediate_factors.sample_flow(
                self.pv_computation.source_components_, 
                self.pv_computation.target_components_
        )
        
        # Compute the consensus representation between each PV
        self._find_common_representation()

        return self

    def transform(self, X, y=None):
        """
        Project data along the geodesic path. 

        Attributes
        -------
        X: numpy.ndarray, shape (n_components, n_features)
            Genomics data use for prediction.

        Return values
        -------
        X_projected: numpy.ndarray, shape (n_components, n_representations)
            Genomics data projected on the consensus representation.
        """
        return self.standard_scaler_input_.fit_transform(X).dot(self.consensus_representation)