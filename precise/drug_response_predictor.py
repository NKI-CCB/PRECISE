"""Drug Response Predictor

@author: Soufiane Mourragui

This module centralizes the domain adaptation strategy towards biology-aware drug response
prediction on in-vivo dataset.

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
    [2] Gong, B., Shi, Y., Grauman, K. (2012) Geodesic Flow Kernel for unsupervised
    domain adaptation. IEEE CVPR
    [3] Goapalan, R., Li, R., Chellappa, R. (2011) Domain Adaptation for object
    recognition, an unsupervised approach. IEEE ICCV
"""

import os
import numpy as np 
import scipy
from pathlib import Path
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.neighbors import KNeighborsRegressor
from joblib import Parallel, delayed
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from copy import deepcopy
import tempfile
from joblib import load, dump


from precise.intermediate_factors import IntermediateFactors
from precise.principal_vectors import PVComputation
from precise.pipeline_routine import FlowProjector, GeodesicMatrixComputer, ConsensusRepresentation


class DrugResponsePredictor:
    """
    Main pipeline for training a tumor-aware drug response predictor. This class contains:
        - principal vectors computation,
        - consensus representation computation,
        - regression model training based on these representations,
        - computation of the predictive performance.

    On top of containing the solution selected by [1], it offers an implementation of the
    Geodesic Flow Sampling [2] and the Geodesic Flow Kernel [3] with the equivalent
    definition derived in supplementary material of [1].

    
    Attributes
    -------
    n_representations : int, default to 100
        Number of representations between source and target principal vectors for interpolation.

    method : str, default to 'consensus'
        Scheme used for the domain adaptation step, i.e. 'consensus', 'elasticnet', or 'gfk'.

    mean_center : bool, default to True
        Whether the different datasets used in the implementation should be mean centered.

    std_unit : bool, default to False 
        Whether the different datasets used in the implementation should be standardized
        (feature-level variance to 1).

    n_factors : int, default to 70 
        Number of domain-specific factors to compute, e.g. PCs.

    n_pv : int, default to 40
        Number of principal vectors to compute from the domain-specific factors.

    dim_reduction : str, default to 'pca' 
        Dimensionality reduction method for the source data,
        i.e. 'pca', 'ica', 'nmf', 'fa', 'sparsepca', pls'.

    dim_reduction_target : str, default to None 
        Dimensionality reduction method for the target data,
        i.e. 'pca', 'ica', 'nmf', 'fa', 'sparsepca', pls'. If None, set to dim_reduction.

    l1_ratio : float, default to 0
        l1 ratio for elasticnet model (0 is Ridge, 1 is Lasso).

    source_data : np.ndarray, default to None
        source data to use in domain adaptation phase.

    target_data : np.ndarray, default to None 
        target data to use in domain adaptation phase.

    n_jobs : int, default to 1
        number of jobs used in parallelisation.

    pv_computation : PVComputation
        Instance computing the principal vectors.

    intermediate_factors : IntermediateFactors
        Instance computing the interpolated features between source and target.

    predictor : BaseEstimator
        Regression model based on feature representation chosen in "method".

    alpha_values: np.ndarray
        Regression coefficients for grid search in regression model.

    cv_fold : int (set to 10)
        Number of cross validation folds used for finding the optimal shrinkage
        coefficient and computing the predictive performance.

    verbose : int (set to 1)
        Level of verbosity in joblib instances.
    """

    def __init__(self, n_representations=100, method='consensus',
                mean_center=True,
                std_unit=False,
                n_factors=70,
                n_pv=40,
                dim_reduction='pca',
                dim_reduction_target=None,
                l1_ratio=0,
                source_data=None,
                target_data=None,
                n_jobs=1):
        """
        Parameters
        -------
        n_representations : int, default to 100
            Number of representations between source and target principal vectors for interpolation.
            0 means source only, -1 means target only.

        method : str, default to 'consensus'
            Scheme used for the domain adaptation step, i.e. 'consensus', 'elasticnet', or 'gfk'.

        mean_center : bool, default to True
            Whether the different datasets used in the implementation should be mean centered.

        std_unit : bool, default to False 
            Whether the different datasets used in the implementation should be standardized
            (feature-level variance to 1).

        n_factors : int, default to 70 
            Number of domain-specific factors to compute, e.g. PCs.

        n_pv : int, default to 40
            Number of principal vectors to compute from the domain-specific factors.

        dim_reduction : str, default to 'pca' 
            Dimensionality reduction method for the source data,
            i.e. 'pca', 'ica', 'nmf', 'fa', 'sparsepca', pls'.

        dim_reduction_target : str, default to None 
            Dimensionality reduction method for the target data,
            i.e. 'pca', 'ica', 'nmf', 'fa', 'sparsepca', pls'. If None, set to dim_reduction.

        l1_ratio : float, default to 0
            l1 ratio for elasticnet model (0 is Ridge, 1 is Lasso).

        source_data : np.ndarray, default to None
            source data to use in domain adaptation phase.

        target_data : np.ndarray, default to None 
            target data to use in domain adaptation phase.

        n_jobs : int, default to 1
            number of jobs used in parallelisation.
        """

        self.n_representations = n_representations
        self.mean_center = mean_center
        self.std_unit = std_unit
        self.method = method
        self.n_factors = n_factors
        self.n_pv = n_pv
        self.l1_ratio = l1_ratio
        self.dim_reduction = dim_reduction
        self.dim_reduction_target = dim_reduction_target
        self.n_jobs = n_jobs

        self.source_data = source_data
        self.target_data = target_data

        self.pv_computation = PVComputation(
            self.n_factors,
            self.n_pv,
            self.dim_reduction,
            self.dim_reduction_target
        )

        self.intermediate_factors = IntermediateFactors(
            self.n_representations
        )

        self.predictor = None
        
        # Default values for CV
        self.alpha_values = np.logspace(-6,10,34)
        self.cv_fold = 10
        self.verbose = 1

    def fit(self, X_source, y_source, mean_center=False, std_unit=False, use_data=True):
        """
        Train the drug response predictor by first computing the feature representation corresponding
        to "method", then projecting on this representation, and finally training a regression model
        on the projected data.

        Parameters
        -------
        X_source : numpy.ndarray, shape (n_components, n_features)
            Genomics data use for prediction.

        y_source : numpy.ndarray, shape (n_components, 1)
            Drug response, i.e. output.

        mean_center : bool, default to False
            Whether X_source features (i.e. genes) should be mean-centered.

        std_unit : bool, default to False
            Whether X_source features (i.e. genes) should be standardized.

        use_data : bool, default to True
            Whether X_source should be also incorporated into the domain adaptation.
            If False, data from "source_data" will solely be used.

        Return values
        -------
        self: returns an instance of self.

        """
        # Sample along the geodesic flow and project on all intermediate features [3].
        if self.method.lower() == 'elasticnet':
            self._fit_elasticnet(X_source, y_source, use_data)

        # Infinite setting [2].
        elif self.method.lower() == 'gfk':
            self._fit_kernel_ridge(X_source, y_source, use_data)

        # Construct from the intermediate features a representation with comparable 
        # probability distribution [1].
        elif self.method.lower() == 'consensus':
            self._fit_consensus(X_source, y_source, use_data)

        else:
            raise NameError('Unknown method: %s, should be \'gfk\' or \'elasticnet\''%(self.method))

        return self

    def _memmap_array(self, x, name=None):
        name = name or ''.join(np.random.choice(list('QWERTYUIOPASDFGHJKLZXCVBNM'), 10))
        filename= os.path.join(tempfile.mkdtemp(), 'joblib_%s.mmap'%(name))

        fp = np.memmap(filename, dtype='float32', mode='w+', shape=x.shape)
        fp[:] = x[:]
        return fp

    def _memmap_data(self):
        # Source
        if self.source_data is not None and self.source_data.shape[0] > 0:
            self.source_data = self._memmap_array(self.source_data, 'source')
        self.target_data = self._memmap_array(self.target_data, 'target')

    def _fit_elasticnet(self, X_source, y_source, use_data=True):
        # Cross validate the alpha
        param_grid ={
            'regression__alpha': self.alpha_values,
        }

        # Put source and target data into memory for joblib
        if self.n_jobs >= 2:
            self._memmap_data()

        self.regression_model_ = GridSearchCV(
            Pipeline([
                ('projector', FlowProjector(
                        source_data=self.source_data,
                        target_data=self.target_data,
                        n_factors=self.n_factors,
                        n_pv=self.n_pv,
                        dim_reduction=self.dim_reduction,
                        dim_reduction_target=self.dim_reduction_target,
                        n_representations=self.n_representations,
                        use_data=use_data,
                        mean_center=self.mean_center,
                        std_unit=self.std_unit
                    )
                ),
                ('regression', ElasticNet(l1_ratio=self.l1_ratio) if self.l1_ratio != 0 else Ridge())
            ]),
            cv=self.cv_fold,
            n_jobs=self.n_jobs,
            pre_dispatch='1.2*n_jobs',
            param_grid=param_grid,
            verbose=self.verbose,
            scoring='neg_mean_squared_error'
        )
        self.regression_model_.fit(X_source, y_source)

        self.predictor = self.regression_model_.best_estimator_

    def _compute_kernel_matrix(self, X1, X2=None):
        X1_projected = X1.dot(self.intermediate_factors.projection)
        if X2 is None:
            X2_projected = X1_projected
        else:
            X2_projected = X2.dot(self.intermediate_factors.projection)

        return X1_projected.dot(self.G).dot(X2_projected.transpose())

    def _fit_kernel_ridge(self, X_source, y_source, use_data=True):
        # Cross validate the alpha
        param_grid ={
            'regression__alpha': self.alpha_values
        }

        # Put source and target data into memory for joblib
        if self.n_jobs >= 2:
            self._memmap_data()

        # Grid search setup        
        self.regression_model_ = GridSearchCV(
            Pipeline([
                ('projector', GeodesicMatrixComputer(
                        source_data=self.source_data,
                        target_data=self.target_data,
                        n_factors=self.n_factors,
                        n_pv=self.n_pv,
                        dim_reduction=self.dim_reduction,
                        dim_reduction_target=self.dim_reduction_target,
                        n_representations=self.n_representations,
                        use_data=use_data,
                        mean_center=self.mean_center,
                        std_unit=self.std_unit
                    )
                ),
                ('regression', KernelRidge(kernel='precomputed'))
            ]),
            cv=self.cv_fold,
            n_jobs=self.n_jobs,
            pre_dispatch='1.2*n_jobs',
            param_grid=param_grid,
            verbose=self.verbose,
            scoring='neg_mean_squared_error'
        )

        #Fit grid search, no need to remove intercept (sklearn handles it)
        self.regression_model_.fit(X_source, y_source)
        self.predictor = self.regression_model_.best_estimator_

    def _fit_consensus(self, X_source, y_source, use_data=True):
        # Cross validate the alpha
        param_grid ={
            'regression__alpha': self.alpha_values,
        }

        # Put source and target data into memory for joblib
        if self.n_jobs >= 2:
            self._memmap_data()

        self.regression_model_ = GridSearchCV(
            Pipeline([
                ('projector', ConsensusRepresentation(
                        source_data=self.source_data,
                        target_data=self.target_data,
                        n_factors=self.n_factors,
                        n_pv=self.n_pv,
                        dim_reduction=self.dim_reduction,
                        dim_reduction_target=self.dim_reduction_target,
                        n_representations=self.n_representations,
                        use_data=use_data,
                        mean_center=self.mean_center,
                        std_unit=self.std_unit
                    )
                ),
                ('scaler', StandardScaler(with_mean=True, with_std=True)),
                ('regression', ElasticNet(l1_ratio=self.l1_ratio) if self.l1_ratio != 0 else Ridge())
            ]),
            cv=self.cv_fold,
            n_jobs=self.n_jobs,
            pre_dispatch='1.2*n_jobs',
            param_grid=param_grid,
            verbose=self.verbose,
            scoring='neg_mean_squared_error'
        )
        self.regression_model_.fit(X_source, y_source)

        self.predictor = self.regression_model_.best_estimator_

    def predict(self, X_target):
        """
        Project the data on the feature representation corresponding to "method",
        and use the predictor trained in "fit" to predict the value of the samples.

        Attributes
        -------
        X_target : numpy.ndarray, shape (n_samples, n_features)
            Genomics data use for prediction.

        Return values
        -------
        y_target : numpy.ndarray, shape (n_samples, 1)
            Drug response predicted for target.
        """

        if self.predictor is None:
            raise ValueError("Instance not fitted")

        return self.predictor.predict(X_target)

    def _cv_predict(train, test, pred, X, y):
        pred.fit(X[train], y[train])
        return pred.predict(X[test]), test

    def compute_predictive_performance(self, X, y, use_data=True):
        """
        Compute the predictive performance of PRECISE by a nested double cross-validation.
        Predictive performance is computed as pearson correlation between the predicted
        response and the real response given in "y"
        
        Attributes
        -------
        X : numpy.ndarray, shape (n_samples, n_features)
            Genomics data use for prediction.
        
        y: numpy.ndarray, shape (n_samples, 1)
            Response data

        use_data: bool, default to True
            Whether data "X" should be incorporated into the domain adaptation step.

        Return values
        -------
        predictive_performance : float
            Predictive performance, between -1 and 1, 1 being the best possible.
        """

        # Copy regression model to avoid interference with the already fitted model.
        pred = deepcopy(self.regression_model_)
        pred.verbose = self.verbose

        # To remove any data aggregated during the learning phase.
        pred.estimator.named_steps['projector'].source_data = self.source_data
        # Restrict the grid search around the optimal shrinkage coefficient.
        alpha_opt = self.regression_model_.best_estimator_.named_steps['regression'].alpha
        pred.param_grid['regression__alpha'] = np.array([0.01, 0.1, 1, 10., 100.]) * alpha_opt

        # Use double loop cross validation.
        k_fold_split = GroupKFold(10)
        results = [DrugResponsePredictor._cv_predict(train, test, pred, X, y)\
                    for train, test in k_fold_split.split(X, y, y)]

        y_predicted = np.zeros(X.shape[0])
        for v, i in results:
            y_predicted[i] = v

        del pred, results

        # Compute predictive performance as pearson correlation.
        return scipy.stats.pearsonr(y_predicted, y)[0]