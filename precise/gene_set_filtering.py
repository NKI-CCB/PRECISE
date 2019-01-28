""" Gene Set Filtering


@author: Soufiane Mourragui

This module performs the gene set filtering, i.e. given a gene set and some data with the exact
same feature order, return the features' values corresponding to a particular gene set.
This class:
- Read a certain gene sets collection and stores it.
- For one given gene set, return the data filtered in the features.
- A ENSEMBL to Entrez feature lookup is assured.

Example
-------
    Examples are given in the vignettes.


Notes
-------
	Examples are given in the vignette
	
References
-------
	[1] Golub, G.H. and Van Loan, C.F., 2012. "Matrix computations" (Vol. 3). JHU Press.
	[2] Gong, B., Shi, Y., Sha, F. and Grauman, K., 2012, June. "Geodesic flow kernel for 
	unsupervised domain adaptation". In Computer Vision and Pattern Recognition (CVPR), 
	2012 IEEE Conference on (pp. 2066-2073). IEEE.
	[3] Mourragui, S., Loog, M., Reinders, M.J.T., Wessels, L.F.A., "TO CHANGE"
"""

import os
import numpy as np
import pandas as pd
import scipy
from pathlib import Path
from copy import copy


import precise


global lookup_genes_file
lookup_genes_file = os.path.abspath('/'.join(precise.__file__.split('/')[:-2]) + '/lookup_data/gene_status.csv')

class GeneSetFiltering:
    """
    Attributes
    -------
    
    """

    def __init__(self, data,
                data_features_names,
                gene_set_name_file,
                data_feature_naming,
                gene_set_naming):
        """
    Parameters
    -------
    - data : list of numpy.ndarray
        List of data with same feature structure, i.e. same number and with the same order.
    - data_features_names : list
        Features names. It should be equivalent to the order in each element of data.
    - gene_set_name_file: str
		File where the gene set is located.
	- data_feature_naming: str
		Data feature convention, e.g. Ensembl, Entrez.
	- gene_set_naming: str
		Gene set feature convention, e.g. Ensembl, Entrez.
    """
    
        self._data = data

        self.data_features_names = data_features_names
        self.gene_set_name_file = gene_set_name_file
        self.data_feature_naming = data_feature_naming.lower()
        self.gene_set_naming = gene_set_naming.lower()

        self._read_gene_set_data()
        self._harmonize_feature_naming()


    def _read_gene_set_data(self):
        """
        Read gene set file and transforms it into a dictionary.
        """
        with open(self.gene_set_name_file, 'r') as file:
            gs_data = file.read()

        gs_data = gs_data.split('\n')
        gs_data = [e.split('\t') for e in gs_data]
        self.gene_sets = {e[0]:np.array(e[2:]).astype(int) for e in gs_data}


    def _harmonize_feature_naming(self):
        """
        Harmonizes the feature in such as a way that the gene sets have the same naming convention
        than the genes. This way the two can be easily compared.
        """
        
        if self.data_feature_naming == self.gene_set_naming:
            print('Already same feature naming')

        print(lookup_genes_file)
        gene_name_lookup = pd.read_csv(lookup_genes_file, index_col=0)

        for gs_name, gs in copy(self.gene_sets).items():
            gs = pd.DataFrame(gs, columns=[self.gene_set_naming])
            gs = gs.merge(gene_name_lookup[[self.data_feature_naming, self.gene_set_naming]],
                            on=self.gene_set_naming,
                            how='left')
            # Remove null
            gs = gs[gs[self.data_feature_naming].notnull()]
            self.gene_sets[gs_name] = gs[self.data_feature_naming].values.astype(str)

    def filter_gene_sets(self, min_number=0, max_number=np.inf):
        """
        Filters out gene sets that have less than min_number gene or more than max_number.

        Parameters
        -------
        - min_number : int
            Minimum number of genes a gene set should have to be considered
        - max_number : int
            Maximum number of genes a gene set should have to be considered
        """

        self.gene_sets = {
            name:gs for name,gs in self.gene_sets.items()\
            if len(gs) >= min_number and len(gs) <= max_number
        }


    def data_gene_set(self, gene_set_name):
        """
        Return the data in the same order as data with only features contained in the given gene set.

        Parameters
        -------
        - gene_set_name : str
            Name of the gene set under consideration.
        Output
        -------
        List of numpy.ndarray in same order as data with only gene set features.
        """
        genes = self.gene_sets[gene_set_name]
        genes_position = np.isin(self.data_features_names, genes)

        return [d[:, genes_position] for d in self.data]


    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, data):
        # Data checker
        if type(data) != list:
            raise ValueError('GeneSetFiltering.data should be a list, not a %s'%(type(data)))
        elif type(data[0]) != np.ndarray:
            raise ValueError('GeneSetFiltering.data elements should be a np.ndarray, not %s'%(type(data[0])))
        elif np.unique([type(e) for e in data]).shape[0] > 1:
            raise ValueError('GeneSetFiltering.data elements not in the same format')
        elif np.unique([d.shape[1] for d in data]).shape[0] > 1:
            raise ValueError('GeneSetFiltering.data elements should all have same 1-axis size')

        self._data = data

    def append_data(self, X):
        if X.shape[1] != np.unique([d.shape[1] for d in self._data])[0]:
            raise ValueError('New data should be same size of the data. %s != %s'%(X.shape[1],
                                                                                np.unique([d.shape[1] for d in self._data])[0]))

        self._data.append(X)
