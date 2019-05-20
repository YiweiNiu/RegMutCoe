#!/usr/bin/env python

'''
build co-expression network

method: WGCNA
'''

import logging
import os
import sys
from copy import deepcopy

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

os.environ["OMP_NUM_THREADS"] = "1" # export OMP_NUM_THREADS=1
os.environ["OPENBLAS_NUM_THREADS"] = "1" # export OPENBLAS_NUM_THREADS=1
os.environ["MKL_NUM_THREADS"] = "1" # export MKL_NUM_THREADS=1
os.environ["VECLIB_MAXIMUM_THREADS"] = "1" # export VECLIB_MAXIMUM_THREADS=1
os.environ["NUMEXPR_NUM_THREADS"] = "1" # export NUMEXPR_NUM_THREADS=1


# logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def bicor(a=None, b=None):
    '''
    biweight midcorrelation
    from: https://github.com/pandas-dev/pandas/pull/9826

    @parameter a: 1D array
    @parameter b: 1D array

    @return: biweight midcorrelation
    '''
    a_median = a.median()
    b_median = b.median()

    # Median absolute deviation
    a_mad = (a - a_median).abs().median()
    b_mad = (b - b_median).abs().median()

    u = (a - a_median) / (9 * a_mad)
    v = (b - b_median) / (9 * b_mad)

    w_a = np.square(1 - np.square(u)) * ((1 - u.abs()) > 0)
    w_b = np.square(1 - np.square(v)) * ((1 - v.abs()) > 0)
    
    a_item = (a - a_median) * w_a
    b_item = (b - b_median) * w_b

    return (a_item * b_item).sum() / (
        np.sqrt(np.square(a_item).sum()) * np.sqrt(np.square(b_item).sum()))


def choose_beta(cor_mat=None, network_type='unsigned', nBreaks=10, RsquaredCut=0.8, eps=0.1):
    '''
    choose soft-threshold power to get scale-free topology

    @parameter df: a pandas dataframe of correlation matrix
    '''
    betas = [[], [], []]
    for beta in range(1, 21):
        if network_type == 'unsigned':
            adj_mat = cor_mat.abs().pow(beta)
        elif network_type == 'signed':
            adj_mat = cor_mat.mul(0.5).add(0.5).abs().pow(beta)
        else:
            logger.error('Unsupported network type %s.' %network_type)
            sys.exit(1)

        # generalized connectivity
        k = adj_mat.sum(axis=1) - 1; mean_k = np.mean(k)
        dk = (np.histogram(k, bins=nBreaks, weights=k)[0] /
                           np.histogram(k, bins=nBreaks)[0])

        pdk, _ = np.histogram(k, bins=nBreaks, density=True)

        # some bins get 0
        tmp = deepcopy(pdk)
        for ix, data in enumerate(tmp):
            if data == 0:
                dk = np.delete(dk, ix)
                pdk = np.delete(pdk, ix)

        # fit linear regression
        log10_dk = np.log10(dk).reshape((-1, 1))
        log10_pdk = np.log10(pdk)

        model = LinearRegression().fit(log10_dk, log10_pdk)
        r_sq = model.score(log10_dk, log10_pdk)
        slope = model.coef_[0]

        if slope > 0:
            sft_fit = -r_sq
        else:
            sft_fit = r_sq

        betas[0].append(beta)
        betas[1].append(sft_fit)    # 
        betas[2].append(mean_k) # mean connectivity

    print (betas[0])
    print (betas[1])
    print (betas[2])

    if max(betas[1]) < RsquaredCut:
        logger.warning('Cannot find beta between 1 and 20, beta=6 would be used.')
        return 6

    for ix, _ in enumerate(betas[0][:-2]):
        if betas[1][ix] >= RsquaredCut:
            d = max(abs(betas[1][ix]-betas[1][ix+1]),
                    abs(betas[1][ix]-betas[1][ix+2]),
                    abs(betas[1][ix+1]-betas[1][ix+2]))

            if d < eps:
                tmp_list = betas[2][ix:ix+3]
                j = tmp_list.index(max(tmp_list)) + ix

                logger.info('Beta selected is %s' %(betas[0][j]))
                return betas[0][j]


def read_ppi(ppi=None):
    '''
    read ppi network
    Two columns, tab or space separated
    line starting with '#' is considered as comment

    @parameter ppi: protein-protein interaction. 
    '''
    G = nx.Graph()
    with open(ppi, 'r') as fin:
        for line in fin:
            line = line.strip()

            if line.startswith('#'):
                continue

            gene1, gene2 = line.split()[:2]
            G.add_edge(gene1, gene2)

    return G


def build_coe(exp_mat=None, cor_meth="pearson", beta=None, sift_by_ppi=False,
                    ppi=None, network_type="unsigned"):
    '''
    Read the gene-sample expression matrix, in which each row is a gene, and each col is a sample.
    The matrix should be a tab-separated palin text file.
    Build co-expression network.

    @parameter exp_mat: gene-sample expression matrix
    @parameter cor_meth: method to calculate correlations
    @parameter beta: soft-threshold power
    @parameter sift_by_ppi: sift co-expression networks by ppi
    @parameter ppi: protein-protein interaction. Two columns, tab-separated.
    @parameter network_type: network type, signed or unsigned

    @return: a weighted graph
    '''
    df = pd.read_csv(exp_mat, sep='\t', index_col=0, header=0)

    # correlation
    if cor_meth == 'pearson':
        cor_mat = df.T.corr()
    elif cor_meth == 'kendall':
        cor_mat = df.T.corr(method='kendall')
    elif cor_meth == 'spearman':
        cor_mat = df.T.corr(method='spearman')
    elif cor_meth == 'bicor':
        cor_mat = df.T.corr(method=bicor)
    else:
        logger.error('Not supported correlation method %s.' %cor_meth)
        sys.exit(1)

    # choose beta, to enhance signal to noise ratio
    if beta is None:
        beta = choose_beta(cor_mat, network_type)
    else:
        logger.info('Using beta provided: %s' %(beta))

    # adjacency
    if network_type == 'signed':
        adj_mat = cor_mat.abs().pow(beta)
    elif network_type == 'unsigned':
        adj_mat = cor_mat.mul(0.5).add(0.5).abs().pow(beta)
    else:
        logger.error('Unsupported network type %s.' %network_type)
        sys.exit(1)

    # build graph
    mapping = {ix:gene for ix, gene in enumerate(adj_mat.columns)}
    coe_net = nx.from_numpy_array(np.asarray(adj_mat))
    coe_net = nx.relabel_nodes(coe_net, mapping, copy=False)

    # sift by ppi
    if sift_by_ppi:
        G = read_ppi(ppi)

        nodes_to_rm = set(G) - set(coe_net)
        G.remove_nodes_from(nodes_to_rm)

        for u,v in G.edges:
            G[u][v]['weight'] = coe_net[u][v]['weight']

        #for node in G:      # self-connected node
            #G.add_edge(node, node, 'weight'=1.0)

        return G
    else:
        return coe_net


def test():
    exp_mat = 'D:/Project/20190510_DriverNetPy/RegMutCoe/test/datExpr.txt'
    G = build_coe(exp_mat)

    print(G.number_of_nodes())
    print(G['MMT00000044']['MMT00000046']['weight'])
    print(G['MMT00000044']['MMT00000051']['weight'])


if __name__ == '__main__':
    try:
        test()
    except KeyboardInterrupt:
        logger.error("User interrupted me! ;-) Bye!")
        sys.exit(0)
