"""

Script with helper functions to load data.

Running the file from bash (as follows) plots the various
latent datasets available::

    $ python data.py

"""

import torch
import numpy as np
from torch import tensor as tt
import sklearn.datasets as skd
#import matplotlib.pyplot as plt
#import pyro.distributions as dist
from utils.data_utils import data_path, resource, dependency

def float_tensor(X): return torch.tensor(X).float()

def _init_pca(Y, latent_dim):
    U, S, V = torch.pca_lowrank(Y, q = latent_dim)
    return torch.nn.Parameter(torch.matmul(Y, V[:,:latent_dim]))

def raw_movies_data():

    import pandas as pd

    data = pd.read_csv('data/ml-100k/u.data', sep='\t', names=[
                       'userId', 'itemId', 'rating', 'timestamp'])

    movies = pd.read_csv('data/ml-100k/u.item', sep='|', names=[
                'itemId', 'title', 'release_date', 'video_release_date',
                'IMDb_URL', 'unknown', 'Action', 'Adventure', 'Animation',
                'Childrens', 'Comedy', 'Crime', 'Documentary', 'Drama',
                'Fantasy', 'Film_Noir', 'Horror', 'Musical', 'Mystery',
                'Romance', 'Sci_Fi', 'Thriller', 'War', 'Western'],
        encoding='latin')

    return data, movies


def load_real_data(dataset_name):
    '''Loads read world datasets.

    Parameters
    ----------
    dataset_name : str
        One of 'iris' (4D), 'oilflow' (12D), 'gene' (48D), 'mnist' (784D),
        'brendan_faces' (560D), 'movie_lens' (1682D)

    Returns
    -------
    n : int
        Number of datapoints.
    d : int
        Number of dataset dimensions.
    q : None
        Number of latent dimensions (undefined).
    X : None
        Latent data (undefined).
    Y : torch.tensor
        Dataset. Return shape is (n x d).
    labels : numpy.array
        Data categories/classes.

    '''

    if dataset_name == 'iris':
        iris_data = skd.load_iris()
        Y = float_tensor(iris_data.data)
        labels = iris_data.target

    elif dataset_name == 'oilflow':
        Y = float_tensor(np.loadtxt('data/oil_data.txt'))
        labels = np.loadtxt('data/oil_labels.txt')
        labels = (labels @ np.diag([1, 2, 3])).sum(axis=1)

    elif dataset_name == 'gene':
        import pandas as pd

        URL = 'https://raw.githubusercontent.com/sods/ods/master/' +\
            'datasets/guo_qpcr.csv'
        gene_data = pd.read_csv(URL, index_col=0)
        Y = float_tensor(gene_data.values)
        raw_labels = np.array(gene_data.index)

        d = dict()
        i = 0
        for label in raw_labels:
            if label not in d:
                d[label] = i
                i += 1
        labels = [d[x] for x in raw_labels]

    elif dataset_name == 'mnist':
        from tensorflow.keras.datasets.mnist import load_data

        (y_train, train_labels), (y_test, test_labels) = load_data()
        labels = np.hstack([train_labels, test_labels])
        n = len(labels)
        Y = np.vstack([y_train, y_test])
        Y = float_tensor(Y.reshape(n, -1))

    elif dataset_name == 'brendan_faces':

        import pods
        Y = float_tensor(pods.datasets.brendan_faces()['Y'])
        labels = None

    elif dataset_name == 'movie_lens_100k':

        # movies = movies.loc[
        #    (movies.Sci_Fi == 1) | (movies.Romance == 1), 'itemId'].tolist()
        # data = data.loc[data.itemId.isin(movies)]

        data, movies = raw_movies_data()

        Y = data.pivot_table(index='userId', columns='itemId',
                             values='rating')

        labels = None
        Y = float_tensor(np.array(Y))
        
    elif dataset_name == 'movie_lens_1m':
        
        import pandas as pd
        
        def _fetch(url, folder):
          resource(target=data_path(folder,  folder + '.zip'),
                   url=url)
          dependency(target=data_path(folder, 'ml'),
                     source=data_path(folder,  folder + '.zip'),
                     commands=['unzip ' + folder + '.zip' + ' -d ' + data_path(folder,'')])
        
        folder = 'movie_lens_1m'
        url =  'http://files.grouplens.org/datasets/movielens/ml-1m.zip'
        _fetch(url, folder)
        data = pd.read_csv(data_path(folder) + '/ml-1m/ratings.dat', sep='::', names=[
                           'userId', 'itemId', 'rating', 'timestamp'])

        # movies = pd.read_csv(data_path(folder) + '/ml-1m/movies.dat', sep='::', names=[
        #             'itemId', 'title', 'genre'],
        #     encoding='latin')
        # movies = pd.merge(movies, movies['genre'].str.split('|', expand=True), left_index=True, right_index=True)
        # return data, movies
        Y = data.pivot_table(index='userId', columns='itemId',
                             values='rating')
        Y = float_tensor(np.array(Y))

        labels = None

    else:
        raise NotImplementedError(str(dataset_name) + ' data not implemented')

    n = len(Y)
    d = len(Y.T)
    q = X = None
    return n, d, q, X, Y, labels


def genre_movie_lens():

    import pandas as pd

    data, movies = raw_movies_data()

    vars_of_interest = ['userId', 'age', 'gender', 'occupation', 'zipCode']
    labels = pd.read_csv('data/ml-100k/u.user', sep='|',
                         names=vars_of_interest)

    vars_of_interest = ['itemId', 'title', 'release_date',
                        'video_release_date', 'IMDb_URL']

    movies_long = pd.melt(movies, id_vars=vars_of_interest, var_name='genre')
    movies_long = movies_long.loc[movies_long.value == 1]
    movies_long = pd.merge(
        movies_long[['itemId', 'genre']], data,
        left_on='itemId', right_on='itemId', how='outer')

    final = movies_long.groupby(['genre', 'userId'])
    final = final.aggregate({'rating': 'mean'}).reset_index()
    final = final.pivot_table(index='userId', columns='genre')
    Y = float_tensor(np.array(final))

    n = len(Y)
    d = len(Y.T)
    q = X = None
    return n, d, q, X, Y, labels


def _load_2d_synthetic_latent(latent_data_shape, n_samples=1000):
    '''Creates synthetic latent variables.

    Parameters
    ----------
    latent_data_shape : str
        One of 'blobs', 'noisy_circles', 'make_moons', 'varied', 'normal'.
    n_samples : int
        Number of data points.

    Returns
    -------
    X : numpy.array
        Latent data (shape nx2).
    labels : numpy.array
        Latent data categories/classes.

    '''

    if latent_data_shape == 'blobs':
        return skd.make_blobs(n_samples=n_samples, random_state=42)

    elif latent_data_shape == 'noisy_circles':
        return skd.make_circles(n_samples=n_samples, factor=.5,
                                noise=.05, random_state=42)

    elif latent_data_shape == 'make_moons':
        return skd.make_moons(n_samples=n_samples, noise=.05, random_state=42)

    elif latent_data_shape == 'varied':
        return skd.make_blobs(n_samples=n_samples, random_state=42,
                              cluster_std=[1.0, 2.5, 0.5])

    elif latent_data_shape == 'normal':
        return np.random.normal(size=(n_samples, 2)), \
               np.random.choice([1, 2], n_samples)

    else:
        raise NotImplementedError(str(latent_data_shape) + ' not recognized.')


def _potential_three(z):
    '''Potential 3 from pymc docs (See ref in `_load_2d_weird_latent`).'''
    z = z.T
    w1 = torch.sin(2.*np.pi*z[0]/4.)
    w2 = 3.*torch.exp(-.5*(((z[0]-1.)/.6))**2)
    p = torch.exp(-.5*((z[1]-w1)/.35)**2)
    p = p + torch.exp(-.5*((z[1]-w1+w2)/.35)**2) + 1e-30
    p = -torch.log(p) + 0.1*torch.abs_(z[0])
    return p


def _load_2d_weird_latent(n=300):
    '''Samples the latent variable from pymc's potential three.

    Parameters
    ----------
    n_samples : int
        Number of data points.

    Returns
    -------
    X : numpy.array
        Latent data (shape nx2).

    Notes
    -----
    From [1]_.

    .. [1] https://docs.pymc.io/notebooks/normalizing_flows_overview.html
    '''

    np.random.seed(42)
    Z = np.linspace(-5, 5, 500)
    Z = np.vstack([np.repeat(Z, 500), np.tile(Z, 500)]).T
    p = torch.exp(-_potential_three(tt(Z))).numpy()
    p /= p.sum()

    choice_idx = range(len(p))
    sample_idx = np.random.choice(choice_idx, n, True, p)
    X = tt(Z[sample_idx, :].copy()).float()
    return X


def generate_synthetic_data(n=300, x_type=None, y_type='hi_dim'):
    '''Creates synthetic data set.

    Parameters
    ----------
    n : int
        Number of data points.
    x_type : None or str
        One of 'blobs', 'noisy_circles', 'make_moons', 'varied', 'normal'.
        If None, potential three is used.
    y_type : str
        One of 'lo_dim' (2 planes), 'hi_dim' (6 non-linear functions), or
        'by_cat' where each label corresponds to a different set of functions.
        'by_cat' will only accept the latent type 'normal'.

    Returns
    -------
    n : int
        Number of datapoints.
    d : int
        Number of dataset dimensions.
    q : None
        Number of latent dimensions (undefined).
    X : None
        Latent data (undefined).
    Y : torch.tensor
        Dataset. Return shape is (n x d).
    labels : numpy.array
        Data categories/classes.

    '''

    def err(): return np.random.normal(size=n)*0.05

    if x_type is None:
        X = _load_2d_weird_latent(n)
        labels = None
    else:
        X, labels = _load_2d_synthetic_latent(x_type, n)

    if y_type == 'hi_dim':
        # sample from gp
        Y = float_tensor(np.vstack([
            0.1 * (X[:, 0] + X[:, 1])**2 - 3.5 + err(),
            0.01 * (X[:, 0] + X[:, 1])**3 + err(),
            2 * np.sin(0.5*(X[:, 0] + X[:, 1])) + err(),
            2 * np.cos(0.5*(X[:, 0] + X[:, 1])) + err(),
            4 - 0.1*(X[:, 0] + X[:, 1])**2 + err(),
            1 - 0.01*(X[:, 0] + X[:, 1])**3 + err(),
        ]).T)

    elif y_type == 'by_cat':
        assert x_type == 'normal'

        Y_1 = float_tensor(np.vstack([
            0.1 * (X[:, 0] + X[:, 1])**2 - 3.5 + err(),
            0.01 * (X[:, 0] + X[:, 1])**3 + err(),
            2 * np.sin(0.5*(X[:, 0] + X[:, 1])) + err()
        ]).T)

        Y_2 = float_tensor(np.vstack([
            2 * np.cos(0.5*(X[:, 0] + X[:, 1])) + err(),
            4 - 0.1*(X[:, 0] + X[:, 1])**2 + err(),
            1 - 0.01*(X[:, 0] + X[:, 1])**3 + err()
        ]).T)

        Y = Y_1.clone()
        Y[labels == 2, :] = Y_2[labels == 2, :]

    elif y_type == 'lo_dim':
        Y = 0.1*float_tensor(np.vstack([
                2*X.sum(axis=1)-1 + X.std()*10*err(),
                5*X.sum(axis=1)-3 + X.std()*10*err()]).T)

    d = len(Y.T)
    q = 2
    return n, d, q, X, Y, labels

