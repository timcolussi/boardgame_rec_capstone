import os
import time
import gc
import argparse
import pickle

# data science imports
import pandas as pd
import scipy.sparse
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

# utils import
from fuzzywuzzy import fuzz


class KnnRecommender:
    """
    This is an item-based collaborative filtering recommender with
    KNN implmented by sklearn
    """
    def __init__(self, sparse_matrix_file, mapper_file):
        """
        Recommender requires path to data: movies data and ratings data
        Parameters
        ----------
        path_movies: str, movies data file path
        path_ratings: str, ratings data file path
        """
        self.sparse_matrix_file = sparse_matrix_file
        self.mapper_file = mapper_file
        self.model = NearestNeighbors()

    def set_model_params(self, n_neighbors, algorithm, metric, n_jobs=None):
        """
        set model params for sklearn.neighbors.NearestNeighbors
        Parameters
        ----------
        n_neighbors: int, optional (default = 5)
        algorithm: {'auto', 'ball_tree', 'kd_tree', 'brute'}, optional
        metric: string or callable, default 'minkowski', or one of
            ['cityblock', 'cosine', 'euclidean', 'l1', 'l2', 'manhattan']
        n_jobs: int or None, optional (default=None)
        """
        if n_jobs and (n_jobs > 1 or n_jobs == -1):
            os.environ['JOBLIB_TEMP_FOLDER'] = '/tmp'
        self.model.set_params(**{
            'n_neighbors': n_neighbors,
            'algorithm': algorithm,
            'metric': metric,
            'n_jobs': n_jobs})

    def _prep_data(self):     

        game_user_mat_sparse = scipy.sparse.load_npz(os.path.join(self.sparse_matrix_file))
        with open(os.path.join(self.mapper_file), 'rb') as fp:
            hashmap = pickle.load(fp)

        return game_user_mat_sparse, hashmap

    def _fuzzy_matching(self, hashmap, fav_movie):
        """
        return the closest match via fuzzy ratio.
        If no match found, return None
        Parameters
        ----------
        hashmap: dict, map movie title name to index of the movie in data
        fav_movie: str, name of user input movie
        Return
        ------
        index of the closest match
        """
        match_tuple = []
        # get match
        for name, idx in hashmap.items():
            ratio = fuzz.ratio(name.lower(), game_name.lower())
            if ratio >= 60:
                match_tuple.append((name, idx, ratio))
        # sort
        match_tuple = sorted(match_tuple, key=lambda x: x[2])[::-1]
        if not match_tuple:
            print('Oops! No match is found')
        else:
            print('Found possible matches in our database: '
                  '{0}\n'.format([x[0] for x in match_tuple]))
            return match_tuple[0][1]

    def _inference(self, model, data, hashmap,
                   game_name, n_recommendations):
        """
        return top n similar movie recommendations based on user's input movie
        Parameters
        ----------
        model: sklearn model, knn model
        data: movie-user matrix
        hashmap: dict, map movie title name to index of the movie in data
        fav_movie: str, name of user input movie
        n_recommendations: int, top n recommendations
        Return
        ------
        list of top n similar movie recommendations
        """
        # fit
        model.fit(data)
        # get input movie index
        print('You have input game:', game_name)
        idx = self._fuzzy_matching(hashmap, game_name)
        # inference
        print('Recommendation system start to make inference')
        print('......\n')
        t0 = time.time()
        distances, indices = model.kneighbors(
            data[idx],
            n_neighbors=n_recommendations+1)
        # get list of raw idx of recommendations
        raw_recommends = \
            sorted(
                list(
                    zip(
                        indices.squeeze().tolist(),
                        distances.squeeze().tolist()
                    )
                ),
                key=lambda x: x[1]
            )[:0:-1]
        print('It took my system {:.2f}s to make inference \n\
              '.format(time.time() - t0))
        # return recommendation (movieId, distance)
        return raw_recommends

    def make_recommendations(self, game_name, n_recommendations):
        """
        make top n movie recommendations
        Parameters
        ----------
        fav_movie: str, name of user input movie
        n_recommendations: int, top n recommendations
        """
        # get data
        game_user_mat_sparse, hashmap = self._prep_data()
        # get recommendations
        raw_recommends = self._inference(
            self.model, game_user_mat_sparse, hashmap,
            game_name, n_recommendations)
        # print results
        reverse_hashmap = {v: k for k, v in hashmap.items()}
        print('Recommendations for {}:'.format(game_name))
        for i, (idx, dist) in enumerate(raw_recommends):
            print('{0}: {1}'.format(i+1, reverse_hashmap[idx]))


def parse_args():
    parser = argparse.ArgumentParser(
        prog="Game Recommender",
        description="Run KNN Game Recommender")
    parser.add_argument('--path', nargs='?', default='C:/Users/timco/Documents/NYCDSA/capstone/boardgame_rec_capstone/',
                        help='input data path')
    parser.add_argument('--sparse_matrix', nargs='?', default='obj/sparse_matrix.npz',
                        help='provide movies filename')
    parser.add_argument('--idx_mapper_file', nargs='?', default='obj/game_to_idx.pkl',
                        help='provide ratings filename')
    parser.add_argument('--game_name', nargs='?', default='',
                        help='provide your favorite game name')
    parser.add_argument('--top_n', type=int, default=10,
                        help='top n game recommendations')
    return parser.parse_args()


if __name__ == '__main__':
    # get args
    args = parse_args()
    data_path = args.path
    sparse_matrix_filename = args.sparse_matrix
    mapper_filename = args.idx_mapper_file
    game_name = args.game_name
    top_n = args.top_n
    # initial recommender system
    recommender = KnnRecommender(
        os.path.join(data_path, sparse_matrix_filename),
        os.path.join(data_path, mapper_filename))
    # set params
    recommender.set_model_params(20, 'brute', 'cosine', -1)
    # make recommendations
    recommender.make_recommendations(game_name, top_n)