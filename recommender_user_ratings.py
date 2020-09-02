import os
import time
import gc
import argparse

# data science imports
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

# utils import
from fuzzywuzzy import fuzz


class KnnRecommender:
    """
    This is an item-based collaborative filtering recommender with
    KNN implmented by sklearn
    """
    def __init__(self, path_games, path_ratings):
        """
        Recommender requires path to data: movies data and ratings data
        Parameters
        ----------
        path_movies: str, movies data file path
        path_ratings: str, ratings data file path
        """
        self.path_games = path_games
        self.path_ratings = path_ratings
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
        """
        prepare data for recommender
        1. movie-user scipy sparse matrix
        2. hashmap of movie to row index in movie-user scipy sparse matrix
        """
        # read data
        df_games = pd.read_csv(
            os.path.join(self.path_games),
            usecols=['id', 'name'],
            dtype={'id': 'int32', 'name': 'str'})
        df_ratings = pd.read_csv(
            os.path.join(self.path_ratings),
            usecols=['username', 'gameid', 'rating'],
            dtype={'username': 'str', 'gameid': 'int32', 'rating': 'float32'})
        # filter data


        user_nrates = pd.DataFrame(df_ratings.username.value_counts())
        low_raters = user_nrates.loc[user_nrates['username'] < 10].index.tolist()
        user_ratings_subset = df_ratings[~df_ratings.username.isin(low_raters)]
        user_ratings_subset.drop_duplicates(inplace=True)
        multis = user_ratings_subset.loc[user_ratings_subset.duplicated(subset=["gameid", "username"], keep=False)]
        df_no_multis = user_ratings_subset.loc[~user_ratings_subset.duplicated(subset=["gameid", "username"], keep=False)]

        # pivot and create movie-user matrix
        game_user_mat = df_no_multis.pivot(
            index='gameid', columns='username', values='rating').fillna(0)
        # create mapper from movie title to index
        hashmap = {
            game: i for i, game in
            enumerate(list(df_games.set_index('id').loc[game_user_mat.index].name)) # noqa
        }
        # transform matrix to scipy sparse matrix
        game_user_mat_sparse = csr_matrix(game_user_mat.values)

        # clean up
        del df_games, user_nrates, low_raters
        del user_ratings_subset, multis, df_no_mulits, game_user_mat
        gc.collect()
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
                   fav_movie, n_recommendations):
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
            print('{0}: {1}, with distance '
                  'of {2}'.format(i+1, reverse_hashmap[idx], dist))


def parse_args():
    parser = argparse.ArgumentParser(
        prog="Game Recommender",
        description="Run KNN Game Recommender")
    parser.add_argument('--path', nargs='?', default='C:/Users/timco/Documents/NYCDSA/capstone/boardgame_rec_capstone/',
                        help='input data path')
    parser.add_argument('--games_filename', nargs='?', default='games_list.csv',
                        help='provide movies filename')
    parser.add_argument('--ratings_filename', nargs='?', default='user_ratings.csv',
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
    games_filename = args.games_filename
    ratings_filename = args.ratings_filename
    game_name = args.game_name
    top_n = args.top_n
    # initial recommender system
    recommender = KnnRecommender(
        os.path.join(data_path, games_filename),
        os.path.join(data_path, ratings_filename))
    # set params
    recommender.set_model_params(20, 'brute', 'cosine', -1)
    # make recommendations
    recommender.make_recommendations(game_name, top_n)