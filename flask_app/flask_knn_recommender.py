# all the imports
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from fuzzywuzzy import fuzz
import os
import time
import gc
import argparse
import pickle
import scipy.sparse
from scipy.sparse import csr_matrix

from flask import Flask, request, render_template
from flask_table import Table, Col

# class ItemTable(Table):
# 	rank = Col('Rank')
# 	game = Col('Game')

app = Flask(__name__) # create the application instance :)
#app.config.from_object(__name__) # load config from this file , flaskr.py

games = pd.read_csv("games_list.csv")
games_list = games.name.tolist()

@app.route('/', methods=["GET", "POST"])
def initial_page():
	if request.method=="POST":
		return render_template('recommendation2.html')
	return render_template('initial2.html', list=games_list)

@app.route('/recommendations', methods=["GET", "POST"])
def getvalues_and_recommend():
	if request.method=="POST":
		fav_game = request.form.get('fav_game')
		with open("game_to_idx.pkl", 'rb') as fp:
			mapper = pickle.load(fp)
		sparse_matrix = scipy.sparse.load_npz("sparse_matrix.npz")

		model_knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20, n_jobs=-2)

		def fuzzy_matching(mapper, fav_game, verbose=True):
			match_tuple = []
			for name, idx in mapper.items():
				ratio = fuzz.ratio(name.lower(), fav_game.lower())
				if ratio >= 60:
					match_tuple.append((name, idx, ratio))
			# sort
			match_tuple = sorted(match_tuple, key=lambda x: x[2])[::-1]
			return match_tuple[0][1]

		def make_recommendation(model_knn, data, mapper, fav_game, n_recommendations):
			model_knn.fit(data)
			# get input movie index
			idx = fuzzy_matching(mapper, fav_game, verbose=True)
			# inference
			distances, indices = model_knn.kneighbors(data[idx], n_neighbors=n_recommendations+1)
			# get list of raw idx of recommendations
			raw_recommends = \
				sorted(list(zip(indices.squeeze().tolist(), distances.squeeze().tolist())), key=lambda x: x[1])[:0:-1]
			# get reverse mapper
			
			reverse_mapper = {v: k for k, v in mapper.items()}
			ranks = np.arange(1, len(raw_recommends)+1, 1)
			recs = []
			for idx, dist in raw_recommends:
				recs.append(reverse_mapper[idx])
			res = {'Rank': ranks, 'Game' : recs}
			return res

		results = make_recommendation(model_knn=model_knn,
			data=sparse_matrix, mapper=mapper, fav_game=fav_game, n_recommendations=10)
		res_df = pd.DataFrame(results)
		return render_template('recommendation2.html', table=res_df.to_html()) 

if __name__ == '__main__':
	app.run(debug=True)
