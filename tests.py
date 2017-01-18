
from recommender import MovieRecommender
from utils import MOVIELENS1M_DATA_FOLDER,check_download_data

# Download MovieLens data
check_download_data()

# Create and initialize recommender
train_ratio = 0.8
rec = MovieRecommender(path=MOVIELENS1M_DATA_FOLDER)
rec.initialize(train_ratio=train_ratio)

# TODO continue