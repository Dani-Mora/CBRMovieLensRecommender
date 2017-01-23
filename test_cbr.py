from recommender import MovieRecommender
from utils import MOVIELENS1M_DATA_FOLDER, check_download_data, save_list

# Download MovieLens data
check_download_data()

# Starting point of the CBR Recommender
# Create and initialize recommender
rec = MovieRecommender(path=MOVIELENS1M_DATA_FOLDER)
scores, sims = rec.test_cbr(test_size=10)

save_list(scores, 'scores.dat')
save_list(sims, 'sims.dat')

