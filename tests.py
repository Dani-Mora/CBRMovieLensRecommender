
from recommender import MovieRecommender
from utils import MOVIELENS1M_DATA_FOLDER, check_download_data

# Download MovieLens data
check_download_data()


#  Testing data
test_user = 2000
test_N = 10

# Create and initialize recommender
train_ratio = 0.8
rec = MovieRecommender(path=MOVIELENS1M_DATA_FOLDER)
rec.initialize(train_ratio=train_ratio)

# Retriving similar users
similar_users = rec.retrieve(test_user)
neightbor_ids = []
for s_user in similar_users:
    neightbor_ids.append(s_user[0])

print "Result of the retrieve:"
print neightbor_ids

# Getting recommendation movies
#recommended_movies_after_reuse = rec.reuse(test_user, neightbor_ids, test_N)


#    """ CBR cycle step for the given user for the given movie and rating """
#    sim_users = self.retrieve(user_id, neighbors=self.neighbors) # Retrieves a set of user ids
#    sim_movies = self.reuse(user_id, neighbors=sim_users, N=self.movies_per_neighbors) # Retrieves a set of MovieInfo
#    feedback = self.review(user_id, ranked=sim_movies, movie_id=movie_id, rating=rating)
#    self.retain(user_id, feedback)

# TODO continue
