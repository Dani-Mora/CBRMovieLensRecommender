import pandas as pd
import numpy as np

from utils import *
from movies import *
from users import *

logger = initialize_logging('cnr')


class MovieRecommenderInterface(object):

    """ Interface for Movie Recommender """

    def retrieve(self, user_id, N):
        """ CBR retrieve cycle that retrieves N similar users for the given user
        Args:
             user_id: Identifier of the query user
             N: Number of users ids to retrieve
        Returns:
            users: List of N identifiers of similar users
        """


    def reuse(self, user_id, neighbors, N):
        """ CBR reuse cycle that returns suitable movies to recommend to query user
        Args:
            user_id: Identifier of the query user
            neighbors: Identifier of the neighboring users
            N: Maximum number of movies to recommend for each neighbor
        Returns:
            movies: List of recommended movies for the user (CandidateInfo class)
        """


    def review(self, user_id, ranked, movie_id, rating):
        """ CBR review cycle that evaluates the recommended movies
        Args:
            user_id: Identifier of the query user
            ranked: Ranked list of recommendations for the user
            movie_id: Identifier of the movie
            rating: Rating the user has given to a movie
        Returns:
            feedback: Recommendation feedback (FeedbackInfo class)
        """


    def retain(self, user_id, feedback):
        """ CBR retain cycle that retains the evaluation cases into the case base
        Args:
           user_id: Identifier of the query user
           feedback: Feedback received from the review phase about the recommendations
        """


class MovieRecommender(MovieRecommenderInterface):

    # TODO: we could integrate the model into sklearn, though this is not a traditional model

    def __init__(self, path, neighbors=10, movies_per_neighbor=15, initial_affinity=0.5, update_rate=0.1):
        """ Constructor of the class
        Args:
            path: Path to MovieLens 1M dataset """
        self._read_data(path)
        self.inverted_movies = None
        self.neighbors = neighbors
        self.movies_per_neighbors = movies_per_neighbor
        self.user_affinity = AffinityCaseBase(initial_preference=initial_affinity,
                                              modifier=update_rate) # User recommendation feedback
        self.genre_willigness = AffinityCaseBase(initial_preference=initial_affinity,
                                                 modifier=update_rate) # Recommendations feedback
        self.genre_affinity = AffinityCaseBase() # User likes
        self.inverted_file = {} # Store movie - user who rated it
        self.test_indexes = {} # Stores the movie identifiers for each user


    def _read_data(self, path):
        """ Reads Movielens 1M data into the class """

        # TODO: do not know if we have to keep this objects forever or whether creating them
        # are memory intensive. We'll fix this later
        # TODO: also need to check whether we have to close it

        self.ratings = pd.read_csv('../ml-1m/ratings.dat',
                                   sep="::",
                                   names=['user_id', 'movie_id', 'rating', 'timestamp'])
        # Each user has at least 20 ratings. No missing data
        self.users = pd.read_csv('../ml-1m/users.dat',
                                 sep="::",
                                 names=['user_id', 'sex', 'age_group', 'occupation', 'zip_code'])

        self.movies = pd.read_csv('../ml-1m/movies.dat',
                                  sep="::",
                                  names=['movie_id', 'name', 'genre'])
        # Read possible genres
        self.genres = self._get_genres()
        # Build dummy column for each genre
        self._genres_to_dummy()


    def _genres_to_dummy(self):
        """ Converts movie genres into dummy columns (binary columns) """

        def build_column(data, name):
            """ Builds the input column taking into account the genes list """
            return data['genre'].apply(lambda l: name in l)

        # Create column for each genre
        for g in self.genres:
            self.movies[g] = build_column(self.movies, g)
        # Delete original one
        self.movies = self.movies.drop('genre', 1)


    @staticmethod
    def separate_genre(x):
        return x.split('|')


    def _get_genres(self):
        """ Returns unique genres in the system """
        separated = self.movies['genre'].apply(self.separate_genre)
        return {g: True for x in separated for g in x}


    def _get_users_list(self):
        """ Returns the list of user identifiers """
        return self.users['user_id'].tolist()


    def initialize(self, train_ratio=0.8):
        """ Initialize case base and fill with random training data """
        # Split between training and test (TODO)

        logger.info("Initializing CBR case base ...")

        ## Index user genres
        users = self._get_users()
        for u_id in users:
            self._set_user_preferences(u_id)
            # TODO

        # Index movies - user
        # TODO


    def get_user_movies(self, user_id):
        """ Returns the movies a user has rated """
        return self.ratings[self.ratings['user_id'] == user_id]['movie_id'].tolist()

    def _process_example(self, user_id, movie_id, rating):
        """ CBR cycle step for the given user for the given movie and rating """
        sim_users = self.retrieve(user_id, neighbors=self.neighbors) # Retrieves a set of user ids
        sim_movies = self.reuse(user_id, neighbors=sim_users, N=self.movies_per_neighbors) # Retrieves a set of MovieInfo
        feedback = self.review(user_id, ranked=sim_movies, movie_id=movie_id, rating=rating)
        self.retain(user_id, feedback)


    def retrieve(self, user_id, neighbors=10):
        """ See base class """
        return NotImplementedError("To be implemented")


    def reuse(self, user_id, neighbors, N):
        """ See base class """
        return NotImplementedError("To be implemented")


    def review(self, user_id, ranked, movie_id, rating):
        """ See base class """
        return NotImplementedError("To be implemented")


    def retain(self, user_id, feedback):
        """ See base class """
        return NotImplementedError("To be implemented")
