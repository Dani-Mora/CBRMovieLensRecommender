import pandas as pd

from utils import *
from users import *
from tqdm import tqdm


logger = initialize_logging('cbr')


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

    def __init__(self, path, neighbors=10, movies_per_neighbor=15, initial_affinity=0.5,
                 correlation_weight=0.75, min_movies_candidate=0.33, update_rate=0.1):
        """ Constructor of the class
        Args:
            path: Path to MovieLens 1M dataset.
            neighbors: Number of neighbors to retrieve in the CBR cycle.
            movies_per_neighbor: Number of movies per neighbor to extract
            initial_affinity: Initial affinity for users.
            correlation_weight: Weight in interval [0, 1] for the rating correlation
                in user similarity computation.
            min_movies_candidate: Ratio of the mean ratings per user that two users need two share
                at least to be considered neighbors. By default one third of the movies must be shared.
            update_rate: Pace at which we update the affinity of the users and the genre in recommendations.
        """
        self._read_data(path)
        self.inverted_movies = None
        self.neighbors = neighbors
        self.movies_per_neighbors = movies_per_neighbor
        self.correlation_weight = correlation_weight
        self.user_affinity = AffinityCaseBase(initial_preference=initial_affinity,
                                              modifier=update_rate) # User recommendation feedback
        self.genre_willigness = AffinityCaseBase(initial_preference=initial_affinity,
                                                 modifier=update_rate) # Recommendations feedback
        self.inverted_file = {} # Store movie - user who rated it
        self.train_indexes = {} # Stores training movie identifiers for each user
        self.test_indexes = {} # Stores the movie identifiers for each user

        check_ratio('Initial affinity', initial_affinity)
        check_ratio('Correlation weight', correlation_weight)
        check_ratio('Uodate rate', update_rate)
        check_ratio('Ratings ratio', min_movies_candidate)

        self.min_movies_candidate = min_movies_candidate * (self.ratings.shape[0] / self.users.shape[0])
        logger.info(self.min_movies_candidate)


    def _read_data(self, path):
        """ Reads Movielens 1M data into the class """

        # TODO: do not know if we have to keep this objects forever or whether creating them
        # are memory intensive. We'll fix this later
        # TODO: also need to check whether we have to close it

        self.ratings = pd.read_csv(os.path.join(path, 'ratings.dat'),
                                   sep="::",
                                   names=['user_id', 'movie_id', 'rating', 'timestamp'])
        # Each user has at least 20 ratings. No missing data
        self.users = pd.read_csv(os.path.join(path, 'users.dat'),
                                 sep="::",
                                 names=['user_id', 'sex', 'age_group', 'occupation', 'zip_code'])

        self.movies = pd.read_csv(os.path.join(path, 'movies.dat'),
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
        return {g: True for x in separated for g in x}.keys()


    def _get_users_list(self):
        """ Returns the list of user identifiers """
        return self.users['user_id'].tolist()


    def initialize(self, train_ratio=0.8):
        """ Initialize case base and fills inverted file index
        Complexity:
            O(u), where u = number of users,
            Considering that ratings_user << number of users
        """
        logger.info("Initializing CBR case base ...")

        # Iterate over all users
        users = self._get_users_list()
        for u_id in tqdm(users, desc="Initializing user information"):

            # Split between training and test movies
            movies = self._get_user_movies(u_id)
            train_ind, test_inds = split_data(len(movies), train_ratio)
            self.train_indexes[u_id] = movies[train_ind]
            self.test_indexes[u_id] = movies[test_inds]

            # Fill inverted file index with user movies
            movies = self._get_user_movies(u_id, list=True)
            for m_id in movies:
                self.add_user_rating(u_id, m_id)


    def add_user_rating(self, user_id, movie_id):
        """ Adds a user rating to the inverted file indexed structure """
        if not movie_id in self.inverted_file:
            self.inverted_file[movie_id] = []
        self.inverted_file[movie_id].append(user_id)


    def _find_users_by_movies(self, movie_id):
        """ Returns the user identifiers related to users who saw the input movie """
        return self.inverted_file[movie_id]


    def _get_user_preferences(self, user_id, input_movies):
        """ Sets the user genre preferences for the input set of movies of the user
        Args:
            user_id: Identifier of the user
            movies: Identifier of the movies
        """
        # User training ratings
        user_ratings = self.ratings[(self.ratings['user_id'] == user_id)
                               & (self.ratings['movie_id'].isin(input_movies))]

        # Get rating-movie information
        movies_user = pd.merge(user_ratings, self.movies, on='movie_id')

        # Get count of genres
        genres_sum = movies_user[self.genres].sum()
        genres_sum_mat = genres_sum.as_matrix()

        # Weight by average of genre within user
        mean_ratings = np.zeros(len(self.genres))
        for i, g in enumerate(genres_sum.index):
            mean_ratings[i] = movies_user[movies_user[g] == True]['rating'].mean()

        # Multiply and replace nans to 0
        return np.nan_to_num(genres_sum_mat * mean_ratings, 0)


    def _get_user_movies(self, user_id, list=False):
        """ Returns the movies a user has rated
        Args:
            list: Whether to return a Matrix (False) or a list (True) """
        content = self.ratings[self.ratings['user_id'] == user_id]['movie_id']
        return content.tolist() if list else content.as_matrix()


    def _get_user_ratings(self, user_id):
        return self.ratings[self.ratings['user_id'] == user_id]


    def get_shared_ratings(self, user1_id, user2_id):
        """ Returns the number of rated movies shared by the two users """
        return pd.merge(self._get_user_ratings(user1_id),
                        self._get_user_ratings(user2_id),
                        on=['movie_id'])


    def _get_user_correlation(self, user1_id, user2_id):
        """ Returns the balance of differences between ratings of rated films by both user.
        Takes only into account those films rated by both """
        shared_ratings = self.get_shared_ratings(user1_id, user2_id)
        return (shared_ratings['rating_x'] - shared_ratings['rating_y']).abs().mean()


    def _get_user_similarity(self, user1_id, user2_id):
        """ Returns the similarity between two users """
        jacc = jaccard_similarity(self._get_user_movies(user1_id),
                                  self._get_user_movies(user2_id))
        corr = self._get_user_correlation(user1_id, user2_id)
        corr_w, jacc_w = self.correlation_weight, 1 - self.correlation_weight
        return corr * corr_w + jacc * jacc_w


    def _get_user_candidates(self, user_id):
        """ Returns the set of possible candidates for neighbors of input user """
        neighs = {}
        for m_id in self._get_user_movies(user_id, list=True):
            users = self._find_users_by_movies(m_id)
            for u_id in users:
                if not u_id in neighs:
                    neighs[u_id] = 0
                neighs[u_id] += 1
        return neighs


    def _process_example(self, user_id, movie_id, rating):
        """ CBR cycle step for the given user for the given movie and rating """
        sim_users = self.retrieve(user_id, neighbors=self.neighbors) # Retrieves a set of user ids
        sim_movies = self.reuse(user_id, neighbors=sim_users, N=self.movies_per_neighbors) # Retrieves a set of MovieInfo
        feedback = self.review(user_id, ranked=sim_movies, movie_id=movie_id, rating=rating)
        self.retain(user_id, feedback)


    def retrieve(self, user_id, neighbors=10):
        """ See base class """

        logger.info("Retrieving phase for user %d" % user_id)

        # User candidates as those who has rated at least one movie in common with query
        candidates = self._get_user_candidates(user_id)

        logger.info("Thresholding candidates (%d)" % len(candidates))
        print(self.min_movies_candidate)
        candidates = [k for (k, v) in candidates.iteritems() if v > self.min_movies_candidate]

        logger.info("Obtaining user similarities (%d)" % len(candidates))

        # Get shared movies and correlation for candidates
        stats = [(c_id, self._get_user_similarity(user_id, c_id)) for c_id in candidates]

        logger.info("Sorting user similarities")

        # Return top
        sorted_stats = sorted(stats, key=lambda tup: tup[1], reverse=True)
        return sorted_stats[:neighbors]


    def reuse(self, user_id, neighbors, N):
        """ See base class """
        return NotImplementedError("To be implemented")


    def review(self, user_id, ranked, movie_id, rating):
        """ See base class """
        return NotImplementedError("To be implemented")


    def retain(self, user_id, feedback):
        """ See base class """
        return NotImplementedError("To be implemented")
