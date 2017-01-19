from users import *
from utils import *
import pandas as pd
from tqdm import tqdm


logger = initialize_logging('casebase')


class CaseBase(object):

    def __init__(self,
                 path,
                 top_movies=10,
                 initial_affinity=0.5,
                 correlation_weight=0.75,
                 min_movies_candidate=0.33,
                 update_rate=0.1,
                 alpha=0.25,
                 beta=0.15,
                 gamma=0.20,
                 theta=0.1,
                 train_ratio=0.8):
        """ Constructor of the class
        Args:
            path: Path to MovieLens 1M dataset.
            top_movies: Top rated movies to use for searching candidates.
            initial_affinity: Initial affinity for users.
            correlation_weight: Weight in interval [0, 1] for the rating correlation
                in user similarity computation.
            min_movies_candidate: Ratio of the mean ratings per user that two users need two share
                at least to be considered neighbors. By default one third of the movies must be shared.
            update_rate: Pace at which we update the affinity of the users and the genre in recommendations.
            alpha: Weight for the user correlation in the movie score.
            beta: Weight for the popularity (mean rating in the system) in the movie score.
            gamma: Weight for the correlation of user preferences in the movie score.
            theta: Weight for the willigness of user preferences in the movie score.
            train_ratio: Fraction of ratings belonging to training.
        """
        self.top_movies = top_movies
        self.correlation_weight = correlation_weight

        # Case base structures
        self.user_affinity = AffinityCaseBase(initial_preference=initial_affinity,
                                              modifier=update_rate)  # User recommendation feedback
        self.genre_willigness = AffinityCaseBase(initial_preference=initial_affinity,
                                                 modifier=update_rate)  # Recommendations feedback
        self.inverted_file = {}  # Store movie - user who rated it
        self.train_indexes = {}  # Stores training movie identifiers for each user
        self.test_indexes = {}  # Stores the movie identifiers for each user

        # Pandas datasets
        self.movies, self.users = None, None
        self.all_ratings, self.ratings, self.test_ratings = None, None, None

        # Case base global structures
        self.popular = None  # Popular movies
        self.mean_movie_rating = None  # Mean score for each movie
        self.mean_user_rating = None  # Mean rating for each user
        self.test_ratings = None

        # Movie score parameters
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.theta = theta

        # Check ratios are in correct interval to avoid unexpected behaviors
        check_ratio('Initial affinity', initial_affinity)
        check_ratio('Correlation weight', correlation_weight)
        check_ratio('Uodate rate', update_rate)
        check_ratio('Ratings ratio', min_movies_candidate)
        check_ratio('Alpha', alpha)
        check_ratio('Beta', beta)
        check_ratio('Gamma', gamma)
        check_ratio('Thetha', theta)
        check_ratio("Training ratio", train_ratio)

        # Read data
        self._read_data(path)


    """ Read data from files"""


    def _read_data(self, path):
        """ Reads Movielens 1M data into the class """
        # Make sure ratings indexed by user_id and movie_id
        self.all_ratings = pd.read_csv(os.path.join(path, 'ratings.dat'),
                                       sep="::",
                                       names=['user_id', 'movie_id', 'rating', 'timestamp'],
                                       engine='python')
        self.all_ratings.set_index(['user_id', 'movie_id'])

        # Users indexed by id. Each user has at least 20 ratings. No missing data
        self.users = pd.read_csv(os.path.join(path, 'users.dat'),
                                 sep="::",
                                 names=['user_id', 'sex', 'age_group', 'occupation', 'zip_code'],
                                 engine='python')
        self.users.set_index(['user_id'])

        # Movies indexed by id
        self.movies = pd.read_csv(os.path.join(path, 'movies.dat'),
                                  sep="::",
                                  names=['movie_id', 'name', 'genre'],
                                  engine='python')
        self.movies.set_index(['movie_id'])

        # Read possible genres
        self.genres = self._get_genres()
        # Build dummy column for each genre
        self._genres_to_dummy()


    def _get_genres(self):
        """ Returns unique genres in the system """
        separated = self.movies['genre'].apply(self.separate_genre)
        return {g: True for x in separated for g in x}.keys()


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


    """ Initialization/finalization functions """


    def initialize(self, train_ratio=0.8):
        """ Initialize case base and fills inverted file index
        Complexity:
            O(u), where u = number of users,
            Considering that ratings_user << number of users
        """
        logger.info("Initializing CBR case base ...")

        # Separate ratings dataframe between train and test
        train, test = split_data(self.all_ratings.shape[0], train_ratio)
        self.ratings = self.all_ratings.loc[self.all_ratings.index.values[train]]
        self.test_ratings = self.all_ratings.loc[self.all_ratings.index.values[test]]

        # Iterate over all users
        users = self._get_users_list()
        for u_id in tqdm(users, desc="Initializing user information"):

            # Fill inverted file index with user movies
            for m_id in self._get_user_movies(u_id, list=True):
                self.add_user_rating(u_id, m_id)

        # Compute global structures
        logger.info("Initializing movie popularity ...")
        self.update_popularity()
        logger.info("Initializing mean movie score ...")
        self.update_mean_movie_rating()
        logger.info("Initializing mean user score ...")
        self.update_mean_user_rating()


    def finalize(self):
        """ Closes all used resources """
        self.ratings.close()
        self.users.close()
        self.movies.close()


    """ Global functions. To be computed periodically, they could be expensive """


    def update_popularity(self):
        """ Computes a data frame sorting movies by its popularity (number of ratings) """
        ratings_count = self.ratings.groupby(['movie_id'])['rating'].count().reset_index()
        sorted_ids = ratings_count.sort_values(['rating'], ascending=[False])
        self.popular = pd.merge(sorted_ids, self.movies, on=['movie_id'])


    def update_mean_movie_rating(self):
        """ Stores the mean rating for each movie """
        self.mean_movie_rating = self.ratings.groupby(['movie_id'])['rating'].mean().reset_index()


    def update_mean_user_rating(self):
        """ Stores the mean rating for each movie """
        self.mean_user_rating = self.ratings.groupby(['user_id'])['rating'].mean().reset_index()


    def get_mean_movie_rating(self, movie_id):
        """ Returns the mean of a movie in the system """
        return self.mean_movie_rating[self.mean_movie_rating['movie_id'] == movie_id]['rating'].item()


    def get_mean_user_rating(self, user_id):
        """ Returns the mean rating of a user in the system """
        return self.mean_user_rating[self.mean_user_rating['user_id'] == user_id]['rating'].item()


    """ User-related functions """


    def _get_users_list(self):
        """ Returns the list of user identifiers """
        return self.users['user_id'].tolist()


    def add_user_rating(self, user_id, movie_id):
        """ Adds a user rating to the inverted file indexed structure """
        if not movie_id in self.inverted_file:
            self.inverted_file[movie_id] = []
        self.inverted_file[movie_id].append(user_id)


    def _find_users_by_movies(self, movie_id):
        """ Returns the user identifiers related to users who saw the input movie """
        return self.inverted_file[movie_id]


    def _get_user_preferences(self, user_id):
        """ Sets the user genre preferences for the training set of movies
        Args:
            user_id: Identifier of the user
        """
        # User training ratings
        user_ratings = self.ratings[(self.ratings['user_id'] == user_id)]

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
        return np.nan_to_num(genres_sum_mat * mean_ratings)


    def _get_user_movies(self, user_id, list=False):
        """ Returns the movies a user has rated
        Args:
            list: Whether to return a Matrix (False) or a list (True) """
        user_movies = self.ratings[self.ratings['user_id'] == user_id]['movie_id']
        return user_movies.tolist() if list else user_movies.as_matrix()


    def _get_user_ratings(self, user_id):
        """ Returns the DataFrame of ratings from a user """
        return self.ratings[self.ratings['user_id'] == user_id]


    def get_shared_ratings(self, user1_id, user2_id):
        """ Returns the Dataframe of  movies rated by the two users """
        return pd.merge(self._get_user_ratings(user1_id),
                        self._get_user_ratings(user2_id),
                        on=['movie_id'])


    def _get_top_movies(self, user_id, top, list=False):
        """ Returns top rated movies for user. Tries to retrieve min(movies_user, top) """
        movies = self._get_user_ratings(user_id).sort_values(['rating'], ascending=[False])
        top_movies = movies.head(n=min(movies.shape[0], top))['movie_id']
        return top_movies.tolist() if list else top_movies.as_matrix()


    def get_user_candidates(self, user_id):
        """ Returns the set of possible candidates for neighbors of input user as the users
        that has seen N top movies from the input user """
        candidates = None
        for m_id in self._get_top_movies(user_id, top=self.top_movies, list=True):
            users = self._find_users_by_movies(m_id)
            candidates = set(users) if candidates is None else candidates.intersection(users)
        # Remove user from candidates
        return candidates - set([user_id])


    """ Movie-related functions """


    def get_suggestions(self, user_id, neighbor_id, movies_per_neighbor):
        """ Returns top movies not seen by the user """
        rated_movies = self.ratings[(self.ratings['user_id'] == neighbor_id)
                                    & (self.ratings['movie_id'].isin(self._get_unseen_movies(user_id, neighbor_id)))]
        return rated_movies.iloc[:movies_per_neighbor]['movie_id']


    def _get_unseen_movies(self, user_id, neighbor_id):
        """ Returns the movies of given neighbor that user_id did not watch """
        neighbor_movies = self._get_user_movies(neighbor_id, list=True)
        movies_watched = self._get_user_movies(user_id, list=True)
        return list(set(neighbor_movies) - set(movies_watched))


    def _get_user_movie_rating(self, user_id, movie_id):
        """ Returns rating of given user_id for movie_id, after subtracting the user mean """
        rating = self.ratings[(self.ratings['user_id'] == user_id) &
                              (self.ratings['movie_id'] == movie_id)]['rating'].iloc[0]
        user_mean = self.get_mean_user_rating(user_id)
        return rating - user_mean


    def _get_genre_correlation(self, user_id, movie_id):
        """ Returns the correlation between the movie genres and the user preferences """
        user_prefs = self._get_user_preferences(user_id)
        movie_prefs = self.movies[self.movies['movie_id'] == movie_id][self.genres].iloc[0].as_matrix()
        return np.dot(user_prefs, movie_prefs)


    def _get_willingness_vector(self, user_id):
        """ Returns the willigness vector for the given user """
        will_vec = np.zeros(len(self.genres))
        for i, g in enumerate(self.genres):
            will_vec[i] = self.genre_willigness.get_affinity(user_id, g)
        return will_vec


    def _get_willingness(self, user_id, movie_id):
        """ Returns the willigness of a user against a certain movie given his
        recommendation willigness preferences """
        user_will = self._get_willingness_vector(user_id)
        movie_prefs = self.movies[self.movies['movie_id'] == movie_id][self.genres].iloc[0].as_matrix()
        return np.dot(user_will, movie_prefs)


    """ Similarity functions """


    def _get_user_correlation(self, user1_id, user2_id):
        """ Returns the balance of differences between ratings of rated films by both user.
        Takes only into account those films rated by both """
        shared_ratings = self.get_shared_ratings(user1_id, user2_id)

        # Substract means for both users
        shared_ratings['rating_x'] -= self.get_mean_user_rating(user1_id)
        shared_ratings['rating_y'] -= self.get_mean_user_rating(user2_id)

        # Compute correlation as inverse of disparity
        disparity = (shared_ratings['rating_x'] - shared_ratings['rating_y']).abs().mean()
        return 1.0/disparity


    def get_user_similarity(self, user1_id, user2_id):
        """ Returns the similarity between two users, as:

            user_similarity(u1, u2) = w_1 * jaccard(u1, u2) + w_2 * correlation(u1, u2)

            Where:
                - 'jaccard(u1,u2)' is the jaccard similarity of the rated set of movies
                    for users u1 and u2
                - 'correlation(u1,u2)' is the inverse of the disparity of scores in
                    rated movies between users u1 and u2
         """
        jacc = jaccard_similarity(self._get_user_movies(user1_id),
                                  self._get_user_movies(user2_id))
        corr = self._get_user_correlation(user1_id, user2_id)
        corr_w, jacc_w = self.correlation_weight, 1 - self.correlation_weight
        return corr * corr_w + jacc * jacc_w


    def get_movie_score(self, movie_id, user_id, neigh_id):
        """ Returns the score for given movie taking into account user and given neighbor, as:

            Score(m, u1, u2) = \alpha * correlation(u1, u2) * rating(u2, m)
                                + \beta * mean_rating(m)
                                + \gamma * genre(u1, m)
                                + \theta * willingness(u1, m)

            Where:
                - 'correlation(u1,u2)' is the inverse of the disparity of scores in
                    rated movies between users u1 and u2
                - 'mean_rating(m) is the mean rating of movie m in the system
                - 'genre(u1,m)' is the dot product between the user u1 preferences and the movie genres from m
                - willigness(u1, m) is the dot product between the genre wwillingness of u1 and
                    the movie genres from m

            Alpha, Beta, Gamma and Theta are preset constants.
        """

        return self.alpha * self._get_user_correlation(user_id, neigh_id) * self._get_user_movie_rating(neigh_id, movie_id) + \
               self.beta * self.get_mean_movie_rating(movie_id) + \
               self.gamma * self._get_genre_correlation(user_id, movie_id) + \
               self.theta * self._get_willingness(user_id, movie_id)
