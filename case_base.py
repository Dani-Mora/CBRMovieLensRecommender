from users import *
from utils import *
import pandas as pd
from tqdm import tqdm

from movies import RatingInfo, CandidateInfo

logger = initialize_logging('casebase')


class SimilarityType:

    JACCARD = 'jaccard'
    PEARSON = 'pearson'


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
                 train_ratio=0.8,
                 sim_measure=SimilarityType.PEARSON):
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
            sim_measure: Similarity measure used to compute user similarities.
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

        # User's neighbor caching for user affinity
        # Key is user ID, value is list of neighbor IDs
        self.user_neighbors = {}

        # User similarity caching
        self.user_cache = {}

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

        self.sim_measure = sim_measure


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


    def next_test_case(self):
        """ Returns the next available test case. None if tests exhausted """
        if self.test_ratings.shape[0] > 0:
            test_case = self.test_ratings.iloc[0]
            return RatingInfo(movie_id=test_case['movie_id'],
                                      user_id=test_case['user_id'],
                                      rating=test_case['rating'],
                                      genres=self._get_genre_vector(test_case['movie_id']),
                                      timestamp=test_case['timestamp'])
        else:
            return None


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


    def get_user_candidates(self, user_id):
        """ Returns the set of possible candidates for neighbors of input user as the users
        that has seen at least N top movies from the input user """
        # Get top movies for user
        movies = self._get_user_ratings(user_id).sort_values(['rating'], ascending=[False])
        top_movies = movies.head(n=min(movies.shape[0], self.top_movies))['movie_id'].tolist()

        # Get intersection of users that have seen the top movies
        candidates = None
        for m_id in top_movies:
            users = self._find_users_by_movies(m_id)
            candidates = set(users) if candidates is None else candidates.intersection(users)

        # Remove user from candidates
        return candidates - set([user_id])

    def save_user_neighbors(self, user_id, neighbors):
        """ Method saves the users from reuse phase to the case base with key user_id """
        self.user_neighbors[user_id] = neighbors


    def _get_user_affinity(self, user_id):
        """ Returns the user_affinity vector to his neighbors for the given user """
        neighbors = self.user_neighbors[user_id]
        aff_vec = np.zeros(len(neighbors))
        for index, neighbor in enumerate(neighbors):
            aff_vec[index] = self.user_affinity.get_affinity(user_id, neighbor)
        return aff_vec


    def update_user_affinity(self, user_id, candidate_with_feedback):
        """ Function updates user_affinity with given feedback
        Args:
            candidate_with_feedback: CandidateInfo object containg information about the movie candidate with given feedback """
        # Update only user's neighbor that is Candidate with feedback
        for index, neighbor in enumerate(self.user_neighbors[user_id]):
            if neighbor == candidate_with_feedback.neighbor_id_rated:
                self.user_affinity.update_preference(user_id, neighbor, candidate_with_feedback.feedback)


    """ Movie-related functions """


    def get_suggestions(self, user_id, neighbor_id, movies_per_neighbor):
        """ Returns top movies not seen by the user """
        # Get movies not seen by user
        neighbor_movies = self._get_user_movies(neighbor_id, list=True)
        movies_watched = self._get_user_movies(user_id, list=True)
        unseen_movies = list(set(neighbor_movies) - set(movies_watched))

        # Return subset of movies top rated movies not seen by query user
        rated_movies = self.ratings[(self.ratings['user_id'] == neighbor_id)
                                    & (self.ratings['movie_id'].isin(unseen_movies))]
        top_movies = rated_movies.sort_values(['rating'], ascending=[False])
        return top_movies.iloc[:movies_per_neighbor]['movie_id']


    def _get_user_movie_rating(self, user_id, movie_id):
        """ Returns rating of given user_id for movie_id, after subtracting the user mean """
        rating = self.ratings[(self.ratings['user_id'] == user_id) &
                              (self.ratings['movie_id'] == movie_id)]['rating'].iloc[0]
        user_mean = self.get_mean_user_rating(user_id)
        return rating - user_mean


    def _get_genre_vector(self, movie_id, list=False):
        """ Returns the genres of the input movie
        Args:
            list: Whether to return a Matrix (False) or a list (True) """
        if list:
            return self.movies[self.movies['movie_id'] == movie_id][self.genres].iloc[0].tolist()
        else:
            return self.movies[self.movies['movie_id'] == movie_id][self.genres].iloc[0].as_matrix()


    def get_movie_similarity(self, mi, mj):
        """ Returns the similarity between two movies as:

            movie_sim(m_i, m_j) = |intersect(U_i, U_j)| / |union(U_i, U_j)|

            Where U_i is the set of users that has seen movie m_i

        """
        u_i = set(self.inverted_file[mi])
        u_j = set(self.inverted_file[mj])
        return float(len(u_i.intersection(u_j))) / float(len(u_i.union(u_j)))

    def _get_willingness_vector(self, user_id):
        """ Returns the willigness vector for the given user """
        will_vec = np.zeros(len(self.genres))
        for i, g in enumerate(self.genres):
            will_vec[i] = self.genre_willigness.get_affinity(user_id, g)
        return will_vec


    def update_genre_willigness(self, user_id, candidate_with_feedback):
        """ Function updates genre_willigness with given feedback
        Args:
            candidate_with_feedback: CandidateInfo object containg information about the movie candidate with given feedback """
        genres = self._get_genre_vector(candidate_with_feedback.movie, list=True)
        genre_indices = [i for i, x in enumerate(genres) if x]
        # Update only genres of candidate_with_feedback movie
        for index, genre in enumerate(self.genres):
            if index in genre_indices:
                self.genre_willigness.update_preference(user_id, genre, candidate_with_feedback.feedback)


    """ Similarity functions """


    def get_user_similarity(self, user1, user2):
        """ Checks whether the input user similarity has been cached """
        if not user1 in self.user_cache or not user2 in self.user_cache:
            self.user_cache[user1] = {}
            self.user_cache[user1][user2] = self._compute_user_similarity(user1, user2)
            return self.user_cache[user1][user2]
        elif user1 in self.user_cache:
            if not user2 in self.user_cache[user1]:
                self.user_cache[user1][user2] = self._compute_user_similarity(user1, user2)
                return self.user_cache[user1][user2]
        else:
            if not user1 in self.user_cache[user2]:
                self.user_cache[user2][user1] = self._compute_user_similarity(user1, user2)
            return self.user_cache[user2][user1]


    def _compute_user_similarity(self, user1, user2):
        """ Computes the similarity/correlation between the input users depending on the
        case based defined similarity measure """
        if self.sim_measure == SimilarityType.JACCARD:
            return self._compute_jaccard(user1, user2)
        elif self.sim_measure == SimilarityType.PEARSON:
            return self._compute_pearson(user1, user2)
        else:
            raise ValueError("Invalid similarity type: %d" % self.sim_measure)


    def _get_correlation(self, user1_id, user2_id):
        """ Returns the balance of differences between ratings of rated films by both user.
        Takes only into account those films rated by both """
        shared_ratings = self.get_shared_ratings(user1_id, user2_id)

        # Substract means for both users
        shared_ratings['rating_x'] -= self.get_mean_user_rating(user1_id)
        shared_ratings['rating_y'] -= self.get_mean_user_rating(user2_id)

        # Compute correlation as inverse of disparity
        disparity = (shared_ratings['rating_x'] - shared_ratings['rating_y']).abs().mean()
        return 1.0/disparity


    def _compute_jaccard(self, user1_id, user2_id):
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
        corr = self._get_correlation(user1_id, user2_id)
        corr_w, jacc_w = self.correlation_weight, 1 - self.correlation_weight
        return corr * corr_w + jacc * jacc_w


    def _compute_pearson(self, user1_id, user2_id):
        """ Returns the Pearson correlation between the two users """
        shared_movies = self.get_shared_ratings(user1_id, user2_id)
        ratings1, ratings2 = shared_movies['rating_x'].as_matrix(), shared_movies['rating_y'].as_matrix()
        mean_user1, mean_user2 = self.get_mean_user_rating(user1_id), self.get_mean_user_rating(user2_id)
        return pearson_correlation(ratings1, ratings2, mean_user1, mean_user2)


    def get_movie_candidate(self, movie_id, user_id, neigh_id):
        """ Returns the candidate information for the given movie, given that is recommended to
        a specific user and coming from another one

        Computes a score for the movie, based on:

            Score(m, u1, u2) = \alpha * similarity(u1, u2) * rating(u2, m)
                        + \beta * mean_rating(m)
                        + \gamma * genre(u1, m)
                        + \theta * willingness(u1, m)

            Where:

                - 'similarity(u1,u2)' is the inverse of the disparity of scores in
                    rated movies between users u1 and u2

                - 'mean_rating(m) is the mean rating of movie m in the system

                - 'genre(u1,m)' is the dot product between the user u1 preferences and the movie genres from m

                - willigness(u1, m) is the dot product between the genre wwillingness of u1 and
                    the movie genres from m

        Args:
            movie_id: Movie identifier
            user_id: User recommendation is made to
            neigh_id: User recommendation comes from
        Returns:
            candidate: CandidateInfo class
        """
        movie_genres = self._get_genre_vector(movie_id)

        # Compute score
        user_term = self.alpha * self.get_user_similarity(user_id, neigh_id) \
                    * self._get_user_movie_rating(neigh_id, movie_id)
        rating_term = self.beta * self.get_mean_movie_rating(movie_id)
        genre_term = self.gamma * np.dot(self._get_user_preferences(user_id), movie_genres)
        will_term = self.theta * np.dot(self._get_willingness_vector(user_id), movie_genres)
        score = user_term + rating_term + genre_term + will_term

        # User-affinty value
        # At the moment not in the use
        aff_vec = self._get_user_affinity(user_id)


        return CandidateInfo(movie_id=movie_id,
                             user_id=user_id,
                             neighbor_id_rated=neigh_id,
                             score=score,
                             genres=movie_genres)
