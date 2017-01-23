from __future__ import division
from users import *
from utils import *
import pandas as pd
from scipy import stats
from movies import RatingInfo, CandidateInfo
import random

logger = get_logger()

class CaseBase(object):

    def __init__(self,
                 path,
                 initial_affinity=0.5,
                 update_rate=0.1,
                 alpha=0.20,
                 beta=0.10,
                 gamma=0.25,
                 theta=0.25,
                 omega=0.10,
                 train_ratio=0.9,
                 ratings_ratio=1.0):
        """ Constructor of the class
        Args:
            path: Path to MovieLens 1M dataset.
            initial_affinity: Initial affinity for users.
            update_rate: Pace at which we update the affinity of the users and the genre in recommendations.
            alpha: Weight for the user correlation in the movie score.
            beta: Weight for the popularity (mean rating in the system) in the movie score.
            gamma: Weight for the correlation of user preferences in the movie score.
            theta: Weight for the willigness of user preferences in the movie score.
            omega: Weight for the willigness of user affinity in the movie score.
            train_ratio: Fraction of ratings belonging to training.
            ratings_ratio: Ratio in interval (0, 1] of the original ratings to consider in the system.
        """
        # Case base structures
        self.user_affinity = AffinityCaseBase(initial_preference=initial_affinity,
                                              modifier=update_rate)  # User recommendation feedback
        self.genre_willigness = AffinityCaseBase(initial_preference=initial_affinity,
                                                 modifier=update_rate)  # Recommendations feedback

        # Pandas datasets
        self.movies, self.users = None, None
        self.all_ratings, self.ratings, self.test_ratings = None, None, None

        # Case base global structures
        self.popular = None  # Popular movies
        self.mean_movie_rating = None  # Mean score for each movie
        self.mean_user_rating = None  # Mean rating for each user

        # Movie score parameters
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.theta = theta
        self.omega = omega

        # Counter for test_rating set
        self.count = 0

        # User's neighbor caching for user affinity
        # Key is user ID, value is list of neighbor IDs
        self.user_neighbors = {}

        # User similarity caching
        self.user_cache = {}
        # Check ratios are in correct interval to avoid unexpected behaviors
        check_ratio('Initial affinity', initial_affinity)
        check_ratio('Update rate', update_rate)
        check_ratio('Alpha', alpha)
        check_ratio('Beta', beta)
        check_ratio('Gamma', gamma)
        check_ratio('Thetha', theta)
        check_ratio("Training ratio", train_ratio)
        check_ratio("Ratings ratio", ratings_ratio)

        # Read data
        self._read_data(path, ratings_ratio)


    """ Read data from files"""


    def _read_data(self, path, ratio):
        """ Reads Movielens 1M data into the class """
        # Make sure ratings indexed by user_id and movie_id
        self.all_ratings = pd.read_csv(os.path.join(path, 'ratings.dat'),
                                       sep="::",
                                       names=['user_id', 'movie_id', 'rating', 'timestamp'],
                                       engine='python')

        # Select subset, if requested
        if ratio < 1.0:
            # Get a subset of users from the ratings
            unique_users = self.all_ratings['user_id'].unique()
            total = len(unique_users)
            subset_users = unique_users[np.random.permutation(total)[:int(total * ratio)]].tolist()
            self.all_ratings = self.all_ratings[self.all_ratings['user_id'].isin(subset_users)]
            logger.info("Got subset of ratings of %d instances" % len(self.all_ratings.index.values))

        # Set rating indexes
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

        if self.count < self.test_ratings.shape[0]:
            test_case = self.test_ratings.iloc[self.count]
            self.count += 1
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


    def update_case_base(self, users, movies, ratings, timestamps):
        """ Updates the ratings case base with the input lists.
        Args:
            - users: List of users of the new ratings
            - movies: List of movies of the new ratings
            - ratings: List of new rating scores
            - timestamps: List of new timestamp
        """
        frame = pd.DataFrame({'user_id': users, 'movie_id': movies, 'rating': ratings, 'timestamp': timestamps})
        self.ratings = pd.concat([self.ratings, frame], ignore_index=True)


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

        # Sort test ratings so they are grouped by the user they belong to
        self.test_ratings = self.test_ratings.sort_values(['user_id'])

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
        self.popular = pd.merge(sorted_ids, self.movies, left_on=['movie_id'], right_on=['movie_id'])


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


    def _find_users_by_movies(self, movie_id):
        """ Returns the user identifiers related to users who saw the input movie """
        return self.ratings[self.ratings['movie_id'] == movie_id]['user_id'].tolist()


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


    def _get_genre_categories(self, movie_id):
        """ Return the list of category names the movie belongs to """
        genres_boolean = self._get_genre_vector(movie_id, list=True)
        genre_indices = [i for i, x in enumerate(genres_boolean) if x]
        genre_representation = []
        for index in genre_indices:
            genre_representation.append(self.genres[index])
        return genre_representation


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


    def get_user_candidates(self, user_id, num_movies, num_neighs, max_k, min_k, sim_thresh):
        """ Returns the set of possible candidates for neighbors of input user
        Args:
            user_id: User identifier
            num_movies: Number of interest movies to user for the user
            num_neighs: Minimum number of users to compute
            max_k: Maximum number of interest movies to be shared between neighbors
            min_k: Minimum number of interest movies to be shared between neighbors
            sim_thresh: Maximum rating distance between movies to be considered shared between two users
        """

        # Get movies of interest as top and lowest rated movies for user
        movies = self._get_user_ratings(user_id).sort_values(['rating'], ascending=[False])
        top_movies = movies.head(int(num_movies / 2))['movie_id'].tolist()
        lowest_rated_movies = movies.tail(int(num_movies / 2))['movie_id'].tolist()
        list_films = top_movies + lowest_rated_movies

        # Iterate from all movies until we get to min_k in decreasing order
        current_k = min(len(list_films), max_k)
        total_neighs = set()
        while current_k >= min_k:

            # Get neighbors for current k
            neighbors_k = set(self._get_neighbors(user_id, list_films, current_k, sim_thresh))
            neighbors_k -= set([user_id])

            # Compute union with previous ones
            total_neighs = total_neighs.union(neighbors_k)

            # If we arrived to number of neighbors, stop. Otherwise, keep iterating
            if len(total_neighs) >= num_neighs:
                break
            else:
                current_k -= 1

        return total_neighs


    def _get_neighbors(self, user_id, movies, current_k, sim_thresh):
        """ Returns the neighbor candidates for K random movies within the top interest movies
        for the given user """

        def movie_subset(k):
            """ Randomly returns k movies from the top ones """
            return np.array(movies)[np.random.permutation(len(movies))[:k]].tolist()

        # Get subset of k movies and ratings
        k_movies = movie_subset(current_k)
        k_ratings = [self._get_user_movie_rating(user_id, m) for m in k_movies]

        # Get users who saw the movies
        users_seen = None
        for m in k_movies:
            saw_movie = set(self._find_users_by_movies(m))
            users_seen = saw_movie if users_seen is None else users_seen.intersection(saw_movie)

        # Check whether any of the users have correlation
        candidates = []
        for u in users_seen:

            # Check if any of the K ratings differ from the maximum interval allowed
            selected = True
            for m, r in zip(k_movies, k_ratings):
                if abs(r - self._get_user_movie_rating(u, m)) > sim_thresh:
                    selected = False
                    break

            if selected:
                candidates.append(u)

        return candidates


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


    def _get_movie_name(self, movie_id):
        """ Return the name of the movie """
        return self.movies[self.movies['movie_id'] == movie_id]['name'].iloc[0]


    def update_user_affinity(self, user_id, candidate_with_feedback):
        """ Function updates user_affinity with given feedback
        Args:
            user_id: Identifier of the user
            candidate_with_feedback: CandidateInfo object containg information about the movie
                candidate with given feedback
        """
        # Update only user's neighbor that is Candidate with feedback
        for index, neighbor in enumerate(self.user_neighbors[user_id]):
            if neighbor == candidate_with_feedback.neighbor_id_rated:
                self.user_affinity.update_preference(user_id, neighbor, candidate_with_feedback.feedback)


    """ Movie-related functions """


    def get_popular_candidates(self, user_id, num=20):
        """ Returns the most popular movies not seen by the user
         Args:
            user_id: User query identifier
            num: Number of movies to retrieve
         """

        def get_random_neighbor(movie_id):
            """ Returns a random user that has been input movie """
            users = self._find_users_by_movies(movie_id)
            return users[random.randint(0, len(users) - 1)]

        def create_popular_candidate(row):
            """ Creates a popular candidate from a rating row """
            return CandidateInfo(name=row['name'],
                                 movie_id=row['movie_id'],
                                 user_id=user_id,
                                 neighbor_id_rated=get_random_neighbor(row['movie_id']),
                                 score=row['rating'],
                                 genres=row[self.genres],
                                 genre_representation=self._get_genre_categories(row['movie_id']))

        not_rated = self.popular[self.popular['movie_id'].isin(self._get_user_movies(user_id)) == False]
        not_rated_info = not_rated.head(n=num)[['name', 'movie_id', 'rating'] + self.genres]
        return [create_popular_candidate(row) for _, row in not_rated_info.iterrows()]


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
        return top_movies.iloc[:movies_per_neighbor]['movie_id'].tolist()


    def _get_user_movie_rating(self, user_id, movie_id):
        """ Returns rating of given user_id for movie_id, after subtracting the user rating mean """
        rating = self.ratings[(self.ratings['user_id'] == user_id) &
                              (self.ratings['movie_id'] == movie_id)]['rating'].iloc[0]
        user_mean = self.get_mean_user_rating(user_id)
        return rating - user_mean


    def _get_user_movie_rating_raw(self, user_id, movie_id):
        """ Returns rating of given user_id for movie_id """
        return self.ratings[(self.ratings['user_id'] == user_id) &
                              (self.ratings['movie_id'] == movie_id)]['rating'].iloc[0]


    def _get_genre_vector(self, movie_id, list=False):
        """ Returns the genres of the input movie
        Args:
            movie_id: Identifier of the movie.
            list: Whether to return a Matrix (False) or a list (True)
        """
        if list:
            return self.movies[self.movies['movie_id'] == movie_id][self.genres].iloc[0].tolist()
        else:
            return self.movies[self.movies['movie_id'] == movie_id][self.genres].iloc[0].as_matrix()


    def get_movie_similarity(self, mi, mi_genres, mj, mj_genres):
        """ Returns the similarity between two movies as:

            movie_sim(m_i, m_j) = (|intersect(U_i, U_j)| / min(|U_i|, |U_j|))
                                + jaccard_similarity_genres
                                + pearson_ratings_both_movies

            Where
                U_i is the set of users that has seen movie m_i
                Jaccard_similarity_genres calculates jaccard similarity between genres of the movies mi and mj
                Pearson_ratings_both_movies is calculation of the pearson correlation between ratings of movies both users rated
        Args:
            mi, mj: Movie_ids of the movies
            mi_genres, mj_genres: Genres of movies to calculate Jaccard similarity
        """
        # Find count of users rated by given movie_id
        u_i = set(self._find_users_by_movies(mi))
        u_j = set(self._find_users_by_movies(mj))

        # Count of users rated both movies
        rated_both = len(u_i.intersection(u_j))
        min_rated = min(len(u_i), len(u_j))
        term_1 = (float(rated_both) / float(min_rated))

        # Fastest numpy array initialization
        ratings_mi = np.empty(rated_both)
        ratings_mj = np.empty(rated_both)

        # Fill np arrays with both ratings of the movie1, movie2 for user
        inter = u_i.intersection(u_j)
        for index, user in enumerate(inter):
            ratings = self._get_user_ratings(user)
            ratings_mi[index] = ratings[(ratings['user_id'] == user) & (ratings['movie_id'] == mi)]['rating'].iloc[0]
            ratings_mj[index] = ratings[(ratings['user_id'] == user) & (ratings['movie_id'] == mj)]['rating'].iloc[0]

        # Calculate pearson correlation between ratings of movies both users rated
        pearson = stats.pearsonr(ratings_mi, ratings_mj)[0]
        pearson = 0.0 if np.isnan(pearson) else pearson

        # Index genres to use them for Jaccard similarity
        genre_indices_mi = [i for i, x in enumerate(mi_genres) if x]
        genre_indices_mj = [i for i, x in enumerate(mj_genres) if x]

        # Calculate jaccard similarity between movie genres
        genre_jaccard = improved_jaccard_similarity(genre_indices_mi, genre_indices_mj)

        # Score similarity of movie
        # Combine term_1 of count rating movie, jaccard similarity between genres and pearson
        score_similarity = term_1 + float(genre_jaccard) + pearson

        # Normalizing similarity score in range [0-1]
        min_value = -0.75
        max_value = 2.75
        normalized_similarity = float(score_similarity - min_value) / float(max_value - min_value)
        return normalized_similarity


    def _get_willingness_vector(self, user_id):
        """ Returns the willigness vector for the given user """
        will_vec = np.zeros(len(self.genres))
        for i, g in enumerate(self.genres):
            will_vec[i] = self.genre_willigness.get_affinity(user_id, g)
        return will_vec


    def update_genre_willigness(self, user_id, candidate_with_feedback):
        """ Function updates genre_willigness with given feedback
        Args:
            user_id: Identifier of the user.
            candidate_with_feedback: CandidateInfo object containg information about the movie
                candidate with given feedback
        """
        genres = self._get_genre_vector(candidate_with_feedback.movie, list=True)
        genre_indices = [i for i, x in enumerate(genres) if x]
        # Update only genres of candidate_with_feedback movie
        for index, genre in enumerate(self.genres):
            if index in genre_indices:
                self.genre_willigness.update_preference(user_id, genre, candidate_with_feedback.feedback)


    """ Similarity functions """


    def get_user_similarity(self, user1, user2):
        """ Checks whether the input user similarity has been cached """
        if user1 not in self.user_cache and user2 not in self.user_cache:
            self.user_cache[user1] = {}
            self.user_cache[user1][user2] = self._compute_user_similarity(user1, user2)
            return self.user_cache[user1][user2]
        elif user1 in self.user_cache:
            if user2 not in self.user_cache[user1]:
                self.user_cache[user1][user2] = self._compute_user_similarity(user1, user2)
            return self.user_cache[user1][user2]
        else:
            if user1 not in self.user_cache[user2]:
                self.user_cache[user2][user1] = self._compute_user_similarity(user1, user2)
            return self.user_cache[user2][user1]


    def _compute_user_similarity(self, user1, user2):
        """ Computes the similarity/correlation between the input users """
        return self._compute_pearson(user1, user2)


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
                        + |omega * user_willigness(u1, u2)
            Where:

                - 'similarity(u1,u2)' is the inverse of the disparity of scores in
                    rated movies between users u1 and u2

                - 'mean_rating(m) is the mean rating of movie m in the system

                - 'genre(u1,m)' is the dot product between the user u1 preferences and the movie genres from m

                - willigness(u1, m) is the dot product between the genre wwillingness of u1 and
                    the movie genres from m

                - user_williness(u1, u2): Willigness of user u1 of receiveing recommendations from user u2

        Args:
            movie_id: Movie identifier
            user_id: User recommendation is made to
            neigh_id: User recommendation comes from
        Returns:
            candidate: CandidateInfo class
        """
        movie_genres = self._get_genre_vector(movie_id)
        genre_representation = self._get_genre_categories(movie_id)
        name = self._get_movie_name(movie_id)

        # Compute score
        user_term = self.alpha * self.get_user_similarity(user_id, neigh_id) * self._get_user_movie_rating(neigh_id, movie_id)
        rating_term = self.beta * self.get_mean_movie_rating(movie_id)
        genre_term = self.gamma * np.dot(self._get_user_preferences(user_id), movie_genres)
        will_term = self.theta * np.dot(self._get_willingness_vector(user_id), movie_genres)
        will_user_term = self.omega * self.user_affinity.get_affinity(user_id, neigh_id)
        score = user_term + rating_term + genre_term + will_term + will_user_term

        return CandidateInfo(name=name,
                             movie_id=movie_id,
                             user_id=user_id,
                             neighbor_id_rated=neigh_id,
                             score=score,
                             genres=movie_genres,
                             genre_representation=genre_representation)
