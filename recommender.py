from utils import *
from case_base import CaseBase
from movies import CandidateInfo


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

    def __init__(self,
                 path,
                 neighbors=10,
                 movies_per_neighbor=15,
                 top_movies=10,
                 initial_affinity=0.5,
                 correlation_weight=0.75,
                 min_movies_candidate=0.33,
                 update_rate=0.1,
                 alpha=0.25,
                 beta = 0.15,
                 gamma = 0.20,
                 theta = 0.1):
        """ Constructor of the class
        Args:
            path: Path to MovieLens 1M dataset.
            neighbors: Number of neighbors to retrieve in the CBR cycle.
            movies_per_neighbor: Number of movies per neighbor to extract
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
        """
        self.cb = CaseBase(path,
                           top_movies=top_movies,
                           initial_affinity = initial_affinity,
                           correlation_weight = correlation_weight,
                           min_movies_candidate = min_movies_candidate,
                           update_rate = update_rate,
                           alpha = alpha,
                           beta = beta,
                           gamma = gamma,
                           theta = theta)
        self.cb.initialize()

        self.neighbors = neighbors
        self.movies_per_neighbors = movies_per_neighbor


    # Example of CBR cycle
    def _process_example(self, user_id, movie_id, rating):
        """ CBR cycle step for the given user for the given movie and rating """
        sim_users = self.retrieve(user_id, neighbors=self.neighbors) # Retrieves a set of user ids
        sim_movies = self.reuse(user_id, neighbors=sim_users, N=self.movies_per_neighbors) # Retrieves a set of MovieInfo
        feedback = self.review(user_id, ranked=sim_movies, movie_id=movie_id, rating=rating)
        self.retain(user_id, feedback)


    # Implementations of MovieRecommenderInterface
    def retrieve(self, user_id, neighbors=10):
        """ See base class """

        logger.info("Retrieving phase for user %d" % user_id)

        # User candidates as those who has rated at least one movie in common with query
        candidates = self.cb.get_user_candidates(user_id)

        logger.info("Obtaining user similarities (%d)" % len(candidates))

        # Get shared movies and correlation for candidates
        stats = [(c_id, self.cb.get_user_similarity(user_id, c_id)) for c_id in candidates]

        logger.info("Sorting user similarities")

        # Return top
        sorted_stats = sorted(stats, key=lambda tup: tup[1], reverse=True)
        return sorted_stats[:neighbors]


    # TODO: cache correlation in some way <- we could cache data for each request

    def reuse(self, user_id, neighbors, N):
        """ See base class """

        logger.info("Reuse phase for user %d" % user_id)

        movies = []

        # Iterate over retrieved neighbors to generate movie candidates
        for neighbor_id in neighbors:

            # Create a candidate for all unseen movies
            unseen_movies = self.cb.get_suggestions(user_id, neighbor_id, N)
            for m_id in unseen_movies:
                score = self.cb.get_movie_score(movie_id=m_id, user_id=user_id, neigh_id=neighbor_id)
                movies.append(CandidateInfo(m_id, neighbor_id, score))

        # Return N top movies
        return sorted(movies, key=lambda x: x.score, reverse=True)[:N]


    def review(self, user_id, ranked, movie_id, rating):
        """ See base class """
        return NotImplementedError("To be implemented")


    def retain(self, user_id, feedback):
        """ See base class """
        return NotImplementedError("To be implemented")
