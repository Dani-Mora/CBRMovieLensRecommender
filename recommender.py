from utils import *
from case_base import CaseBase, SimilarityType
from movies import FeedbackType


logger = initialize_logging('cbr')

# TODO and future work:
# We are not using timestamps and actually order here is important. Could be added or commented as future work
# End process example function

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


    def review(self, rated, recommended):
        """ CBR review cycle that evaluates the recommended movies. It resembles the user feedback
        done after the recommendation.
        Args:
            rated: RatedInfo class with the next rated movie recommended
            recommended: List of CandidateInfo objects representing the recommended movies
        Returns:
            feedback: Recommendation feedback (FeedbackInfo class)
        """


    def retain(self, rated_case, feedback_list):
        """ CBR retain cycle that retains the evaluation cases into the case base
        Args:
           rated_case: RatingInfo about the reviewed case
           feedback_list: List of CandidateInfo containing feedback from thr review phase """


class MovieRecommender(MovieRecommenderInterface):


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
                 beta=0.15,
                 gamma=0.20,
                 theta=0.1,
                 movie_threshold=0.10,
                 sim_measure=SimilarityType.PEARSON):
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
            movie_threshold: Maximum distance between two movies to be considered similar
            sim_measure: Similarity measure used to compute user similarities.
        """
        self.cb = CaseBase(path,
                           top_movies=top_movies,
                           initial_affinity=initial_affinity,
                           correlation_weight=correlation_weight,
                           min_movies_candidate=min_movies_candidate,
                           update_rate=update_rate,
                           alpha=alpha,
                           beta=beta,
                           gamma=gamma,
                           theta=theta,
                           sim_measure=sim_measure)
        self.cb.initialize()

        self.neighbors = neighbors
        self.movies_per_neighbors = movies_per_neighbor
        self.movie_thresh = movie_threshold


    # Example of CBR cycle
    def _process_example(self):
        """ CBR cycle step for the given user for the given movie and rating """
        rated = self.cb.next_test_case()
        sim_users = self.retrieve(rated.user, neighbors=self.neighbors) # Retrieves a set of user ids
        sim_movies = self.reuse(rated.user, neighbors=sim_users, N=self.movies_per_neighbors) # Retrieves a set of MovieInfo
        feedback = self.review(rated, sim_movies)
        self.retain(rated, feedback)
        # TODO: Tell cbr to add incorporate 'rated' into the ratings database


    """ Implementations of the Recommender interface """


    def retrieve(self, user_id, neighbors=10):
        """ See base class """

        logger.info("Retrieving phase for user %d" % user_id)

        # User candidates as those who has rated at least one movie in common with query
        candidates = self.cb.get_user_candidates(user_id)

        # Get shared movies and correlation for candidates
        logger.info("Obtaining user similarities (%d)" % len(candidates))
        stats = [(c_id, self.cb.get_user_similarity(user_id, c_id)) for c_id in candidates]

        # Return top
        logger.info("Sorting user similarities")
        sorted_stats = sorted(stats, key=lambda tup: tup[1], reverse=True)
        return sorted_stats[:neighbors]


    def reuse(self, user_id, neighbors, N):
        """ See base class """

        # Save user's neighbors in CaseBase for user_affinity
        neighbor_id_list = [(n[0]) for n in neighbors]
        self.cb.save_user_neighbors(user_id, neighbor_id_list)

        logger.info("Reuse phase for user %d" % user_id)
        movies = []

        # Iterate over retrieved neighbors to generate movie candidates
        for (neighbor_id, _) in neighbors:

            # Create a candidate for all unseen movies
            unseen_movies = self.cb.get_suggestions(user_id, neighbor_id, N)
            for m_id in unseen_movies:
                candidate = self.cb.get_movie_candidate(movie_id=m_id, user_id=user_id, neigh_id=neighbor_id)
                movies.append(candidate)

        # Return N top movies
        return sorted(movies, key=lambda x: x.score, reverse=True)[:N]


    def review(self, rated, recommended):
        """ See base class """
        logger.info("Reuse phase for user %d" % rated.user)

        # Fill feedback of recommendations
        for rec in recommended:


            print "Candidate Info ", rec.user
            print "Movie ",
            # Debug
            sim = self.cb.get_movie_similarity(rec.movie, rated.movie)
            rat = self.cb.get_mean_user_rating(rec.user)

            print('Mean rating is %f and similarity is %f' % (rat, sim))

            if rec.movie == rated.movie and rated.rating > self.cb.get_mean_user_rating(rec.user):
                logger.info("Recommended movie %d is the same - Good rating" % rec.movie)
                rec.feedback = FeedbackType.GOOD

            elif rec.movie == rated.movie and rated.rating <= self.cb.get_mean_user_rating(rec.user):
                logger.info("Recommended movie %d was the same - Bad rating" % rec.movie)
                rec.feedback = FeedbackType.BAD

            elif self.cb.get_movie_similarity(rec.movie, rated.movie) > self.movie_thresh:

                if rated.rating > self.cb.get_mean_user_rating(rec.user):
                    logger.info("Movie %d is similar to %s - Good rating" % (rec.movie, rated.movie))
                    rec.feedback = FeedbackType.GOOD
                else:
                    logger.info("Movie %d is similar to %s - Bad rating" % (rec.movie, rated.movie))
                    rec.feedback = FeedbackType.BAD
            else:
                # Movie is different
                if rated.rating > self.cb.get_mean_user_rating(rec.user):
                    logger.info("Movie %d is not similar to %s - Good rating" % (rec.movie, rated.movie))
                    rec.feedback = FeedbackType.BAD
                else:
                    logger.info("Movie %d is not similar to %s - Bad rating" % (rec.movie, rated.movie))
                    rec.feedback = FeedbackType.NEUTRAL

        return recommended


    def retain(self, rated_case, feedback_list):
        """ See base class
        Args:
           rated_case: RatingInfo about the reviewed case
           feedback_list: List of CandidateInfo containing feedback from thr review phase """

        user_id = rated_case.user
        logger.info("Retaining phase for user %d", user_id)

        for c in feedback_list:
            # Updating genre willignes of user_id depening on CandidateInfo object that was reviewed
            self.cb.update_genre_willigness(user_id, c)
            # Updating user affinity of user_id
            self.cb.update_user_affinity(user_id, c)


        # Adding rated case into inverted file structur
        # Updating means of CaseBase
        return
