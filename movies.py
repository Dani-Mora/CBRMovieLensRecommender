
""" File for including movie(item)-related code """


class FeedbackType(object):

    VERY_GOOD = 'very_good'
    GOOD = 'good'
    NEUTRAL = 'neutral'
    BAD = 'bad'


class CandidateInfo(object):

    """ Represents a candidate movie recommended from a certain user in the neighborhood """

    def __init__(self, movie_id, user_id, score, genres, feedback=FeedbackType.NEUTRAL):
        """ Constructs a movie candidate from the ids of the user and the movie """
        self.movie = movie_id
        self.user = user_id
        self.score = score
        self.genres = genres
        self.feedback = feedback


class RatingInfo(object):

    """ Represents the feedback of the recommendation of a movie from a certain user
    in the neighborhood """

    def __init__(self, movie_id, user_id, rating, genres, timestamp):

        """ Builds movie feedback from a movie and user. The feedback is given as rea in [0, 1] """
        self.movie = movie_id
        self.user = user_id
        self.rating = rating
        self.genres = genres
        self.rating = rating
        self.timestamp = timestamp
