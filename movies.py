
""" File for including movie(item)-related code """


class CandidateInfo(object):

    """ Represents a candidate movie recommended from a certain user in the neighborhood """

    def __init__(self, movie_id, user_id, score):
        """ Constructs a movie candidate from the ids of the user and the movie """
        self.movie = movie_id
        self.user = user_id
        self.score = score


class FeedbackInfo(CandidateInfo):

    """ Represents the feedback of the recommendation of a movie from a certain user
    in the neighborhood """

    def __init__(self, movie_id, user_id, feedback):
        """ Builds movie feedback from a movie and user. The feedback is given as rea in [0, 1] """
        super(FeedbackInfo, self).__init__(movie_id, user_id)
        self.feedback = feedback
