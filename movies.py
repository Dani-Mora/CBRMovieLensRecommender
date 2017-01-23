
""" File for including movie(item)-related code """


class FeedbackType(object):

    GOOD = 'good'
    NEUTRAL = 'neutral'
    BAD = 'bad'


class CandidateInfo(object):

    """ Represents a candidate movie recommended from a certain user in the neighborhood """

    def __init__(self, name, movie_id, user_id, neighbor_id_rated, score, genres, genre_representation, feedback=FeedbackType.NEUTRAL):
        """ Constructs a movie candidate from the ids of the user and the movie """
        self.name = name
        self.movie = movie_id
        self.user = user_id
        self.neighbor_id_rated = neighbor_id_rated
        self.score = score
        self.genres = genres
        self.genre_representation = genre_representation
        self.feedback = feedback

    def __repr__(self):
        return ("Movie candidate : " + str(self.name) + ", movie genres: " +
            self._get_representation())


    def _get_representation(self):
        string = ''
        for g in self.genre_representation:
            string += ' ' + str(g) + ' |'
        return string


class RatingInfo(object):

    """ Represents the feedback of the recommendation of a movie from a certain user
    in the neighborhood """

    def __init__(self, movie_id, user_id, rating, genres, timestamp):

        """ Builds movie feedback from a movie and user. """
        self.movie = movie_id
        self.user = user_id
        self.rating = rating
        self.genres = genres
        self.rating = rating
        self.timestamp = timestamp

    def __repr__(self):
        return ("RatingInfo object with movie ID: " + str(self.movie) + "\n"
                "User ID: " + str(self.user) + "\n"
                "Movie rating: " + str(self.rating))
