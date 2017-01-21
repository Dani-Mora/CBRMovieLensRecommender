
import numpy as np
from movies import FeedbackType

""" File for including user(customer)-related code """


class AffinityCaseBase:

    """ Case base including the recommendation preference between system elements.

    Examples:
        CaseBase.user_affinity    = user preference against another
        CaseBase.genre_willigness = user preference against a movie genre

    Preference is a real value in [-1, 1] where -1 means that there is no affinity
    , 0 means neutrality and 1 means highets affinity
    """

    def __init__(self, initial_preference=0, modifier=0.05):
        """ Class constructor
        Args:
            initial_preference: Initial affinity measure. In interval [-1, 1]
            modifier: Rate at which we update affinity. In interval [0, 1]
        """
        if initial_preference < -1 or initial_preference > 1:
            raise ValueError("Initial preference must be in interval [-1, 1]")
        if modifier < 0 or modifier > 1:
            raise ValueError("Update rate must be in interval [0, 1]")
        self.rate = modifier
        self.preference = {}
        self.init_value = initial_preference


    def _check_affinity(self, elem1, elem2):
        """ Checks whether the affinity between element 1 and 2 exists. Otherwise,
        creates it from scratch with the default value """
        if not elem1 in self.preference:
            self.preference[elem1] = {}
        if not elem2 in self.preference[elem1]:
            self.preference[elem1][elem2] = self.init_value


    def _check_value(self, elem1, elem2):
        """ Checks whether affinity value is out of range [-1, 1] and clips it in that case """
        if self.preference[elem1][elem2] < -1:
            self.preference[elem1][elem2] = -1
        elif self.preference[elem1][elem2] > 1:
            self.preference[elem1][elem2] = 1


    def get_affinity(self, elem1, elem2):
        """ Returns affinity between both elements """
        self._check_affinity(elem1, elem2)
        return self.preference[elem1][elem2]


    def update_preference(self, elem1, elem2, feedback):
        """ Modifies the preference of element elem1 towards elem2
        Args:
            user1: First element of the preference relation
            user2: Second element of the preference relation
            feedback: Enumerateed feedback of the affinity. Positive feedback are good
                while feedback below 0 represents a bad affinity update """

        feedback_value = 0
        if feedback == FeedbackType.GOOD:
                feedback_value = 1
        elif feedback == FeedbackType.BAD:
                feedback_value = -1
        elif feedback == FeedbackType.NEUTRAL:
                feedback_value = 0

        self._check_affinity(elem1, elem2)
        self.preference[elem1][elem2] += (feedback_value * self.rate)
        self._check_value(elem1, elem2)
