from recommender import MovieRecommender
from utils import MOVIELENS1M_DATA_FOLDER, check_download_data

# Download MovieLens data
#check_download_data()

# Starting point of the CBR Recommender
# Create and initialize recommender
rec = MovieRecommender(path=MOVIELENS1M_DATA_FOLDER,
                       top_movies=6)

while(1):
      keypressed = raw_input('\nCBR Recommender System. Press c to get the new case, press q to quit the application: ')
      if keypressed == 'q':
          print "Exiting the application."
          break
      elif keypressed == 'c':
          print "Getting the new case..."
          rec._process_next_case()
      else:
           print("Unknown input, inputs are c and q.")
