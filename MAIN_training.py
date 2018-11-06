import pickle
import CFG_paths as paths

with open(paths.pickle_path_count, 'rb') as f:
    count_features = pickle.load(f)