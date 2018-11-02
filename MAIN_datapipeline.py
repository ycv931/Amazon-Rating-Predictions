import CLASS_ReviewDataSet as rds
import CFG_paths as paths

# Split percentage
target_test_size = 0.2

# Import, cleaning and training/test split
reviews = rds.ReviewDataset(paths.raw_data_path)
reviews.clean_text(clean_data_path = paths.clean_data_path, 
                   write = True)
reviews.split(target_test_size, 
              training_data_path = paths.training_data_path, 
              test_data_path = paths.test_data_path, 
              write = True)

# Text representations creation
reviews.generate_representation("set")
reviews.generate_representation("count")
reviews.generate_representation("tfidf")