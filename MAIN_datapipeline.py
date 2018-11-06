import CLASS_ReviewDataSet as rds
import CFG_paths as paths

# Split percentage
target_test_size = 0.2

# Import, cleaning and training/test split
reviews = rds.ReviewDataset(paths.raw_data_path)
reviews.clean_text()
reviews.split(target_test_size)

# Text representations creation
reviews.generate_representation("set")
reviews.generate_representation("count")
reviews.generate_representation("tfidf")

# Save data
reviews.save_data(paths.clean_data_csv)
reviews.save_train_test(paths.training_data_csv, paths.test_data_csv)
reviews.pickle_train_test_scores(paths.pickle_score_train, paths.pickle_score_test)
reviews.pickle_representation("count", paths.pickle_path_count_train, 
                              paths.pickle_path_count_test)
reviews.pickle_representation("set", paths.pickle_path_set_train, 
                              paths.pickle_path_set_test)
reviews.pickle_representation("tfidf", paths.pickle_path_tfidf_train, 
                              paths.pickle_path_tfidf_test)





