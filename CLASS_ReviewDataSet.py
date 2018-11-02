import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer

# Defines a class for structured review data.
# The template for data will have the structure of our dataset.
# The class is initialized:
# - Rating: the rating associated to the review
# - Summary: the review's summary
# - Text: the body of the review
# Summary and text need to be either strings or NaN. NaN cleaning in these
# columns is handled via the clean_text method.
# 

supported_representations = ["set", "count", "tfidf"]
column_names = ['Score', 'Summary', 'Text']

############################## Class definition ##############################

class ReviewDataset():
    def __init__(self, raw_data_path: str):            
        self.import_path = raw_data_path
        try:     
            self.data = pd.read_csv(raw_data_path) 
        except:
            raise ImportError("Pandas could not read a csv in the given path.")
            
        if list(self.data.columns.values) != ['Score', 'Summary', 'Text']:
            raise ValueError("Imported csv has wrong columns. These should be "
                             + str(column_names))
        
        self.train_set = None
        self.test_set = None
        
        self.count_features = None
        self.count_test_features = None
        
        self.set_features = None
        self.set_test_features = None
        
        self.tfidf_features = None
        self.tfidf_set_features = None
    
    # Clean fills the NaN in the text columns with empty strings. If write is
    # True, it writes the cleaned dataset to the path specified in the 
    # clean_data_path argument
    
    def clean_text(self, clean_data_path = None, 
                   write = False):
        self.data["Summary"] = self.data["Summary"].fillna("")
        self.data["Text"] = self.data["Text"].fillna("")
        if write:
            try:
                self.data.to_csv(clean_data_path)
            except:
                raise ValueError("clean_data_path has a non admissible input.")
    
    # Split divides the dataset into training and test sample. If write is
    # active, it writes the training and test data sets to the paths specified
    # in the training_data_path and test_data_path arguments
    
    def split(self, test_size: float, 
              training_data_path = None, 
              test_data_path = None,
              write = False):
        train, test = train_test_split(self.data, test_size = test_size)
        self.train_set = train
        self.test_set = test
        if write:
            try:
                train.to_csv(training_data_path)
                test.to_csv(test_data_path)
            except:
                raise ValueError("A data path argument has a non admissible input.")
        return None
    
   # Generate representation assigns to the relevant attributes of the class 
   # a vector representation of the text. The nature of this representation is
   #based on the representation flag.
    
    def generate_representation(self, representation: str):
        if representation in supported_representations:
            train_summary_values = self.train_set["Summary"].values
            test_summary_values = self.test_set["Summary"].values
            
            if representation == "count":
                count_vectorizer = CountVectorizer(input = "content")
                self.count_features = count_vectorizer.fit_transform(train_summary_values)
                self.count_test_features = count_vectorizer.transform(test_summary_values)
                
            if representation == "set":
                set_vectorizer = CountVectorizer(input = "content", binary = True)
                self.set_features = set_vectorizer.fit_transform(train_summary_values)
                self.set_test_features = set_vectorizer.transform(test_summary_values)                  
    
            if representation == "tfidf":
                tfidf_vectorizer = TfidfVectorizer(input = "content")
                self.tfidf_features = tfidf_vectorizer.fit_transform(train_summary_values)
                self.tfidf_test_features = tfidf_vectorizer.transform(test_summary_values) 
            return None
        else:
            raise ValueError("The representation is not supported.")




        
