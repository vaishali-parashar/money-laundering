"""
This is the Entry point for Training the Machine Learning Model.

Written By: Vaishali Parashar
Version: 1.0
Revisions: None

"""

# Doing the necessary imports
import pandas as pd
from sklearn.model_selection import train_test_split
from application_logging import logger
from best_model_finder import tuner
from data_ingestion import data_loader
from data_preprocessing import preprocessing

# Creating the common Logging object
from file_operations import file_methods


class trainModel:

    def __init__(self):
        self.log_writer = logger.App_Logger()
        self.file_object = open("Training_Logs/ModelTrainingLog.txt", 'a+')

    def trainingModel(self):
        # Logging the start of Training
        self.log_writer.log(self.file_object, 'Start of Training')
        try:
            # Getting the data from the source
            data_getter = data_loader.Data_Getter(self.file_object, self.log_writer)
            data = data_getter.get_data()

            """doing the data preprocessing"""
            self.log_writer.log(self.file_object, 'Start of data preprocessing')

            preprocessor = preprocessing.Preprocessor(self.file_object, self.log_writer)
            # check if missing values are present in the dataset
            is_null_present, cols_with_missing_values = preprocessor.is_null_present(data)

            # if missing values are there, replace them appropriately.
            if is_null_present:
                data = preprocessor.impute_missing_values(data, cols_with_missing_values)  # missing value imputation

            # selecting useful data
            data = preprocessor.select_useful_data(data)

            # remove unwanted columns.

            data = preprocessor.remove_columns(data, ['step', 'nameOrig', 'nameDest', 'isFlaggedFraud'])

            # encode categorical data
            data = preprocessor.encode_categorical_columns(data)

            # creating new columns
            data = preprocessor.create_new_columns(data)

            # create separate features and labels
            X, Y = preprocessor.separate_label_feature(data, label_column_name='isFraud')

            # Split data into training and test set
            X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=35)

            # Scaling down numerical columns
            scaledX_train, scaledX_test = preprocessor.scale_numerical_columns(X_train, X_test)

            # converting scaledX_test to dataframe and then to csv file
            test = pd.DataFrame(scaledX_test)
            test.to_csv('testdata.csv', index=False)

            self.log_writer.log(self.file_object, 'Successful End of data preprocessing')

            # entering the model building stage
            self.log_writer.log(self.file_object, 'Starting the Model building stage')

            model_finder = tuner.Model_Finder(self.file_object, self.log_writer)  # object initialization

            # getting the best model for each of the clusters
            best_model_name, best_model = model_finder.get_best_model(scaledX_train, y_train, scaledX_test, y_test)

            # saving the best model to the directory.
            file_op = file_methods.File_Operation(self.file_object, self.log_writer)
            file_op.save_model(best_model, best_model_name)

            # logging the successful Training
            self.log_writer.log(self.file_object, 'Successful End of Training')
            self.file_object.close()

        except Exception as e:
            # logging the unsuccessful Training
            self.log_writer.log(self.file_object, 'Unsuccessful End of Training')
            self.file_object.close()
            raise Exception
