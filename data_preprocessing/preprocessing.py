import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn_pandas import CategoricalImputer

class Preprocessor:
    """
        This class shall  be used to clean and transform the data before training.

        Written By: Vaishali Parashar
        Version: 1.0
        Revisions: None

        """

    def __init__(self, file_object, logger_object):
        self.file_object = file_object
        self.logger_object = logger_object

    def is_null_present(self,data):
        """
                                Method Name: is_null_present
                                Description: This method checks whether there are null values present in the pandas Dataframe or not.
                                Output: Returns True if null values are present in the DataFrame, False if they are not present and
                                        returns the list of columns for which null values are present.
                                On Failure: Raise Exception

                                Written By: Vaishali Parashar
                                Version: 1.0
                                Revisions: None

                        """
        self.logger_object.log(self.file_object, 'Entered the is_null_present method of the Preprocessor class')
        self.null_present = False
        self.cols_with_missing_values=[]
        self.cols = data.columns
        try:
            self.null_counts=data.isna().sum() # check for the count of null values per column
            for i in range(len(self.null_counts)):
                if self.null_counts[i]>0:
                    self.null_present=True
                    self.cols_with_missing_values.append(self.cols[i])
            if(self.null_present): # write the logs to see which columns have null values
                self.dataframe_with_null = pd.DataFrame()
                self.dataframe_with_null['columns'] = data.columns
                self.dataframe_with_null['missing values count'] = np.asarray(data.isna().sum())
                self.dataframe_with_null.to_csv('preprocessing_data/null_values.csv') # storing the null column information to file
            self.logger_object.log(self.file_object,
                                   'Finding missing values is a success.Data written to the null values file. Exited the is_null_present method of the Preprocessor class')
            return self.null_present, self.cols_with_missing_values
        except Exception as e:
            self.logger_object.log(self.file_object,
                                   'Exception occured in is_null_present method of the Preprocessor class. Exception message:  ' + str(e))
            self.logger_object.log(self.file_object,
                                   'Finding missing values failed. Exited the is_null_present method of the Preprocessor class')
            raise Exception()

    def impute_missing_values(self, data, cols_with_missing_values):
        """
                                        Method Name: impute_missing_values
                                        Description: This method replaces all the missing values in the Dataframe using KNN Imputer.
                                        Output: A Dataframe which has all the missing values imputed.
                                        On Failure: Raise Exception

                                        Written By: Vaishali Parashar
                                        Version: 1.0
                                        Revisions: None
                     """
        self.logger_object.log(self.file_object, 'Entered the impute_missing_values method of the Preprocessor class')
        self.data= data
        self.cols_with_missing_values = cols_with_missing_values
        try:
            self.imputer = CategoricalImputer()
            for col in self.cols_with_missing_values:
                self.data[col] = self.imputer.fit_transform(self.data[col])
            self.logger_object.log(self.file_object,
                                   'Imputing missing values Successful. Exited the impute_missing_values method of the Preprocessor class')
            return self.data
        except Exception as e:
            self.logger_object.log(self.file_object,
                                   'Exception occured in impute_missing_values method of the Preprocessor class. Exception message:  ' + str(e))
            self.logger_object.log(self.file_object,
                                   'Imputing missing values failed. Exited the impute_missing_values method of the Preprocessor class')
            raise Exception()

    def select_useful_data(self,data):
        """
                                    Method Name: select_useful_data
                                    Description: This method selects useful data from the whole dataset.
                                    As during EDA we came to know that fraud only occurs in 'TRANSFER's and
                                    'CASH_OUT's. So we assemble only the corresponding data in data
                                     for analysis.

                                    On Failure: Raise Exception

                                    Written By: Vaishali Parashar
                                    Version: 1.0
                                    Revisions: None
                             """
        self.logger_object.log(self.file_object, 'Entered the select_useful_data method of the Preprocessor class')
        self.data = data
        try:
            self.data = self.data.loc[(data.type == 'TRANSFER') | (data.type == 'CASH_OUT')]# selected only the data where type column contains 'TRANSFER's and 'CASH_OUT's.
            self.logger_object.log(self.file_object,
                                   'Data Selection successful.Exited the select_useful_data method of the preprocessor class')
            return self.data
        except Exception as e:
            self.logger_object.log(self.file_object,'Exception occured in select_useful_data method of the Preprocessor class. Exception message:  '+str(e))
            self.logger_object.log(self.file_object,
                                   'Data selection Unsuccessful. Exited the select_useful_data method of the Preprocessor class')
            raise Exception()

    def remove_columns(self,data,columns):
        """
                            Method Name: remove_columns
                            Description: This method removes the given columns from a pandas dataframe.
                            Output: A pandas DataFrame after removing the specified columns.
                            On Failure: Raise Exception

                            Written By: Vaishali Parashar
                            Version: 1.0
                            Revisions: None

        """
        self.logger_object.log(self.file_object, 'Entered the remove_columns method of the Preprocessor class')
        self.data = data
        self.columns = columns
        try:
            self.useful_data=self.data.drop(labels=self.columns, axis=1) # drop the labels specified in the columns
            self.logger_object.log(self.file_object,
                                   'Column removal Successful.Exited the remove_columns method of the Preprocessor class')
            return self.useful_data
        except Exception as e:
            self.logger_object.log(self.file_object,'Exception occured in remove_columns method of the Preprocessor class. Exception message:  '+str(e))
            self.logger_object.log(self.file_object,
                                   'Column removal Unsuccessful. Exited the remove_columns method of the Preprocessor class')
            raise Exception()

    def encode_categorical_columns(self, data):
        """
                                Method Name: encode_categorical_columns
                                Description: This method encodes the categorical values to numeric values.
                                Output: dataframe with categorical values converted to numerical values
                                On Failure: Raise Exception

                                Written By: Vaishali Parashar
                                Version: 1.0
                                Revisions: None
                                """
        self.logger_object.log(self.file_object,
                               'Entered the encode_categorical_columns method of the Preprocessor class')

        self.data = data
        try:
            data['type'] = np.where(data['type'] == 'TRANSFER', 0, 1)

            data['type'] = data['type'].astype(int)  # convert data_type('O') to data_type(int)

            self.logger_object.log(self.file_object,
                                   'Encoding for categorical values successful. Exited the encode_categorical_columns method of the Preprocessor class')

            return self.data
        except Exception as e:
            self.logger_object.log(self.file_object,'Exception occurred in encode_categorical_columns method of the Preprocessor class. Exception message:  ' + str(e))
            self.logger_object.log(self.file_object, 'encoding for categorical columns Failed. Exited the encode_categorical_columns method of the Preprocessor class')
            raise Exception()

    def create_new_columns(self,data):
        """
                                       Method Name: create_new_columns
                                       Description: This method creates new columns from the existing columns.
                                       On Failure: Raise Exception

                                       Written By: Vaishali Parashar
                                       Version: 1.0
                                       Revisions: None
                                       """
        self.logger_object.log(self.file_object,
                               'Entered the create_new_columns method of the Preprocessor class')

        self.data = data
        try:
            data['errorBalanceOrg'] = data.newbalanceOrig + data.amount - data.oldbalanceOrg
            data['errorBalanceDest'] = data.oldbalanceDest + data.amount - data.newbalanceDest
            self.logger_object.log(self.file_object,
                               'Creating new columns successful. Exited the create_new_columns method of the Preprocessor class')

            return self.data

        except Exception as e:
            self.logger_object.log(self.file_object,
                               'Exception occured in create_new_columns method of the Preprocessor class. Exception message:  ' + str(
                                   e))
            self.logger_object.log(self.file_object,
                               'Creating new columns Failed. Exited the create_new_columns method of the Preprocessor class')
            raise Exception()

    def separate_label_feature(self, data, label_column_name):
        """
                        Method Name: separate_label_feature
                        Description: This method separates the features and a Label Columns.
                        Output: Returns two separate Dataframes, one containing features and the other containing Labels .
                        On Failure: Raise Exception

                        Written By: Vaishali Parashar
                        Version: 1.0
                        Revisions: None

                """
        self.logger_object.log(self.file_object, 'Entered the separate_label_feature method of the Preprocessor class')
        try:
            self.X = data.drop(labels=label_column_name,axis=1)# drop the columns specified and separate the feature columns
            self.Y = data[label_column_name] # Filter the Label columns
            self.logger_object.log(self.file_object,
                                   'Label Separation Successful. Exited the separate_label_feature method of the Preprocessor class')
            return self.X, self.Y
        except Exception as e:
            self.logger_object.log(self.file_object,'Exception occured in separate_label_feature method of the Preprocessor class. Exception message:  ' + str(e))
            self.logger_object.log(self.file_object, 'Label Separation Unsuccessful. Exited the separate_label_feature method of the Preprocessor class')
            raise Exception()

    def scale_numerical_columns(self, X_train, X_test):
        """
                            Method Name: scale_numerical_columns
                            Description: This method scales the X_train and X_test data using the Standard scaler.
                            Output: X_train and X_test with scaled values
                            On Failure: Raise Exception

                            Written By: Vaishali Parashar
                            Version: 1.0
                            Revisions: None
                                     """
        self.logger_object.log(self.file_object,
                               'Entered the scale_numerical_columns method of the Preprocessor class')

        self.X_train = X_train
        self.X_test = X_test

        try:

            self.scale = StandardScaler()
            self.scaledX_train = self.scale.fit_transform(X_train)
            self.scaledX_test = self.scale.transform(X_test)

            self.logger_object.log(self.file_object,
                                   'scaling for numerical values successful. Exited the scale_numerical_columns method of the Preprocessor class')
            return self.scaledX_train, self.scaledX_test

        except Exception as e:
            self.logger_object.log(self.file_object,
                                   'Exception occured in scale_numerical_columns method of the Preprocessor class. Exception message:  ' + str(
                                       e))
            self.logger_object.log(self.file_object,
                                   'scaling for numerical columns Failed. Exited the scale_numerical_columns method of the Preprocessor class')
            raise Exception()