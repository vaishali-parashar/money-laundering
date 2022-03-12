from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, accuracy_score


class Model_Finder:
    """
                This class shall  be used to find the model with the best accuracy and AUC score.
                Written By: Vaishali Parashar
                Version: 1.0
                Revisions: None

                """

    def __init__(self, file_object, logger_object):
        self.file_object = file_object
        self.logger_object = logger_object
        self.dt_classifier = DecisionTreeClassifier()
        self.xgb = XGBClassifier(objective='binary:logistic', n_jobs=-1)

    def get_best_params_for_decisiontree(self, train_scaledx, train_y):
        """
        Method Name: get_best_params_for_decisiontree
        Description: get the parameters for the decision tree Algorithm which give the best accuracy.
                     Use Hyper Parameter Tuning.
        Output: The model with the best parameters
        On Failure: Raise Exception

        Written By: Vaishali Parashar
        Version: 1.0
        Revisions: None

                        """
        self.logger_object.log(self.file_object,
                               'Entered the get_best_params_for_decisiontree method of the Model_Finder class')
        try:
            # initializing with different combination of parameters
            self.param_grid = {"max_depth": [2, 4, 6, 8, 10],
                               "criterion": ['gini', 'entropy'],
                               "min_samples_leaf": [1, 2, 3, 4, 5]}

            # Creating an object of the Grid Search class
            self.grid = GridSearchCV(estimator=self.dt_classifier, param_grid=self.param_grid, cv=5, verbose=3)
            # finding the best parameters
            self.grid.fit(train_scaledx, train_y)

            # extracting the best parameters
            self.max_depth = self.grid.best_params_['max_depth']
            self.criterion = self.grid.best_params_['criterion']
            self.min_samples_leaf = self.grid.best_params_['min_samples_leaf']

            # creating a new model with the best parameters
            self.dt_classifier = DecisionTreeClassifier(max_depth=self.max_depth, criterion=self.criterion,
                                                        min_samples_leaf=self.min_samples_leaf)
            # training the mew model
            self.dt_classifier.fit(train_scaledx, train_y)
            self.logger_object.log(self.file_object,
                                   'DecisionTree best params: ' + str(
                                       self.grid.best_params_) + '. Exited the get_best_params_for_decisiontree method of the Model_Finder class')

            return self.dt_classifier
        except Exception as e:
            self.logger_object.log(self.file_object,
                                   'Exception occured in get_best_params_for_decisiontree method of the Model_Finder class. Exception message:  ' + str(
                                       e))
            self.logger_object.log(self.file_object,
                                   'SVM training  failed. Exited the get_best_params_for_decisiontree method of the Model_Finder class')
            raise Exception()

    def get_best_params_for_xgboost(self, train_scaledx, train_y):

        """
                                    Method Name: get_best_params_for_xgboost
                                    Description: get the parameters for XGBoost Algorithm which give the best accuracy.
                                                 Use Hyper Parameter Tuning.
                                    Output: The model with the best parameters
                                    On Failure: Raise Exception

                                    Written By: Vaishali Parashar
                                    Version: 1.0
                                    Revisions: None

                                """
        self.logger_object.log(self.file_object,
                               'Entered the get_best_params_for_xgboost method of the Model_Finder class')
        try:
            # initializing with different combination of parameters
            self.param_grid_xgboost = {"n_estimators": [50, 75, 100],
                                       "learning_rate": [0.1, 0.3, 0.5, 0.7],
                                       "max_depth": [3, 4, 5, 6]}
            # Creating an object of the Grid Search class
            self.grid = GridSearchCV(XGBClassifier(objective='binary:logistic'), self.param_grid_xgboost, verbose=3,
                                     cv=5)
            # finding the best parameters
            self.grid.fit(train_scaledx, train_y)

            # extracting the best parameters
            self.n_estimators = self.grid.best_params_['n_estimators']
            self.learning_rate = self.grid.best_params_['learning_rate']
            self.max_depth = self.grid.best_params_['max_depth']


            # creating a new model with the best parameters
            self.xgb = XGBClassifier(learning_rate=self.learning_rate, max_depth=self.max_depth,
                                     n_estimators=self.n_estimators,
                                     n_jobs=-1)
            # training the mew model
            self.xgb.fit(train_scaledx, train_y)
            self.logger_object.log(self.file_object,
                                   'XGBoost best params: ' + str(
                                       self.grid.best_params_) + '. Exited the get_best_params_for_xgboost method of the Model_Finder class')
            return self.xgb

        except Exception as e:
            self.logger_object.log(self.file_object,
                                   'Exception occured in get_best_params_for_xgboost method of the Model_Finder class. Exception message:  ' + str(
                                       e))
            self.logger_object.log(self.file_object,
                                   'XGBoost Parameter tuning  failed. Exited the get_best_params_for_xgboost method of the Model_Finder class')
            raise Exception()

    def get_best_model(self, train_scaledx, train_y, test_scaledx, test_y):
        """
                                                Method Name: get_best_model
                                                Description: Find out the Model which has the best AUC score.
                                                Output: The best model name and the model object
                                                On Failure: Raise Exception

                                                Written By: Vaishali Parashar
                                                Version: 1.0
                                                Revisions: None

                                        """
        self.logger_object.log(self.file_object,
                               'Entered the get_best_model method of the Model_Finder class')
        # create best model for XGBoost
        try:
            self.xgboost = self.get_best_params_for_xgboost(train_scaledx, train_y)
            self.prediction_xgboost = self.xgboost.predict(test_scaledx)  # Predictions using the XGBoost Model

            if len(test_y.unique()) == 1:  # if there is only one label in y, then roc_auc_score returns error. We will use accuracy in that case
                self.xgboost_score = accuracy_score(test_y, self.prediction_xgboost)
                self.logger_object.log(self.file_object, 'Accuracy for XGBoost:' + str(self.xgboost_score))  # Log AUC
            else:
                self.xgboost_score = roc_auc_score(test_y, self.prediction_xgboost)  # AUC for XGBoost
                self.logger_object.log(self.file_object, 'AUC for XGBoost:' + str(self.xgboost_score))  # Log AUC

            # create best model for Decision Tree
            self.decisiontree = self.get_best_params_for_decisiontree(train_scaledx, train_y)
            self.prediction_decisiontree = self.decisiontree.predict(test_scaledx)  # prediction using the DecisionTree Algorithm

            if len(test_y.unique()) == 1:  # if there is only one label in y, then roc_auc_score returns error. We will use accuracy in that case
                self.decisiontree_score = accuracy_score(test_y, self.prediction_decisiontree)
                self.logger_object.log(self.file_object, 'Accuracy for DecisionTree:' + str(self.decisiontree_score))
            else:
                self.decisiontree_score = roc_auc_score(test_y, self.prediction_decisiontree)  # AUC for Random Forest
                self.logger_object.log(self.file_object, 'AUC for DecisionTree:' + str(self.decisiontree_score))

            # comparing the two models
            if self.decisiontree_score < self.xgboost_score:
                return 'XGBoost', self.xgboost
            else:
                return 'DecisionTree', self.decisiontree

        except Exception as e:
            self.logger_object.log(self.file_object,
                                   'Exception occured in get_best_model method of the Model_Finder class.'
                                   ' Exception message:  ' + str(e))
            self.logger_object.log(self.file_object,
                                   'Model Selection Failed. Exited the get_best_model method of the Model_Finder class')
            raise Exception()
