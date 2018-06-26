import argparse
from collections import Counter
from random import shuffle

from sklearn import metrics
from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

from review_reader import ReviewReader


class ActiveMultiClassClassifier:
    def __init__(self, review_type, initial_train_size, algorithm, minimum_test_set_size, train_increment_size):
        self.review_type = review_type
        self.initial_train_size = initial_train_size
        self.algorithm = algorithm
        self.minimum_test_set_size = minimum_test_set_size
        self.train_increment_size = train_increment_size

        username, password, host, database_name = ActiveMultiClassClassifier.get_db_credentials()
        database = ReviewReader(username, password, host, database_name)
        self.bug_reviews, self.feature_reviews, self.rating_reviews, self.userexperience_reviews = \
            database.get_app_reviews_for_multi_class(self.review_type)
        database.close()

    @staticmethod
    def get_db_credentials():
        config_file = open("credentials.config", "r")  # Filename should be a constant
        lines = config_file.readlines()
        username = lines[0].split("=")[1].strip()
        password = lines[1].split("=")[1].strip()
        host = lines[2].split("=")[1].strip()
        database_name = lines[3].split("=")[1].strip()
        config_file.close()
        return username, password, host, database_name

    def run_experiments(self):
        shuffle(self.bug_reviews)  # Shuffle data first
        shuffle(self.feature_reviews)  # Shuffle data first
        shuffle(self.rating_reviews)  # Shuffle data first
        shuffle(self.userexperience_reviews)  # Shuffle data first

        self.run_experiments_one_iteration('baseline')
        print()
        self.run_experiments_one_iteration('active')

    def run_experiments_one_iteration(self, classfication_type):
        training_reviews, training_reviews_classes, test_reviews, test_reviews_classes = self.get_initial_data()

        training_reviews_features, test_reviews_features = self.vectorize_reviews(training_reviews, test_reviews)
        # print('Initial train size: ',  training_reviews_features.shape, len(training_reviews_classes))
        # print('Initial test size: ', test_reviews_features.shape, len(test_reviews_classes))

        while len(test_reviews_classes) >= self.minimum_test_set_size:
            training_reviews_features, test_reviews_features = self.vectorize_reviews(training_reviews, test_reviews)
            print('Initial train size: ', training_reviews_features.shape, len(training_reviews_classes))
            # print('Initial test size: ', test_reviews_features.shape, len(test_reviews_classes))

            test_reviews_predicted_classes, test_reviews_predicted_class_probabilities = \
                self.classify_app_reviews(training_reviews_features, training_reviews_classes, test_reviews_features)

            precision, recall, f1_score, macro = self.calculate_classifier_performance_metrics(
                test_reviews_classes, test_reviews_predicted_classes)

            print('precision, recall, f1_score, macro: ', precision, recall, f1_score, macro)

            if len(test_reviews_classes) >= self.train_increment_size:
                number_of_rows_to_add = self.train_increment_size
            else:
                number_of_rows_to_add = len(test_reviews_classes)

            if Counter(test_reviews_classes).get(1) > number_of_rows_to_add and \
               Counter(test_reviews_classes).get(3) > number_of_rows_to_add and \
               Counter(test_reviews_classes).get(5) > number_of_rows_to_add and \
               Counter(test_reviews_classes).get(7) > number_of_rows_to_add:
                if classfication_type == 'baseline':
                    self.update_training_test_sets_baseline(
                        training_reviews, training_reviews_classes, test_reviews, test_reviews_classes,
                        number_of_rows_to_add)
                elif classfication_type == 'active':
                    if Counter(test_reviews_predicted_classes).get(1) > number_of_rows_to_add and \
                       Counter(test_reviews_predicted_classes).get(3) > number_of_rows_to_add and \
                       Counter(test_reviews_predicted_classes).get(5) > number_of_rows_to_add and \
                       Counter(test_reviews_predicted_classes).get(7) > number_of_rows_to_add:
                        self.update_training_test_sets_active(
                            training_reviews, training_reviews_classes, test_reviews, test_reviews_classes,
                            number_of_rows_to_add, test_reviews_predicted_classes, test_reviews_predicted_class_probabilities)
                    else:
                        break
                else:
                    print('Invalid classification type')
                    exit(-2)
            else:
                break

    def get_initial_data(self):
        initial_training_reviews = self.bug_reviews[:self.initial_train_size] + \
                                   self.feature_reviews[:self.initial_train_size] + \
                                   self.rating_reviews[:self.initial_train_size] + \
                                   self.userexperience_reviews[:self.initial_train_size]

        initial_training_classes = [1] * len(self.bug_reviews[:self.initial_train_size]) + \
                                   [3] * len(self.feature_reviews[:self.initial_train_size]) + \
                                   [5] * len(self.rating_reviews[:self.initial_train_size]) + \
                                   [7] * len(self.userexperience_reviews[:self.initial_train_size])

        initial_testing_reviews = self.bug_reviews[self.initial_train_size:] + \
                                  self.feature_reviews[self.initial_train_size:] + \
                                  self.rating_reviews[self.initial_train_size:] + \
                                  self.userexperience_reviews[self.initial_train_size:]

        initial_testing_classes = [1] * len(self.bug_reviews[self.initial_train_size:]) + \
                                  [3] * len(self.feature_reviews[self.initial_train_size:]) + \
                                  [5] * len(self.rating_reviews[self.initial_train_size:]) + \
                                  [7] * len(self.userexperience_reviews[self.initial_train_size:])

        # initial_training_features, initial_test_features = self.vectorize_reviews(
        #     initial_training_reviews, initial_testing_reviews)

        return initial_training_reviews, initial_training_classes, initial_testing_reviews, initial_testing_classes

    def vectorize_reviews(self, train_reviews, test_reviews):
        vectorizer = TfidfVectorizer(binary=True, use_idf=False, norm=None)  # Bag of words
        traing_reviews_features = vectorizer.fit_transform(train_reviews)
        test_reviews_features = vectorizer.transform(test_reviews)
        return traing_reviews_features, test_reviews_features

    def classify_app_reviews(self, train_reviews_features, train_reviews_classes, test_reviews_features):
        classifier = self.get_classifier()
        classifier.fit(train_reviews_features, train_reviews_classes)
        test_reviews_predicted_classes = classifier.predict(test_reviews_features)
        test_reviews_predicted_class_probabilities = classifier.predict_proba(test_reviews_features).tolist()
        return test_reviews_predicted_classes, test_reviews_predicted_class_probabilities

    def get_classifier(self):
        if self.algorithm == 'MultinomialNB':
            return MultinomialNB()
        elif self.algorithm == 'LogisticRegression':
            return LogisticRegression()
        elif self.algorithm == 'SVM':
            return svm.SVC(probability=True, kernel='linear')
        else:
            print('Classifier ' + self.algorithm + ' not supported')
            exit(-1)

    def calculate_classifier_performance_metrics(self, test_reviews_classes, predicted_test_reviews_classes):
        precision = metrics.precision_score(test_reviews_classes, predicted_test_reviews_classes, average=None)
        recall = metrics.recall_score(test_reviews_classes, predicted_test_reviews_classes, average=None)
        f1_score = metrics.f1_score(test_reviews_classes, predicted_test_reviews_classes, average=None)
        macro = metrics.precision_score(test_reviews_classes, predicted_test_reviews_classes, average='macro')
        return precision, recall, f1_score, macro

    def update_training_test_sets_baseline(self, training_reviews, training_reviews_classes,
                                           test_reviews, test_reviews_classes, number_of_rows_to_add):
        for i in range(number_of_rows_to_add):
            # Add instances from the bug class
            training_reviews.append(test_reviews.pop(0))
            training_reviews_classes.append(test_reviews_classes.pop(0))

            # Add instances from the feature class
            training_reviews.append(test_reviews.pop(test_reviews_classes.index(3)))
            training_reviews_classes.append(test_reviews_classes.pop(test_reviews_classes.index(3)))

            # Add instances from the rating class
            training_reviews.append(test_reviews.pop(test_reviews_classes.index(5)))
            training_reviews_classes.append(test_reviews_classes.pop(test_reviews_classes.index(5)))

            # Add instances from the user experience class
            training_reviews.append(test_reviews.pop(test_reviews_classes.index(7)))
            training_reviews_classes.append(test_reviews_classes.pop(test_reviews_classes.index(7)))

    def update_training_test_sets_active(self, training_reviews, training_reviews_classes, test_reviews,
                                         test_reviews_classes, number_of_rows_to_add, test_reviews_predicted_classes,
                                         test_reviews_predicted_class_probabilities):

        test_reviews_predicted_classes = test_reviews_predicted_classes.tolist()

        # for i in range(len(test_reviews_predicted_class_probabilities)):
        #     test_reviews_predicted_class_probabilities[i] = abs(test_reviews_predicted_class_probabilities[i][1] - 0.5)

        self.calculate_least_confident_probabilities(test_reviews_predicted_class_probabilities)

        bug_class_index_to_predicted_probabilities = dict()
        feature_class_index_to_predicted_probabilities = dict()
        rating_class_index_to_predicted_probabilities = dict()
        user_experience_class_index_to_predicted_probabilities = dict()

        for i in range(len(test_reviews_predicted_class_probabilities)):
            if test_reviews_predicted_classes[i] == 1:
                bug_class_index_to_predicted_probabilities[i] = test_reviews_predicted_class_probabilities[i]
            elif test_reviews_predicted_classes[i] == 3:
                feature_class_index_to_predicted_probabilities[i] = test_reviews_predicted_class_probabilities[i]
            elif test_reviews_predicted_classes[i] == 5:
                rating_class_index_to_predicted_probabilities[i] = test_reviews_predicted_class_probabilities[i]
            elif test_reviews_predicted_classes[i] == 7:
                user_experience_class_index_to_predicted_probabilities[i] = test_reviews_predicted_class_probabilities[i]

        pop_index_list = []

        number_of_bug_rows_added = 0
        for bug_index in sorted(bug_class_index_to_predicted_probabilities,
                                key=bug_class_index_to_predicted_probabilities.get):
            pop_index_list.append(bug_index)
            training_reviews_classes.append(test_reviews_classes[bug_index])
            training_reviews.append(test_reviews[bug_index])

            number_of_bug_rows_added += 1
            if number_of_bug_rows_added >= number_of_rows_to_add:
                break

        number_of_feature_rows_added = 0
        for feature_index in sorted(feature_class_index_to_predicted_probabilities,
                                    key=feature_class_index_to_predicted_probabilities.get):
            pop_index_list.append(feature_index)
            training_reviews_classes.append(test_reviews_classes[feature_index])
            training_reviews.append(test_reviews[feature_index])
            number_of_feature_rows_added += 1
            if number_of_feature_rows_added >= number_of_rows_to_add:
                break

        number_of_rating_rows_added = 0
        for rating_index in sorted(rating_class_index_to_predicted_probabilities,
                                   key=rating_class_index_to_predicted_probabilities.get):
            pop_index_list.append(rating_index)
            training_reviews_classes.append(test_reviews_classes[rating_index])
            training_reviews.append(test_reviews[rating_index])
            number_of_rating_rows_added += 1
            if number_of_rating_rows_added >= number_of_rows_to_add:
                break

        number_of_user_experience_rows_added = 0
        for user_experience_index in sorted(user_experience_class_index_to_predicted_probabilities,
                                            key=user_experience_class_index_to_predicted_probabilities.get):
            pop_index_list.append(user_experience_index)
            training_reviews_classes.append(test_reviews_classes[user_experience_index])
            training_reviews.append(test_reviews[user_experience_index])
            number_of_user_experience_rows_added += 1
            if number_of_user_experience_rows_added >= number_of_rows_to_add:
                break

        pop_index_list.sort(reverse=True)
        for i in pop_index_list:
            test_reviews_classes.pop(i)
            test_reviews.pop(i)

    def calculate_least_confident_probabilities(self, test_reviews_predicted_class_probabilities):

        for i in range(len(test_reviews_predicted_class_probabilities)):
            one = 1 - test_reviews_predicted_class_probabilities[i][0]
            three = 1 - test_reviews_predicted_class_probabilities[i][1]
            five = 1 - test_reviews_predicted_class_probabilities[i][2]
            seven = 1 - test_reviews_predicted_class_probabilities[i][3]

            test_reviews_predicted_class_probabilities[i] = max(one, three, five, seven)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', action="store", dest="review_type", help="Review Type", type=str)
    args = parser.parse_args()

    initial_train_size = 100
    algorithm = "MultinomialNB"
    minimum_test_set_size = 80  # minimum_test_set_size should be at least twice the amount of train_increment_size
    train_increment_size = 10

    active_review_classifier = ActiveMultiClassClassifier(
        args.review_type, initial_train_size, algorithm, minimum_test_set_size, train_increment_size)

    active_review_classifier.run_experiments()
