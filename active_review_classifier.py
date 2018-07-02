import csv
import argparse
from random import shuffle
from collections import Counter
import constants
from sklearn import metrics
from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

from review_reader import ReviewReader


class ActiveReviewClassifier:
    def __init__(self, review_type, initial_train_size, algorithm, minimum_test_set_size, train_increment_size):
        self.review_type = review_type
        self.initial_train_size = initial_train_size
        self.algorithm = algorithm
        self.minimum_test_set_size = minimum_test_set_size
        self.train_increment_size = train_increment_size

        username, password, host, database_name = ActiveReviewClassifier.get_db_credentials()
        database = ReviewReader(username, password, host, database_name)
        self.reviews_pos_cls, self.reviews_neg_cls = database.get_app_reviews(self.review_type)
        database.close()

        self.baseline_results = dict()
        self.active_results = dict()

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
        shuffle(self.reviews_pos_cls)  # Shuffle data first
        shuffle(self.reviews_neg_cls)  # Shuffle data first

        self.baseline_results = self.run_experiments_one_iteration('baseline')
        print()
        self.active_results = self.run_experiments_one_iteration('active')

        return self.baseline_results, self.active_results

    def run_experiments_one_iteration(self, classfication_type):
        one_iteration_results = dict()

        training_reviews, training_reviews_classes, test_reviews, test_reviews_classes = self.get_initial_data()

        while len(test_reviews_classes) >= self.minimum_test_set_size:
            training_reviews_features, test_reviews_features = self.vectorize_reviews(training_reviews, test_reviews)
            # print('Initial train size: ',  training_reviews_features.shape, len(training_reviews_classes))
            # print('Initial test size: ', test_reviews_features.shape, len(test_reviews_classes))

            test_reviews_predicted_classes, test_reviews_predicted_class_probabilities = \
                self.classify_app_reviews(training_reviews_features, training_reviews_classes, test_reviews_features)

            precision, recall, f1_score = self.calculate_classifier_performance_metrics(
                test_reviews_classes, test_reviews_predicted_classes)

            # print('train size, precision, recall, f1_score: ', len(training_reviews), precision, recall, f1_score)
            one_iteration_results[len(training_reviews)] = [precision, recall, f1_score]

            if len(test_reviews_classes) >= self.train_increment_size:
                number_of_rows_to_add = self.train_increment_size
            else:
                number_of_rows_to_add = len(test_reviews_classes)

            if Counter(test_reviews_classes).get(1) > number_of_rows_to_add and \
               Counter(test_reviews_classes).get(0) > number_of_rows_to_add:
                if classfication_type == 'baseline':
                    self.update_training_test_sets_baseline(
                        training_reviews, training_reviews_classes, test_reviews, test_reviews_classes,
                        number_of_rows_to_add)
                elif classfication_type == 'active':
                    if Counter(test_reviews_predicted_classes).get(1) > number_of_rows_to_add and \
                       Counter(test_reviews_predicted_classes).get(0) > number_of_rows_to_add:
                        self.update_training_test_sets_active(
                            training_reviews, training_reviews_classes,
                            test_reviews, test_reviews_classes, number_of_rows_to_add,
                            test_reviews_predicted_classes, test_reviews_predicted_class_probabilities)
                    else:
                        break
                else:
                    print('Invalid classification type')
                    exit(-2)
            else:
                break

        return one_iteration_results

    def get_initial_data(self):
        initial_training_reviews = self.reviews_pos_cls[:self.initial_train_size] + \
                                   self.reviews_neg_cls[:self.initial_train_size]

        initial_training_classes = [1] * len(self.reviews_pos_cls[:self.initial_train_size]) + \
                                   [0] * len(self.reviews_neg_cls[:self.initial_train_size])

        initial_testing_reviews = self.reviews_pos_cls[self.initial_train_size:] + \
                                  self.reviews_neg_cls[self.initial_train_size:]

        initial_testing_classes = [1] * len(self.reviews_pos_cls[self.initial_train_size:]) + \
                                  [0] * len(self.reviews_neg_cls[self.initial_train_size:])

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
        precision = metrics.precision_score(test_reviews_classes, predicted_test_reviews_classes)
        recall = metrics.recall_score(test_reviews_classes, predicted_test_reviews_classes)
        f1_score = metrics.f1_score(test_reviews_classes, predicted_test_reviews_classes)
        return precision, recall, f1_score

    def update_training_test_sets_baseline(self, training_reviews, training_reviews_classes,
                                           test_reviews, test_reviews_classes, number_of_rows_to_add):
        for i in range(number_of_rows_to_add):
            # Add instances from the positive class
            training_reviews.append(test_reviews.pop(0))
            training_reviews_classes.append(test_reviews_classes.pop(0))

            # Add instances from the negative class
            training_reviews.append(test_reviews.pop(test_reviews_classes.index(0)))
            training_reviews_classes.append(test_reviews_classes.pop(test_reviews_classes.index(0)))

    def update_training_test_sets_active(self, training_reviews, training_reviews_classes, test_reviews,
                                         test_reviews_classes, number_of_rows_to_add, test_reviews_predicted_classes,
                                         test_reviews_predicted_class_probabilities):

        test_reviews_predicted_classes = test_reviews_predicted_classes.tolist()

        for i in range(len(test_reviews_predicted_class_probabilities)):
            test_reviews_predicted_class_probabilities[i] = abs(test_reviews_predicted_class_probabilities[i][1] - 0.5)
        positive_class_index_to_predicted_probabilities = dict()
        negative_class_index_to_predicted_probabilities = dict()
        for i in range(len(test_reviews_predicted_class_probabilities)):
            if test_reviews_predicted_classes[i] == 1:
                positive_class_index_to_predicted_probabilities[i] = test_reviews_predicted_class_probabilities[i]
            elif test_reviews_predicted_classes[i] == 0:
                negative_class_index_to_predicted_probabilities[i] = test_reviews_predicted_class_probabilities[i]

        number_of_pos_rows_added = 0
        pop_index_list = []
        for pos_index in sorted(positive_class_index_to_predicted_probabilities,
                                key=positive_class_index_to_predicted_probabilities.get):
            pop_index_list.append(pos_index)
            training_reviews_classes.append(test_reviews_classes[pos_index])
            training_reviews.append(test_reviews[pos_index])

            number_of_pos_rows_added += 1
            if number_of_pos_rows_added >= number_of_rows_to_add:
                break

        number_of_neg_rows_added = 0
        for neg_index in sorted(negative_class_index_to_predicted_probabilities,
                                key=negative_class_index_to_predicted_probabilities.get):
            pop_index_list.append(neg_index)
            training_reviews_classes.append(test_reviews_classes[neg_index])
            training_reviews.append(test_reviews[neg_index])
            number_of_neg_rows_added += 1
            if number_of_neg_rows_added >= number_of_rows_to_add:
                break

        pop_index_list.sort(reverse=True)
        for i in pop_index_list:
            test_reviews_classes.pop(i)
            test_reviews.pop(i)


def write_results_csv(outfile_name, runs_results):
    with open(outfile_name, 'w', newline='') as outfile:
        csv_writer = csv.writer(outfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)

        for run_index in range(len(runs_results)):
            run_results = runs_results[run_index]
            for key, value in run_results.items():
                row_to_write = list()
                row_to_write.append(run_index)
                row_to_write.append(key)
                for metric_value in range(len(value)):
                    row_to_write.append(value[metric_value])
                csv_writer.writerow(row_to_write)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', action="store", dest="review_type", help="Review Type", type=str)
    args = parser.parse_args()

    initial_train_size = 50
    algorithm = "MultinomialNB"
    minimum_test_set_size = 20  # minimum_test_set_size should be at least twice the amount of train_increment_size
    train_increment_size = 10
    num_of_runs = 30

    baseline_runs_results = [None] * num_of_runs
    active_runs_results = [None] * num_of_runs

    for i in range(num_of_runs):
        active_review_classifier = ActiveReviewClassifier(
            args.review_type, initial_train_size, algorithm, minimum_test_set_size, train_increment_size)

        baseline_runs_results[i], active_runs_results[i] = active_review_classifier.run_experiments()

    write_results_csv(constants.BINARY_BASELINE_OUTFILE, baseline_runs_results)
    write_results_csv(constants.BINARY_ACTIVE_OUTFILE, active_runs_results)
