import argparse
from collections import Counter
from operator import itemgetter
from random import shuffle

from sklearn import metrics
from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

from review_reader import ReviewReader


class ActiveReviewClassifier:
    def __init__(self, review_type, initial_train_size, algorithm, minimum_test_set_size, train_increment_size):
        username, password, host, database_name = self.get_db_credentials()
        database = ReviewReader(username, password, host, database_name)
        reviews_pos_cls, reviews_neg_cls = database.get_app_reviews(review_type)
        database.commit()
        database.close()

        data = reviews_pos_cls + reviews_neg_cls
        target = [1] * len(reviews_pos_cls)
        target += [0] * len(reviews_neg_cls)

        data_target = list(zip(data, target))
        data_target = self.shuffle_data(data_target, initial_train_size)
        data, target = self.split_data(data_target)

        train_data = data[:initial_train_size]
        train_target = target[:initial_train_size]

        test_data = data[initial_train_size:]
        test_target = target[initial_train_size:]

        train_size_list, precision_list, recall_list, f1_score_list, auc_list = self.train_classifier(train_data.copy(),
                                                                                                      train_target.copy(),
                                                                                                      test_data.copy(),
                                                                                                      test_target.copy(),
                                                                                                      algorithm,
                                                                                                      initial_train_size,
                                                                                                      minimum_test_set_size,
                                                                                                      train_increment_size,
                                                                                                      active_learning=False)

        print(train_size_list)
        print(precision_list)
        print(recall_list)
        print(f1_score_list)
        print(auc_list)

        train_size_list, precision_list, recall_list, f1_score_list, auc_list = self.train_classifier(train_data.copy(),
                                                                                                      train_target.copy(),
                                                                                                      test_data.copy(),
                                                                                                      test_target.copy(),
                                                                                                      algorithm,
                                                                                                      initial_train_size,
                                                                                                      minimum_test_set_size,
                                                                                                      train_increment_size,
                                                                                                      active_learning=True)

        print(train_size_list)
        print(precision_list)
        print(recall_list)
        print(f1_score_list)
        print(auc_list)

    def train_classifier(self, train_data, train_target, test_data, test_target, algorithm, initial_train_size,
                         minimum_test_set_size, train_increment_size, active_learning):
        train_size_list = []
        precision_list = []
        recall_list = []
        f1_score_list = []
        auc_list = []

        while len(test_data) >= minimum_test_set_size:
            tfidf_train_data, tfidf_test_data = self.vectorize_data(train_data, test_data)
            predicted_test_target, predicted_test_target_probabilities = self.classify_app_reviews(algorithm,
                                                                                                   tfidf_train_data,
                                                                                                   train_target,
                                                                                                   tfidf_test_data)
            precision, recall, f1_score, auc = self.calculate_classifier_performance_metrics(test_target,
                                                                                             predicted_test_target)

            if len(train_data) != initial_train_size:
                train_size_list.append(len(train_data))
                precision_list.append(precision)
                recall_list.append(recall)
                f1_score_list.append(f1_score)
                auc_list.append(auc)

                if len(train_data) == 380:
                    print(precision, ",", recall, ",", f1_score)

            if len(test_data) >= train_increment_size:
                number_of_rows_to_add = train_increment_size
            else:
                number_of_rows_to_add = len(test_data)

            if active_learning:
                abs_bug_probability_confidence = []
                for i in predicted_test_target_probabilities:
                    abs_bug_probability_confidence.append(abs(i[1] - 0.5))

                test_data_with_confidence_and_test_target = list(
                    zip(test_data, abs_bug_probability_confidence, test_target))
                test_data_with_confidence_and_test_target.sort(key=itemgetter(1))

                data_chosen_to_add_to_train_from_test, predicted_positive_probability, chosen_data_target = zip(
                    *test_data_with_confidence_and_test_target)

                pos_counter = 0
                for i in data_chosen_to_add_to_train_from_test:
                    if pos_counter < number_of_rows_to_add:
                        index = test_data.index(i)
                        train_data.append(test_data.pop(index))
                        train_target.append(test_target.pop(index))
                    pos_counter += 1
            else:
                for i in range(number_of_rows_to_add):
                    train_data.append(test_data.pop(0))
                    train_target.append(test_target.pop(0))

        return train_size_list, precision_list, recall_list, f1_score_list, auc_list

    def vectorize_data(self, data_train, data_test):
        vectorizer = TfidfVectorizer(binary=True, use_idf=False, norm=None)
        tfidf_train_data = vectorizer.fit_transform(data_train)
        tfidf_test_data = vectorizer.transform(data_test)
        return tfidf_train_data, tfidf_test_data

    def classify_app_reviews(self, algorithm, tfidf_train_data, target_train, tfidf_test_data):
        if algorithm == 'MultinomialNB':
            classifier = MultinomialNB()
        if algorithm == 'LogisticRegression':
            classifier = LogisticRegression()
        if algorithm == 'SVM':
            classifier = svm.SVC(probability=True, kernel='linear')
        classifier.fit(tfidf_train_data, target_train)
        predicted_target_test = classifier.predict(tfidf_test_data)
        predicted_test_target_probabilities = classifier.predict_proba(tfidf_test_data).tolist()
        return predicted_target_test, predicted_test_target_probabilities

    def calculate_classifier_performance_metrics(self, target_test, predicted_target_test):
        number_of_classes = 2
        precision = metrics.precision_score(target_test, predicted_target_test)
        recall = metrics.recall_score(target_test, predicted_target_test)
        f1_score = metrics.f1_score(target_test, predicted_target_test)
        if number_of_classes == len(Counter(target_test)):
            auc = metrics.roc_auc_score(target_test, predicted_target_test)
        else:
            auc = -1
        return precision, recall, f1_score, auc

    def split_data(self, train_data):
        data = []
        target = []

        for x in train_data:
            if x[1] == 1:
                data.append(x[0].review)
                target.append(1)
            else:
                data.append(x[0].review)
                target.append(0)

        return data, target

    def shuffle_data(self, data, initial_train_size):
        done = True
        while done:
            shuffle(data)
            l2 = []
            for i in data:
                if i[1] == 1:
                    l2.append(1)
                else:
                    l2.append(0)
            if int(0.45 * initial_train_size) < Counter(l2[:initial_train_size])[1] < int(
                    0.55 * initial_train_size):
                done = False
        return data

    def get_db_credentials(self):
        config_file = open("credentials.config", "r")
        lines = config_file.readlines()
        username = lines[0].split("=")[1].strip()
        password = lines[1].split("=")[1].strip()
        host = lines[2].split("=")[1].strip()
        database_name = lines[3].split("=")[1].strip()
        config_file.close()
        return username, password, host, database_name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', action="store", dest="review", help="Review Type", type=str)
    args = parser.parse_args()

    review_type = args.review
    initial_train_size = 100
    algorithm = "MultinomialNB"
    minimum_test_set_size = 10
    train_increment_size = 20
    active_review_classifier = ActiveReviewClassifier(review_type, initial_train_size, algorithm, minimum_test_set_size,
                                                      train_increment_size)


if __name__ == '__main__':
    main()
