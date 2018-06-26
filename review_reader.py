import MySQLdb

import constants
from review import Review


class ReviewReader:
    def __init__(self, username, password, host, database_name):
        try:
            db = MySQLdb.connect(host=host, user=username, passwd=password, db=database_name)
            db.autocommit(True)
            db.begin()
            cur = db.cursor()
            self.db = db
            self.cur = cur
            print("Database connection established successfully")

        except MySQLdb.Error as err:
            print('Database connection failed for ' + username + '@' + host + '/' + database_name)
            exit()

    def get_app_reviews(self, review_type):
        reviews_pos_cls = []
        reviews_neg_cls = []

        table_names = self.get_table_names(review_type)

        for i in table_names:
            sql = "SELECT * FROM " + i
            self.cur.execute(sql)

            for row in self.cur.fetchall():
                app_review = str(row[17])
                review = Review(app_review)
                if i.startswith('not'):
                    reviews_neg_cls.append(app_review)
                else:
                    reviews_pos_cls.append(app_review)

        return reviews_pos_cls, reviews_neg_cls

    def get_app_reviews_for_multi_class(self, review_type):
        bug_reviews = []
        feature_reviews = []
        rating_reviews = []
        userexperience_reviews = []

        table_names = self.get_table_names(review_type)

        for i in table_names:
            sql = "SELECT * FROM " + i
            self.cur.execute(sql)

            for row in self.cur.fetchall():
                app_review = str(row[17])
                if i.startswith(constants.BUG_REVIEW_TYPE):
                    bug_reviews.append(app_review)
                if i.startswith(constants.FEATURE_REVIEW_TYPE):
                    feature_reviews.append(app_review)
                if i.startswith(constants.RATING_REVIEW_TYPE):
                    rating_reviews.append(app_review)
                if i.startswith(constants.USER_EXPERIENCE_REVIEW_TYPE):
                    userexperience_reviews.append(app_review)

        return bug_reviews, feature_reviews, rating_reviews, userexperience_reviews

    def get_table_names(self, review_type):
        table_names = []
        if review_type == constants.BUG_REVIEW_TYPE:
            table_names.append(constants.BUG_TRAIN)
            table_names.append(constants.BUG_TEST)
            table_names.append(constants.NOT_BUG_TRAIN)
            table_names.append(constants.NOT_BUG_TEST)
        if review_type == constants.FEATURE_REVIEW_TYPE:
            table_names.append(constants.FEATURE_TRAIN)
            table_names.append(constants.FEATURE_TEST)
            table_names.append(constants.NOT_FEATURE_TRAIN)
            table_names.append(constants.NOT_FEATURE_TEST)
        if review_type == constants.RATING_REVIEW_TYPE:
            table_names.append(constants.RATING_TRAIN)
            table_names.append(constants.RATING_TEST)
            table_names.append(constants.NOT_FEATURE_TRAIN)
            table_names.append(constants.NOT_FEATURE_TEST)
        if review_type == constants.USER_EXPERIENCE_REVIEW_TYPE:
            table_names.append(constants.USER_EXPERIENCE_TRAIN)
            table_names.append(constants.USER_EXPERIENCE_TEST)
            table_names.append(constants.NOT_USER_EXPERIENCE_TRAIN)
            table_names.append(constants.NOT_USER_EXPERIENCE_TEST)
        if review_type == constants.MULTI_CLASS:
            table_names.append(constants.BUG_TRAIN)
            table_names.append(constants.BUG_TEST)
            table_names.append(constants.FEATURE_TRAIN)
            table_names.append(constants.FEATURE_TEST)
            table_names.append(constants.RATING_TRAIN)
            table_names.append(constants.RATING_TEST)
            table_names.append(constants.USER_EXPERIENCE_TRAIN)
            table_names.append(constants.USER_EXPERIENCE_TEST)

        return table_names

    def commit(self):
        self.db.commit()

    def close(self):
        self.db.close()
