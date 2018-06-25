import argparse

from review_reader import ReviewReader


class ActiveReviewClassifier:
    def __init__(self, review_type):
        username, password, host, database_name = self.get_db_credentials()
        database = ReviewReader(username, password, host, database_name)

        reviews_pos_cls, reviews_neg_cls = database.get_app_reviews(review_type)
        for i in reviews_pos_cls:
            print(i.review)
        for i in reviews_neg_cls:
            print(i.review)
        database.commit()
        database.close()

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
    active_review_classifier = ActiveReviewClassifier(review_type)


if __name__ == '__main__':
    main()
