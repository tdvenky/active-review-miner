
# Active Review Miner: Active Learning Framework for Python

Active learning is a machine learning paradigm that can be used to reduce the human effort involved in supervied learning.

This project contains Python implementation for the following **uncertanity sampling** strategies in active learning:

* Least Confident Prediction
* Smallest Margin
* Highest Entropy

### Prerequisites

(IN PROGRESS)

This project requires [scikit-learn](http://scikit-learn.org/stable/index.html) and [SciPy](https://www.scipy.org/), [MySQLdb](https://github.com/PyMySQL/mysqlclient-python) to run.

* Scikit-learn

    The instructions to install scikit-learn can be found at [Advanced installation instructions](http://scikit-learn.org/stable/developers/advanced_installation.html). Scikit-learn requires python, NumPy and SciPy to be installed first. 

    The following example is particular to windows, but instructions for MacOS and Linux can be found from the link above.

    ```
    pip install -U scikit-learn
    ```
* MySQLdb

    ```
    pip install mysqlclient
    ```



## Getting Started

(IN PROGRESS)

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

## Motivation

(IN PROGRESS)

We motivate the need for active learning via an example scenario, demonstrating that choosing the right training set can potentially enhance the review classification accuracy. Consider an app from google play store whose developers want to analyze it's reviews to identify the features requested by the app's users. They would like to employ an automated classifier to identify reviews corrosponding to feature requests.

### Installing

(IN PROGRESS)

A step by step series of examples that tell you how to get a development env running

Say what the step will be

```
Give the example
```

And repeat

```
until finished
```

End with an example of getting some data out of the system or using it for a little demo


## Built With

* [scikit-learn](http://scikit-learn.org/stable/index.html) - Machine Learning in Python



## Authors

* **Venkatesh Thimma Dhinakaran**
* **[Pradeep Kumar Murukannaiah](http://www.se.rit.edu/~pkm/)**

