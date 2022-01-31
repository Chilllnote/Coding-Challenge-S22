# ACM Research Coding Challenge (Spring 2022)

## [](https://github.com/ACM-Research/-DRAFT-Coding-Challenge-S22#no-collaboration-policy)No Collaboration Policy

**You may not collaborate with anyone on this challenge.**  You  _are_  allowed to use Internet documentation. If you  _do_  use existing code (either from Github, Stack Overflow, or other sources),  **please cite your sources in the README**.

## [](https://github.com/ACM-Research/-DRAFT-Coding-Challenge-S22#submission-procedure)Submission Procedure

Please follow the below instructions on how to submit your answers.

1.  Create a  **public**  fork of this repo and name it  `ACM-Research-Coding-Challenge-S22`. To fork this repo, click the button on the top right and click the "Fork" button.

2.  Clone the fork of the repo to your computer using  `git clone [the URL of your clone]`. You may need to install Git for this (Google it).

3.  Complete the Challenge based on the instructions below.

4.  Submit your solution by filling out this [form](https://acmutd.typeform.com/to/uTpjeA8G).

## Assessment Criteria 

Submissions will be evaluated holistically and based on a combination of effort, validity of approach, analysis, adherence to the prompt, use of outside resources (encouraged), promptness of your submission, and other factors. Your approach and explanation (detailed below) is the most weighted criteria, and partial solutions are accepted. 

## [](https://github.com/ACM-Research/-DRAFT-Coding-Challenge-S22#question-one)Question One

[Binary classification](https://en.wikipedia.org/wiki/Binary_classification) is a type of classification task that labels elements of a set (i.e. dataset) into two different groups. An example of this type of classification would be identifying if people had a specific disease or not based on certain health characteristics. The dataset found in `mushrooms.csv` holds data (22 different characteristics, specifically) about different types of mushrooms, including a mushroom's cap shape, cap surface texture, cap color, bruising, odor, and more. Remember to split the data into test and training sets (you can choose your own percent split). Information about the meaning of the letters under each column can be found within the file `attributelegend.txt`.

**With the file `mushrooms.csv`, use an algorithm of your choice to classify whether a mushroom is poisonous or edible.**

**You may use any programming language you feel most comfortable. We recommend Python because it is the easiest to implement. You're allowed to use any library or API you want to implement this, just document which ones you used in this README file.** Try to complete this as soon as possible.

Regardless if you can or cannot answer the question, provide a short explanation of how you got your solution or how you think it can be solved in your README.md file. However, we highly recommend giving the challenge a try, you just might learn something new!

## Alright let's begin

For binary classification, if the data is cleaned up enough, you don't really neeed to do too much before the predicting. Of course, if you want a higher accuracy you CAN couple the data you have with other data that can match it and train your model on the combination....but there is not current need to do that as you will see below. I did the test using multiple models so that I could check wether or not I was using the most accurate model.

###### Libraries, packages, and tutorials used:

Sklearn
Matplotlib

Frankly, sklearn was used for basically all the models so, to clarify, I used these classes from sklearn (libraries imported are right beside them if necessary):

Metrics
Linear_model
Tree
Svm
Ensemble (RandomForestClassifier, ExtraTreesClassifier, VotingClassifier
Model_selection (cross_validate, train_test_split)
Discriminatn_analysis
Cluster
Neighbors
Naive_bayes
Metrics (mean_squared_error, precision_recall_fscore_support
Preprocessing
feature_selection (SelectFromModel, RFECV)

tutorial 1: https://towardsdatascience.com/data-cleaning-with-python-and-pandas-detecting-missing-values-3e9c6ebcf78b (Used to clean up the missing data within the dataset, the parts where the data is just "?")

tutorial 2: https://www.kaggle.com/klaudiajankowska/binary-classification-multiple-method-comparison (used to create the measures for the multiple binary classification models)

(As before, I did my code within the Spyder IDE via anaconda. It's the coding environment I'm most used to for data analysis in python. I could, of course, do jupyter notebook but I wanted to be in my "natural place" if you will.)

# First test

There were some......Astonishing results 

<img width="631" alt="Screenshot 2022-01-31 020339" src="https://user-images.githubusercontent.com/22717191/151767984-5347968f-7835-4c39-88d5-d8f935cf3ec6.png">

The accuracy was high, ***Too*** high, in fact. As always if the accuracy of a training set is 100% for a majority of models, it should be taken into question. So, I looked into an answer and found one:

“regression techniques involve computing a matrix inversion, which is inaccurate if the determinant is close to 0 (i.e. two or more variables are almost a linear combination of each other)”

It seems some of the variables for some of the mushrooms don't *only* have a high correlation to being either poisonous or edible, but have a high correlation with *each other*, in general. If you want a better explanation imagine you have two variables x and y that both effect a z. If x decreases, z increase. If y increases, z increase. Both variables are linearly correlated with one another so a change in one will have an equal change in the other, either changes, most likely, resulting in z increasing. This is known in statistics as multicollinearity. It’s defined like so: 

"**Multicollinearity** (also collinearity) is a phenomenon in which one predictor variable in a multiple regression model can be linearly predicted from the others with a substantial degree of accuracy." The predictor variable, however, can be *multiple* variables within the set.

Forgive me for believing that I was losing my mind (though I didn't mention it, I *was* wondering how I got such a high accuracy), it just seems like the variables, when it comes to the mushrooms, are very strongly correlated. Strong enough, in fact, that accuracy of prediction skyrockets. You can see how this correlation comes about by just scrolling through the data set, in fact:

Stalk-color-above-ring: almost always w

Stalk-color-below-ring: almost always w

Veil-type: almost always p

Veil-color: almost always w

Ring-number: almost always 0

Ring-type: almost always p.

Gill-attachment: almost always f

I’m certain that the multicollinearity comes from the repetitions of variables that can be seen within the set. If you have a stalk color above the ring that is white and a veil-type that is p, you will, with almost full certainty, have a ring-type of p. This collinearity leads to a higher model prediction. Test aren’t done, although, as the dataset has some data that isn’t defined (“left as ‘?’”), so let's fill that data and do one more test.

# Second test

<img width="628" alt="Screenshot 2022-01-31 020339" src="https://user-images.githubusercontent.com/22717191/151768649-94a2b05e-98c3-4a92-9b60-77c60749cbb8.png">

The accuracy took a slight decrease but nothing much has changed (which wasn’t unexpected. All the missing values got filled in by the most frequent letter. I imagine not a lot of change was gonna happen from such a thing.

Either way, use of the Decision Tree, SVM, Quadratic Discriminant, Random Forest, or k-Nearest Neighbors model are recommended for the set. As always, I tested for multiple models to see which one would be the most accurate. 
