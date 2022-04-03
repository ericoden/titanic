# Titanic

A streamlit app to for experimenting with various prediction models for Kaggle's [Titanic prediction challenge](https://www.kaggle.com/competitions/titanic). The app is deployed via Heroku here: https://titanic-ericoden.herokuapp.com/.

The aim is to develop a model that predicts whether a passenger survived based on the following information:

- Name (including title)
- Ticket Class
- Sex
- Age (in years)
- \# of siblings / spouses aboard the Titanic
- \# of parents / children aboard the Titanic
- Ticket
- Fare,
- Cabin Number
- Port of Embarkation

The app allows you to subset the features used, choose between three different classifiers, and select parameters for each classifier. Given a selection of features, the app presents a 2D visualization of the subsetted data, with points colored by outcome. Given the selected features, classifier and parameters, the app reports the accuracy, precision, recall of the resulting model, as well as a confusion matrix.
