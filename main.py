import streamlit as st
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA

# import tensorflow as tf
# from tensorflow import keras
# from keras.models import Model, Sequential
# from keras.layers import Dense, Dropout

# from keras.wrappers.scikit_learn import KerasClassifier
# from scikeras.wrappers import KerasClassifier


from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix

st.title("Titanic Dataset Modeling")

st.sidebar.write(
    """
## Features
"""
)

all_feature_names = [
    "Pclass",
    "Name",
    "Sex",
    "Age",
    "SibSp",
    "Parch",
    "Ticket",
    "Fare",
    "Cabin",
    "Embarked",
]
all_cat_features = ["Name", "Sex", "Ticket", "Cabin", "Embarked"]
all_num_features = ["Pclass", "Age", "SibSp", "Parch", "Fare"]


features = {
    name: st.sidebar.checkbox(name, value=True) for name in all_feature_names
}

selected_features = [name for name in all_feature_names if features[name]]
cat_features = [f for f in all_cat_features if f in selected_features]
num_features = [f for f in all_num_features if f in selected_features]

st.sidebar.write(
    """
## Classifier
"""
)


clf_name = st.sidebar.selectbox(
    "Select Classifier", ("Random Forest", "KNN", "SVM")
)


selected_params = {}


def add_param_ui(clf_name):
    params = {}
    if clf_name == "Random Forest":
        params["max_depth"] = {"min_value": 2, "max_value": 5, "step": 1}
        params["n_estimators"] = {
            "min_value": 10,
            "max_value": 100,
            "step": 10,
        }
    elif clf_name == "SVM":
        params["C"] = {"min_value": 0.1, "max_value": 10.0, "step": 0.1}
    elif clf_name == "KNN":
        params["n_neighbors"] = {"min_value": 1, "max_value": 15, "step": 1}
    elif clf_name == "Neural Network":
        params["dropout"] = {"min_value": 0.0, "max_value": 0.5, "step": 0.05}
        params["n_layers"] = {"min_value": 1, "max_value": 5, "step": 1}
        params["n_nodes_per_layer"] = {
            "min_value": 16,
            "max_value": 128,
            "step": 8,
        }
    for key, value in params.items():
        selected_params[key] = st.sidebar.slider(key, **value)


add_param_ui(clf_name)


train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
X = train.drop("Survived", axis=1)
X = X[selected_features]
y = train["Survived"]
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42
)


class CategoricalFeatureAdder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.common_titles = []
        self.common_ticket_alphas = []
        pass

    def fit(self, X, y=None):
        if "Name" in cat_features:
            titles = [name.split(" ")[1] for name in X["Name"]]
            counts = {title: titles.count(title) for title in set(titles)}
            counts = {
                k: v
                for k, v in sorted(
                    counts.items(), key=lambda item: item[1], reverse=True
                )
            }
            self.common_titles = [
                title for i, title in enumerate(counts.keys()) if i < 5
            ]

        if "Ticket" in cat_features:
            ticket_alphas = [
                "".join(i for i in ticket if i.isalpha()).upper()
                for ticket in X["Ticket"]
            ]
            ticket_alphas = [
                ticket_alpha if ticket_alpha != "" else "None"
                for ticket_alpha in ticket_alphas
            ]
            counts = {
                ticket_alpha: ticket_alphas.count(ticket_alpha)
                for ticket_alpha in set(ticket_alphas)
            }
            counts = {
                k: v
                for k, v in sorted(
                    counts.items(), key=lambda item: item[1], reverse=True
                )
            }
            self.common_ticket_alphas = [
                ticket_alpha
                for i, ticket_alpha in enumerate(counts.keys())
                if i < 5
            ]
        return self

    def transform(self, X, y=None):
        X_new = X.copy(deep=True)
        if "Name" in cat_features:
            titles = [name.split(" ")[1] for name in X["Name"]]
            X_new["title"] = [
                title if title in self.common_titles else np.nan
                for title in titles
            ]
            X_new.drop("Name", axis=1, inplace=True)
        if "Ticket" in cat_features:
            ticket_alphas = [
                "".join(i for i in ticket if i.isalpha()).upper()
                for ticket in X["Ticket"]
            ]
            ticket_alphas = [
                ticket_alpha if ticket_alpha != "" else "None"
                for ticket_alpha in ticket_alphas
            ]
            X_new["ticket_alpha"] = [
                ticket_alpha
                if ticket_alpha in self.common_ticket_alphas
                else "NA"
                for ticket_alpha in ticket_alphas
            ]
            X_new.drop("Ticket", axis=1, inplace=True)
        if "Cabin" in cat_features:
            cabins = [str(i)[0].upper() for i in X["Cabin"]]
            X_new["Cabin"] = [
                cabin if cabin in ["A", "B", "C", "D", "E"] else "Unknown"
                for cabin in cabins
            ]
        return X_new


cat_pipeline = Pipeline(
    [
        ("feature_engineering", CategoricalFeatureAdder()),
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("one-hot", OneHotEncoder()),
    ]
)

num_pipeline = Pipeline(
    [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
)

ct = ColumnTransformer(
    [
        (
            "numerical_pipeline",
            num_pipeline,
            num_features,
        ),
        (
            "categorical_pipeline",
            cat_pipeline,
            cat_features,
        ),
    ]
)

n_cols = np.size(ct.fit_transform(X_train), axis=1)

# create classifier
if clf_name == "Random Forest":
    clf = RandomForestClassifier(random_state=42)
elif clf_name == "SVM":
    clf = SVC()
# elif clf_name == "Neural Network":

#     def create_model(
#         n_layers,
#         n_nodes_per_layer,
#         dropout,
#         kernel_initializer,
#         epochs=4,
#         optimizer="adagrad",
#     ):
#         model = Sequential()
#         model.add(
#             Dense(n_nodes_per_layer, activation="relu", input_shape=(n_cols,))
#         )
#         for _ in range(n_layers):
#             model.add(Dropout(dropout))
#             model.add(Dense(n_nodes_per_layer, activation="relu"))
#         model.add(Dropout(dropout))
#         model.add(Dense(1, activation="sigmoid"))
#         model.compile(
#             loss="binary_crossentropy",
#             optimizer=optimizer,
#             metrics=["accuracy"],
#         )
#         return model

#     early_stopping = tf.keras.callbacks.EarlyStopping(
#         monitor="loss", patience=3
#     )
#     clf = KerasClassifier(
#         create_model,
#         n_layers=1,
#         n_nodes_per_layer=32,
#         epochs=4,
#         dropout=0.2,
#         kernel_initializer="glorot_uniform",
#         verbose=0,
#         callbacks=[early_stopping],
#         random_state=42,
#     )
else:
    clf = KNeighborsClassifier()
clf.set_params(**selected_params)

full_pipeline = Pipeline([("preprocessing", ct), ("clf", clf)])


def pca_plot():
    X_train_prepped = ct.fit_transform(X_train)
    pca = PCA(2, random_state=42)
    X_train_2D = pca.fit_transform(X_train_prepped)
    x1 = X_train_2D[:, 0]
    x2 = X_train_2D[:, 1]
    fig = plt.figure()
    for value in [0, 1]:
        if value == 0:
            color = "red"
            label = "Died"
        else:
            color = "blue"
            label = "Survived"
        indices = [i for i, v in enumerate(y_train) if v == value]
        plt.scatter(x1[indices], x2[indices], c=color, label=label)

    plt.legend(loc="upper right")
    plt.title("2D Feature Projection")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    st.pyplot(fig)


if n_cols > 0:
    full_pipeline.fit(X_train, y_train)
    preds = full_pipeline.predict(X_valid)

    conf_matrix = pd.DataFrame(
        confusion_matrix(y_valid, preds),
        columns=["Died (Predicted)", "Survived (Predicted)"],
        index=["Died (True)", "Survived (True)"],
    )
    if np.size(X_train, axis=1) >= 2:
        pca_plot()
    st.write(
        """
    ## Model Performance
    """
    )
    st.write(f"Accuracy: {100 * accuracy_score(y_valid, preds):.2f}%")
    st.write(f"Precision: {100 * precision_score(y_valid, preds):.2f}%")
    st.write(f"Recall: {100 * recall_score(y_valid, preds):.2f}%")

    st.table(conf_matrix)
