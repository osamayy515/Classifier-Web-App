import streamlit as st
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

st.title("Streamlit Example")

st.write("""
# Explore Different Classifier
which one is the best?
""")

dataset_name = st.sidebar.selectbox("Select Dataset", ("Iris", "Breast Cancer", "Wine Dataset", "Digits Dataset"))


classifier_name = st.sidebar.selectbox("Select Classifier", ("KNN", "SVM", "Random Forest"))

def get_dataset(dataset_name):
    if dataset_name == "Iris":
        data = datasets.load_iris()
    elif dataset_name == "Breast Cancer":
        data = datasets.load_breast_cancer()
    elif dataset_name == "Wine Dataset":
        data = datasets.load_wine()
    else:
        data = datasets.load_digits()
    X = data.data
    y = data.target
    return X, y
X, y = get_dataset(dataset_name)
st.write("shape of dataset", X.shape)
st.write("number of classes", len(np.unique(y)))

def add_parameter_ui(classifier_name):
    params = dict()
    if classifier_name == "KNN":
        K = st.sidebar.slider("K", 1, 15)
        params["K"] = K
    elif classifier_name == "SVM":
        C = st.sidebar.slider("C", 0.01, 10.0)
        params["C"] = C
    else:
        max_depth = st.sidebar.slider("max_depth", 2, 15)
        no_of_estimators = st.sidebar.slider("no_of_estimator", 1, 100)
        params["max_depth"] = max_depth
        params["no_of_estimators"] = no_of_estimators
    return params

params = add_parameter_ui(classifier_name)

def get_classifier(classifier_name, params):
    if classifier_name == "KNN":
        classifier = KNeighborsClassifier(n_neighbors = params["K"])
    elif classifier_name == "SVM":
        classifier = SVC(C = params["C"])
    else:
        classifier = RandomForestClassifier(n_estimators = params["no_of_estimators"], 
        max_depth = params["max_depth"], random_state = 1234)
    return classifier

classifier = get_classifier(classifier_name, params)

#Classification 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

acc = accuracy_score(y_test, y_pred)
st.write(f"Classifier = {classifier_name}")
st.write(f"Accuracy = {acc}")

#Plot
pca = PCA(2)
X_projected = pca.fit_transform(X)

x1 = X_projected[:, 0]
x2 = X_projected[:, 1]

fig = plt.figure()
plt.scatter(x1, x2, c=y, alpha = 0.8, cmap = "viridis")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.colorbar()

# plt.show()
st.set_option('deprecation.showPyplotGlobalUse', False)
st.pyplot()