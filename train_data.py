from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
import pickle
from preprocess_data import preprocess

def train_decision_tree(file,hascf):
    data = preprocess(file)
    data = np.matrix(data)
    print("\nFinal data: ")
    print(data)
    if(hascf):
        X = data[:,:-1]
        Y = data[:,-1]
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33,random_state=42)
        clf = tree.DecisionTreeClassifier()
        clf = clf.fit(X_train, Y_train)
        Predicted_Y = clf.predict(X_test)
        acc = metrics.accuracy_score(Y_test, Predicted_Y)
        print("Test size: 33%, Accuracy: ",acc)
        tree.plot_tree(clf)
        print("Saving decision tree model in saved_models/")
        with open('saved_models/decision_tree.pkl', 'wb') as file:
            pickle.dump(clf, file)
        plt.show()
    else:
        print("No class feature, skipping decision tree")
