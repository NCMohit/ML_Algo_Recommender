from sklearn import tree
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
import pickle
from preprocess_data import preprocess

def train_decision_tree(file,hascf,result):
    data = preprocess(file)
    data = np.matrix(data)
    print("\nFinal data: ")
    print(data)
    if(hascf):
        X = data[:,:-1]
        Y = data[:,-1]
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2,random_state=42)
        clf = tree.DecisionTreeClassifier()
        clf = clf.fit(X_train, Y_train)
        Predicted_Y = clf.predict(X_test)
        acc = metrics.accuracy_score(Y_test, Predicted_Y)
        res = "Test size: 33%, DT Accuracy: "+str(round(acc,2))
        result.set(res)
        tree.plot_tree(clf)
        print("Saving decision tree model in saved_models/")
        with open('saved_models/decision_tree.pkl', 'wb') as file:
            pickle.dump(clf, file)
        plt.show()
    else:
        print("No class feature, skipping decision tree")

def train_linear_regression(file,hascf,result):
    data = preprocess(file)
    data = np.matrix(data)
    if(hascf):
        X = data[:,:-1]
        Y = data[:,-1]
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2,random_state=42)
        reg = LinearRegression().fit(X_train, Y_train)
        Y_pred = reg.predict(X_test)
        result.set(result.get()+"\n"+"Linear regression score: "+str(round(reg.score(X_test,Y_test),2)))

        ytest = [i[0] for i in Y_test]
        ypred = [i[0] for i in Y_pred]
        plt.scatter(range(len(X_test)), ytest, color="black")
        plt.plot(range(len(X_test)), ypred, color="blue", linewidth=3)
        plt.xticks(())
        plt.yticks(())
        plt.show()

        print("Saving linear regression model in saved_models/")
        with open('saved_models/linear_regression.pkl', 'wb') as file:
            pickle.dump(reg, file)
    else:
        print("No class feature, Linear regression")