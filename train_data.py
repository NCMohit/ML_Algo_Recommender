from sklearn import tree
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn import metrics
import matplotlib.pyplot as plt
import pickle

def train_decision_tree(data,hascf,result):
    if(hascf):
        X = data[:,:-1]
        Y = data[:,-1]
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2,random_state=42)
        high_depth = 0
        high_acc = 0
        criteria = "entropy"
        entropy_accs = []
        gini_accs = []
        for depth in range(2,11):
            clf = tree.DecisionTreeClassifier(max_depth=depth,criterion="entropy")
            clf = clf.fit(X_train, Y_train)
            Predicted_Y = clf.predict(X_test)
            acc = metrics.accuracy_score(Y_test, Predicted_Y)
            entropy_accs.append(acc)
            if(high_acc < acc):
                high_acc = acc
                high_depth = depth
        for depth in range(2,11):
            clf = tree.DecisionTreeClassifier(max_depth=depth,criterion="gini")
            clf = clf.fit(X_train, Y_train)
            Predicted_Y = clf.predict(X_test)
            acc = metrics.accuracy_score(Y_test, Predicted_Y)
            gini_accs.append(acc)
            if(high_acc < acc):
                high_acc = acc
                high_depth = depth
                criteria = "gini"
        clf = tree.DecisionTreeClassifier(max_depth=high_depth,criterion=criteria)
        clf = clf.fit(X_train, Y_train) 
        plt.subplot(1,2,1)
        plt.xlabel("Entropy depths")
        plt.ylabel("Accuracies")
        plt.plot(range(2,11),entropy_accs)
        plt.subplot(1,2,2)
        plt.xlabel("Gini depths")
        plt.ylabel("Accuracies")
        plt.plot(range(2,11),gini_accs)
        plt.show()
        res = "Test size: 20%\nBest Decision Tree Accuracy: "+str(round(high_acc,2))+" at max height: "+str(high_depth)+" and criteria: "+criteria
        result.set(result.get()+"\n"+res)
        tree.plot_tree(clf)
        print("Saving decision tree model in saved_models/")
        with open('saved_models/decision_tree.pkl', 'wb') as file:
            pickle.dump(clf, file)
        plt.show()
    else:
        res = "No class feature, skipping decision tree"
        result.set(result.get()+"\n"+res)

def train_linear_regression(data,hascf,result):
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
        res = "No class feature set, skipping Linear regression"
        result.set(result.get()+"\n"+res)

def train_gaussian_naive_bayes(data,hascf,result):
    if(hascf):
        feature_set = []
        max_accuracy_feature = 0
        max_accuracy = 0
        for i in range(10):
            old_acc = max_accuracy
            for feature in range(1,10):
                if(feature not in feature_set):
                    X = data[:,feature_set+[feature]]
                    Y = data[:, -1]
                    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2,random_state=42)
                    clf = GaussianNB()
                    clf = clf.fit(X_train, Y_train)
                    Predicted_Y = clf.predict(X_test)
                    acc = metrics.accuracy_score(Y_test, Predicted_Y)

                    if(acc > max_accuracy):
                        max_accuracy_feature = feature_set + [feature]
                        max_accuracy = acc

            print("\nMax Accuracy: ",max_accuracy," with feature set ",max_accuracy_feature)
            feature_set = max_accuracy_feature
            print("Feature set: ",feature_set)
            if(max_accuracy - old_acc <= 0):
                break
        feature_set_string = "["
        for feat in feature_set:
            feature_set_string += str(feat)
            feature_set_string += ", "
        feature_set_string += "]"
        res = "Gaussian Naive Bayes, best feature set: "+feature_set_string+" with accuracy: "+str(round(max_accuracy,2))
        result.set(result.get()+"\n"+res)

        print("Saving linear regression model in saved_models/")
        with open('saved_models/gaussian_naive_bayes.pkl', 'wb') as file:
            pickle.dump(clf, file)
    else:
        print("No class feature, skipping Gaussian Naive Bayes")

def train_knn(data,hascf,result):
    X = data[:,:]
    if(hascf):
        print("Has class feature, not considerg class feature column")
        X = data[:,:-1]
    max_k = 0
    max_score = 0
    silhouettes = []
    for k in range(2,11):
        kmeans = KMeans(n_clusters=k)
        kmeans.fit_predict(X)
        score = silhouette_score(X, kmeans.labels_, metric='euclidean')
        silhouettes.append(score)
        if(max_score < score):
            max_score = score
            max_k = k
    kmeans = KMeans(n_clusters=max_k)
    kmeans.fit_predict(X)
    plt.xlabel("K Values")
    plt.ylabel("Silhouette Scores")
    plt.plot(range(2,11),silhouettes)
    plt.show()
    res = "Best K Means Silhouette score: "+str(round(max_score,3))+" at k value: "+str(max_k)
    result.set(result.get()+"\n"+res)
    print("Saving decision tree model in saved_models/")
    with open('saved_models/kmeans.pkl', 'wb') as file:
        pickle.dump(kmeans, file)