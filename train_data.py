from sklearn import tree, svm, metrics
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import pickle
import numpy as np

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

        xtest = np.squeeze(np.asarray(X_test[:,4]))
        ypred = np.squeeze(np.asarray(Y_pred[:,0]))
        ytest = np.squeeze(np.asarray(Y_test[:,0]))

        plt.scatter(xtest, ytest, color="black")
        plt.scatter(xtest, ypred, color="blue", linewidth=3)
        plt.xticks(())
        plt.yticks(())
        plt.xlabel("First feature values")
        plt.ylabel("Class feature values")
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
        features_strs = []
        accs = []
        best_feature = ""
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
                    features_str = ""
                    for f in feature_set + [feature]:
                        features_str += str(f)
                    features_strs.append(features_str)
                    accs.append(acc)
                    if(acc > max_accuracy):
                        max_accuracy_feature = feature_set + [feature]
                        max_accuracy = acc

            print("\nMax Accuracy: ",max_accuracy," with feature set ",max_accuracy_feature)
            feature_set = max_accuracy_feature
            print("Feature set: ",feature_set)
            if(max_accuracy - old_acc <= 0):
                break
        plt.xlabel("Feature columns")
        plt.ylabel("Accuracies")
        plt.xticks(rotation=90)
        plt.plot(features_strs,accs)
        plt.show()
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
    print("Saving knn in saved_models/")
    with open('saved_models/kmeans.pkl', 'wb') as file:
        pickle.dump(kmeans, file)

def plot_svm_kernel(C,kernel,subplot,X_train,X_test,Y_train,Y_test):
    global high_C
    global high_acc
    global high_gamma
    global high_kernel
    gamma = 0.1
    gamma_vals = []
    accs = []
    while(gamma <= 1):
        model = svm.SVC(kernel=kernel, gamma=gamma, C=C)
        clf = model.fit(X_train,np.squeeze(np.asarray(Y_train)))
        
        Predicted_Y = clf.predict(X_test)
        accuracy = metrics.accuracy_score(Y_test, Predicted_Y)
        accs.append(accuracy)
        if(high_acc < accuracy):
            high_acc = accuracy
            high_gamma = gamma
            high_C = C
            high_kernel = kernel
        gamma_vals.append(gamma)
        gamma += 0.1
    plt.subplot(2,2,subplot)
    plt.xlabel("C="+str(C)+" gamma values "+kernel+" kernel SVM")
    plt.ylabel("Accuracies")
    plt.plot(gamma_vals,accs)

def train_svm(data,hascf,result):
    global high_C
    global high_acc
    global high_gamma
    global high_kernel
    if(hascf):
        X = data[:,:-1]
        Y = data[:,-1]
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2,random_state=42)
        high_gamma = 0
        high_C = 0
        high_acc = 0

        plot_svm_kernel(0.1,'rbf',1,X_train, X_test, Y_train, Y_test)
        plot_svm_kernel(1,'rbf',2,X_train, X_test, Y_train, Y_test)
        plot_svm_kernel(10,'rbf',3,X_train, X_test, Y_train, Y_test)
        plot_svm_kernel(100,'rbf',4,X_train, X_test, Y_train, Y_test)

        # plot_svm_kernel(0.1,'linear',5,X_train, X_test, Y_train, Y_test)
        # plot_svm_kernel(1,'linear',6,X_train, X_test, Y_train, Y_test)
        # plot_svm_kernel(10,'linear',7,X_train, X_test, Y_train, Y_test)
        # plot_svm_kernel(100,'linear',8,X_train, X_test, Y_train, Y_test)
        plt.show()

        model = svm.SVC(kernel=high_kernel, gamma=high_gamma, C=high_C)
        clf = model.fit(X_train,Y_train)

        res = "Best SVM Accuracy: "+str(round(high_acc,3))+" at C value: "+str(high_C)+" at gamma value: "+str(high_gamma)
        result.set(result.get()+"\n"+res)
        print("Saving SVM Model in saved_models/")
        with open('saved_models/svm.pkl', 'wb') as file:
            pickle.dump(clf, file)
    else:
        res = "No class feature set, Support Vector Machine"
        result.set(result.get()+"\n"+res)

def plot_dbscan(min_samples,X,subplot):
    global max_min_samples
    global max_eps
    global max_dbscan_score
    silhouettes = []
    for eps in range(10,21):
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        dbscan.fit_predict(X)
        # print(dbscan.labels_)
        score = silhouette_score(X, dbscan.labels_, metric='euclidean')
        silhouettes.append(score)
        if(max_dbscan_score < score):
            max_dbscan_score = score
            max_eps = eps
            max_min_samples = min_samples
    plt.subplot(2,2,subplot)
    plt.xlabel("Min samples: "+str(min_samples)+", Epsilon Values")
    plt.ylabel("Silhouette Scores")
    plt.plot(range(10,21),silhouettes)

def train_dbscan(data,hascf,result):
    global max_min_samples
    global max_eps
    global max_dbscan_score
    X = data[:,:]
    if(hascf):
        print("Has class feature, not considerg class feature column")
        X = data[:,:-1]
    max_eps = 0
    max_min_samples = 0
    max_dbscan_score = 0

    plot_dbscan(9,X,1)
    plot_dbscan(10,X,2)
    plot_dbscan(11,X,3)
    plot_dbscan(12,X,4)
    plt.show()
    res = "Best DBSCAN Silhouette score: "+str(round(max_dbscan_score,3))+" at eps value: "+str(max_eps)+" and min_samples: "+str(max_min_samples)
    result.set(result.get()+"\n"+res)
    dbscan = DBSCAN(eps=max_eps, min_samples=max_min_samples)
    dbscan.fit_predict(X)
    print("Saving knn in saved_models/")
    with open('saved_models/dbscan.pkl', 'wb') as file:
        pickle.dump(dbscan, file)