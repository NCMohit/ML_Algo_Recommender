import tkinter
import tkinter.font as tkFont
from tkinter import *
from tkinter.filedialog import askopenfilename
from train_data import train_decision_tree, train_linear_regression, train_gaussian_naive_bayes, train_knn, train_svm, train_dbscan
import numpy as np
from preprocess_data import preprocess
import os

directory = os.getcwd()+"/saved_models/"
try:
    os.stat(directory)
except:
    os.mkdir(directory)

top = tkinter.Tk()

def open_dataset():
    global filename 
    global data
    try:
        filename = askopenfilename()
        result.set("Loaded dataset !")
        data = preprocess(filename)
        data = np.matrix(data)
        print("\nFinal data: ")
        print(data)
    except Exception as e:
        print(e)
        result.set("Error: "+str(e))

def traindt():
    global filename
    global result
    global cf
    global data
    try:
        train_decision_tree(data,cf.get(),result)
    except Exception as e:
        print(e)
        result.set("Error: "+str(e))
def trainlr():
    global filename
    global result
    global data
    global cf
    try:
        train_linear_regression(data,cf.get(),result)
    except Exception as e:
        print(e)
        result.set("Error: "+str(e))
def traingnb():
    global filename
    global result
    global cf
    global data
    try:
        train_gaussian_naive_bayes(data,cf.get(),result)
    except Exception as e:
        print(e)
        result.set("Error: "+str(e))

def trainknn():
    global filename
    global result
    global cf
    global data
    try:
        train_knn(data,cf.get(),result)
    except Exception as e:
        print(e)
        result.set("Error: "+str(e)) 

def trainsvm():
    global filename
    global result
    global cf
    global data
    try:
        train_svm(data,cf.get(),result)
    except Exception as e:
        print(e)
        result.set("Error: "+str(e)) 

def traindbscan():
    global filename
    global result
    global cf
    global data
    try:
        train_dbscan(data,cf.get(),result)
    except Exception as e:
        print(e)
        result.set("Error: "+str(e))      

cf = IntVar()
result = StringVar()

reslabel = Label( top, textvariable=result, relief=RAISED )
ld = tkinter.Button(top, text ="Load Data", command = open_dataset)
hcf = tkinter.Checkbutton(top, text="Has Feature column", variable=cf, onvalue=1, offvalue=0)
tdt = tkinter.Button(top, text ="Decision Tree", command = traindt)
tlg = tkinter.Button(top, text ="Linear Regression", command = trainlr)
tgnb = tkinter.Button(top, text ="Gaussian Naive Bayes", command = traingnb)
tknn = tkinter.Button(top, text ="K Nearest Neighbours", command = trainknn)
tsvm = tkinter.Button(top, text ="SVM", command = trainsvm)
tdbscan = tkinter.Button(top, text ="DBSCAN", command = traindbscan)

reslabel.config(font=('Helvatical bold',15))
ld.config(font=('Helvatical bold',15))
hcf.config(font=('Helvatical bold',15))
tdt.config(font=('Helvatical bold',15))
tlg.config(font=('Helvatical bold',15))
tgnb.config(font=('Helvatical bold',15))
tknn.config(font=('Helvatical bold',15))
tsvm.config(font=('Helvatical bold',15))
tdbscan.config(font=('Helvatical bold',14))


ld.pack()
hcf.pack()
tdt.pack()
tlg.pack()
tgnb.pack()
tknn.pack()
tsvm.pack()
tdbscan.pack()
reslabel.pack()

ld.place(x=25,y=25)
hcf.place(x=25,y=70)
tdt.place(x=25,y=100)
tlg.place(x=200,y=100)
tgnb.place(x=420,y=100)
tknn.place(x=25,y=140)
tsvm.place(x=285,y=140)
tdbscan.place(x=370,y=140)
reslabel.place(x=25,y=200)

top.title("ML Algo recommender")
top.geometry("800x600")
top.mainloop()