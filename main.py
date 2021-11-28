import tkinter
from tkinter import *
from tkinter.filedialog import askopenfilename
from train_data import train_decision_tree, train_linear_regression, train_gaussian_naive_bayes
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
    filename = askopenfilename()
    result.set("Loaded dataset !")

def traindt():
    global filename
    global result
    global cf
    try:
        data = preprocess(filename)
        data = np.matrix(data)
        print("\nFinal data: ")
        print(data)
        train_decision_tree(data,cf.get(),result)
    except Exception as e:
        print(e)
        result.set("Error: "+str(e))
def trainlr():
    global filename
    global result
    global cf
    try:
        data = preprocess(filename)
        data = np.matrix(data)
        train_linear_regression(data,cf.get(),result)
    except Exception as e:
        print(e)
        result.set("Error: "+str(e))
def traingnb():
    global filename
    global result
    global cf
    try:
        data = preprocess(filename)
        data = np.matrix(data)
        train_gaussian_naive_bayes(data,cf.get(),result)
    except Exception as e:
        print(e)
        result.set("Error: "+str(e))

cf = IntVar()
result = StringVar()

reslabel = Label( top, textvariable=result, relief=RAISED )
ld = tkinter.Button(top, text ="Load Data", command = open_dataset)
hcf = tkinter.Checkbutton(top, text="Has Class feature", variable=cf, onvalue=1, offvalue=0)
tdt = tkinter.Button(top, text ="Decision Tree", command = traindt)
tlg = tkinter.Button(top, text ="Linear Regression", command = trainlr)
tgnb = tkinter.Button(top, text ="Gaussian Naive Bayes", command = traingnb)

ld.pack()
hcf.pack()
tdt.pack()
tlg.pack()
tgnb.pack()
reslabel.pack()

ld.place(x=25,y=25)
hcf.place(x=25,y=70)
tdt.place(x=25,y=100)
tlg.place(x=150,y=100)
tgnb.place(x=300,y=100)
reslabel.place(x=25,y=150)

top.title("ML Algo recommender")
top.geometry("500x400")
top.mainloop()