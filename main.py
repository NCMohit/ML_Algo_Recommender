import tkinter
from tkinter import *
from tkinter.filedialog import askopenfilename
from train_data import train_decision_tree, train_linear_regression
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
def hasClassFeature():
    global hascf 
    hascf = cf.get()
def train():
    global result
    train_decision_tree(filename,hascf,result)
    train_linear_regression(filename,hascf,result)

cf = IntVar()
result = StringVar()

reslabel = Label( top, textvariable=result, relief=RAISED )
ld = tkinter.Button(top, text ="Load Data", command = open_dataset)
hcf = tkinter.Checkbutton(top, text="Has Class feature", variable=cf, onvalue=1, offvalue=0, command=hasClassFeature)
td = tkinter.Button(top, text ="Train Data", command = train)

ld.pack()
hcf.pack()
td.pack()
reslabel.pack()

ld.place(x=25,y=50)
hcf.place(x=25,y=100)
td.place(x=25,y=150)
reslabel.place(x=25,y=200)

top.title("ML Algo recommender")
top.geometry("400x400")
top.mainloop()