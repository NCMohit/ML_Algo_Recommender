import tkinter
from tkinter import *
from tkinter.filedialog import askopenfilename
from train_data import train_decision_tree
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
    train_decision_tree(filename,hascf)

cf = IntVar()

tkinter.Button(top, text ="Load Data", command = open_dataset).pack()
tkinter.Checkbutton(top, text="Has Class feature", variable=cf, onvalue=1, offvalue=0, command=hasClassFeature).pack()
tkinter.Button(top, text ="Train Data", command = train).pack()

top.geometry("300x300")
top.mainloop()