from tkinter import filedialog
from tkinter import *
from tkinter import messagebox

import Classifier
import Validation
import os.path
from os import path
import pandas as pd
import numpy as np
import operator
from importlib import reload


class GUI:

    def __init__(self, root):

        self.A = 0
        self.B = 0
        self.C = 0
        self.D = 0
        self.bin = 0
        self.folder_path = StringVar()
        self.validate = Validation.Validation()

        root.title("Naïve Bayes Classifier")
        root.configure(background="white")
        root.geometry("750x350+200+200")

        # horizonaly space
        Label(root, text="", bg="white").grid(row=0, column=0)
        Label(root, text="", bg="white").grid(row=1, column=0)
        Label(root, text="", bg="white").grid(row=2, column=0)

        Directory_Path = Label(root, text="Directory_Path:", bg="white", width=15)
        Directory_Path.grid(row=3, column=0, ipady=10, sticky=W)

        self.ThePath = Label(master=root, textvariable=self.folder_path, bd=1, relief="solid", bg="white", width=20)
        self.ThePath.grid(row=3, column=1, ipadx=200, sticky=SW, pady=10, ipady=2)

        Label(root, text="  ", bg="white").grid(row=3, column=2)  # vertically space

        # Browse Button
        self.Browse = Button(text="Browse", command=self.Browse, bd=1, relief="solid", width=10)
        self.Browse.grid(row=3, column=3)

        Label(root, text="Discretization Bins:", bg="white").grid(row=5, column=0)  # horizonaly space

        # Entry of the bin number

        reg = root.register(self.switcherbutton)
        self.entry = Entry(root, validate="key", validatecommand=(reg, '%P'), bd=1, relief="solid", width=30,
                           bg="white")  # , width=15 ,  relief="solid"
        self.entry.configure(state="disabled")
        self.entry.grid(row=5, column=1, sticky=W)

        Label(root, text="", bg="white").grid(row=6, column=0)  # horizonaly space

        # Build Button
        self.Build = Button(text="Build", command=self.Build, bd=1, relief="solid", width=15, )
        self.Build.configure(state="disabled")
        self.Build.grid(row=7, column=1, padx=200)

        Label(root, text="", bg="white").grid(row=8, column=0)  # horizonaly space

        # Classify Button
        Classify = Button(text="Classify", command=self.Classify, bd=1, relief="solid", width=15)
        Classify.grid(row=9, column=1, padx=200)

    # def validate_path( self , pth):
    #
    #    self.A= pth
    #    self.B= pth+'/Structure.txt'
    #    self.C= pth+'/train.csv'
    #    self.D= pth+'/test.csv'
    #
    #    try:
    #        B1 = os.stat(self.B).st_size
    #        C1 = os.stat(self.C).st_size
    #        D1 = os.stat(self.D).st_size
    #
    #
    #        if(B1 >0  and C1 > 0 and D1 >0 ):
    #                return True
    #        else:
    #              messagebox.showinfo(title='Naïve Bayes Classifier', message='File Is Empty')
    #              return False
    #
    #    except :
    #        messagebox.showinfo(title='Naïve Bayes Classifier', message='File Is Missing')
    #        return False
    #
    #
    #
    # def validate_bin(self, new_text):
    #
    #        if (not new_text) :  # the field is being cleared
    #            self.Build.configure(state="disabled")
    #            self.bin = 0
    #            return True
    #        if(new_text[0] == '0'):
    #            self.Build.configure(state="disabled")
    #            self.bin = 0
    #            return False
    #        try:
    #          self.bin = int(new_text)
    #          self.Build.configure(state="normal")
    #
    #          return True
    #
    #
    #        except ValueError:
    #          return False

    def switcherbutton(self, new_text):

        res = self.validate.validate_bin(new_text)

        if (res==1):

            self.Build.configure(state="disabled")
            self.bin = 0
            return True


        if(res==2):

            self.Build.configure(state="disabled")
            self.bin = 0
            return False
        else:

         try:
            self.bin = int(new_text)
            self.Build.configure(state="normal")
            # self.gui.setBuildEnabled(new_text)
            return True
         except ValueError:
             return False



    def Browse(self):

        # global folder_path


        filename = filedialog.askdirectory()
        self.A = filename
        self.B = filename + '/Structure.txt'
        self.C = filename + '/train.csv'
        self.D = filename + '/test.csv'

        isvalidate = self.validate.validate_path(filename,  self.B , self.C , self.D  )

        if (isvalidate):
            self.folder_path.set(filename)
            self.entry.configure(state="normal")

        else:
            self.folder_path.set("")
            self.entry.configure(state="disabled")

    def Build(self):

        # array = []
        # with open(self.B, 'r') as f:  # better way to open file
        #  for line in f:  # for each line
        #   for i in line :
        #   #out = [line[i:i + 1] for i in range(0, len(line) - 1)]
        #    array.extend(line)
        #   array=[]
        # print(self.B)

        Structure = pd.read_csv(self.B, header=None, sep=' ', names=['A', 'Name', 'Type'])
        Structure = Structure.iloc[:, 1:3]

        train = pd.read_csv(self.C, header=0)
        test = pd.read_csv(self.D, header=0)

        self.data = pd.concat([train, test])



        # print(self.data)

        self.clsfr = Classifier.Classifier(Structure, self.bin)
        self.clsfr.preprocessing(self.data , train.shape , test.shape)

        # self.clsfr.fit_the_model()
        messagebox.showinfo(title='Naïve Bayes Classifier', message='Building classifier using train-set is done!')

        # train.to_csv('outcms.csv', index = False)

    def Classify(self):

        # test = pd.read_csv(self.D, header=0)
        self.clsfr.predict(self.A)  #test, self.A)

        messagebox.showinfo(title='Naïve Bayes Classifier', message='The Classification Is Done')
        root.destroy()


root = Tk()

my_gui = GUI(root)
root.mainloop()
