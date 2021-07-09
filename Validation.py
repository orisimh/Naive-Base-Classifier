
from tkinter import filedialog
from tkinter import *
from tkinter import messagebox
import Classifier
import os.path
# import GUI



from os import path
import pandas as pd
import numpy as np
import operator


class Validation:

 def __init__(self):

     # self.root =root
     pass



 def validate_path( self , pth , B , C ,D ):



     try:
        B1 = os.stat(B).st_size
        C1 = os.stat(C).st_size
        D1 = os.stat(D).st_size

     except :

        if( not path.exists(B)):
         messagebox.showerror(title='Naïve Bayes Classifier', message='File Structure Is Missing')
         return False
        if (not path.exists(C)):
         messagebox.showerror(title='Naïve Bayes Classifier', message='File train Is Missing')
         return False
        if (not path.exists(D)):
         messagebox.showerror(title='Naïve Bayes Classifier', message='File test Is Missing')
         return False

     try:
         Structure = pd.read_csv(B, header=None, sep=' ', names=['A', 'Name', 'Type'])
         train = pd.read_csv(C, header=0)
         number_of_rows_train = len(train.index)
         train_cols = train.columns
     except:
         messagebox.showerror(title='Naïve Bayes Classifier', message='File train is Empty')
         return False

     try:
      test = pd.read_csv(D, header=0)
      number_of_rows_test = len(test.index)
      test_cols = test.columns

     except:
      messagebox.showerror(title='Naïve Bayes Classifier', message='File test is Empty')
      return False

    # if(B1 >0  and C1 > 0 and D1 >0 ):

     i = 0
     for row in Structure.iterrows():

      if (row[1][1] not in  train_cols):
             messagebox.showerror(title='Naïve Bayes Classifier', message='column '+ row[1][1] + ' is missing in the train' )
             return False
      if (row[1][1] not in test_cols):
             messagebox.showerror(title='Naïve Bayes Classifier', message='column '+ row[1][1] + ' is missing in the test')
             return False

      # i = i + 1



      if(number_of_rows_train<=1):
                 messagebox.showerror(title='Naïve Bayes Classifier',message='rows are missing , only columns exist in the train set')
                 return False

      if (number_of_rows_test <= 1):
                 messagebox.showerror(title='Naïve Bayes Classifier', message='rows are missing , only columns exist in the test set')
                 return False


     return True





        # else:
        #       messagebox.showerror(title='Naïve Bayes Classifier', message='File Is Empty')
        #       return False




         # return False



 def validate_bin(self, new_text):


        if (not new_text) :  # the field is being cleared
            # self.gui.\
            # setBuildDisabled()
            return 1

        if(new_text[0] == '0'):
            # self.gui.setBuildDisabled()
            return 2
        else:
            return 3


