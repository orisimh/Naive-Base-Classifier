from tkinter import filedialog
from tkinter import *
from tkinter import messagebox

import os.path
from os import path
import pandas as pd
import numpy as np
import operator
import statistics as stats

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold  # For K-fold cross validation
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics

class Classifier:

 def __init__(self , Strucutre , bin  ):

  self.data = None
  self.Structure = Strucutre
  self.bin = bin
  self.fit= None
  self.train = None
  self.test = None

 def preprocessing(self , data, train_shape , test_shape):

    self.data = data

    for row in self.Structure.iterrows():

        if(row[1][0]=='class'): # if it the last row in the structure - class , break
          break

        if(row[1][1]=='NUMERIC' ): # if the type of the column is numeric

            #self.train.iloc[self.train['class'] == x[self.train.shape[1] ] , row[1][0]].mean()

            # self.train.loc[self.train['class'] == x[self.train.shape[1]], row[1][0]].mean()
            # a [loc[x.name, row[1][0]]]

            # # fill na of the column by the mean per label in the class
            # self.data[row[1][0]]= self.data[row[1][0]].fillna(self.data.apply(lambda x: self.data.loc[self.data['class'] == x[self.data.shape[1]-1], row[1][0]].mean() , axis=1))

            # check if bin number smaller than unique values of the column
            num = len(self.data[row[1][0]].unique())
            if(num > self.bin):

             # fill na of the column by the mean per label in the class
             self.data[row[1][0]] = self.data[row[1][0]].fillna(self.data.apply(lambda x: self.data.loc[self.data['class'] == x[self.data.shape[1] - 1], row[1][0]].mean(), axis=1))

             # execute discritization to numeric coulumn
             self.data[row[1][0]] =pd.cut(self.data[row[1][0]], bins=self.bin ) #, #labels = [ "bin_"+str(i) for i in range(self.bin)]  )

            else:
                self.data[row[1][0]].fillna(stats.mode(self.data[row[1][0]]), inplace=True)

        else: # if the type of the column is nominali/categorali
            self.data[row[1][0]].fillna(stats.mode(self.data[row[1][0]]), inplace=True)



    # self.train = data.iloc[:train_shape[0], :train_shape[1]]
    # self.test = data.iloc[:test_shape[0], :test_shape[1]]


    # print(self.train.head())

    le= LabelEncoder()
    self.data.iloc[:,:data.shape[1]-1] = self.data.iloc[:,:data.shape[1]-1].apply(lambda x: le.fit_transform(x) , axis =0)

    self.train = data.iloc[:train_shape[0], :train_shape[1]]
    self.test = data.iloc[:test_shape[0], :test_shape[1]]


    # self.X = self.data.iloc[:, : self.data.shape[1] - 1]
    # self.Y = self.data.iloc[:, self.data.shape[1] - 1]


 #def fit_the_model(self):

 #    self.X = self.train.iloc[:, : self.data.shape[1] - 1]
 #    self.Y = self.train.iloc[:, self.data.shape[1] - 1]

     # Fit the model:
 #    self.fit = GaussianNB()
 #    self.fit.fit(self.X, self.Y)

     # Make predictions on training set:
 #    predictions = self.fit.predict(self.X)

     # Accuracy
 #    accuracy = metrics.accuracy_score(predictions, self.Y)

     # print the accuracy on the training set
 #    print ("the Accuracy of the training set : %s" % "{0:.3%}".format(accuracy))


 def predict(self , pth ): #,test , pth ):

     # Fit the model:
     # self.preprocessing(test)



     y= self.test['class']
     f = open(pth + '/output.txt', "w+")
     frequency_class = self.train['class'].value_counts()
     ClassValues = self.train['class'].unique()
     # print(ClassValues)
     m = 2 # use m-estimator = 2
     i=0
     results = []
     # P_Xk_Ci = {}
     for row in self.test.iterrows():

      P_Xk_Ci = {}
      for classvalue in ClassValues:
       P_Xk_Ci[classvalue] = []
       frequency = frequency_class.loc[classvalue,]

       for columnname in  self.Structure.iterrows():

        if (columnname[1][0] == 'class'):
          break

        p = 1 / len(self.data[columnname[1][0]].unique())

        if(columnname[1][1] != 'NUMERIC'):

         flag = False
         Xk_Ci =  self.train.loc[ (self.train[columnname[1][0]] == row[1][columnname[1][0]]) & (self.train['class'] == classvalue )  ,'class'].count()
         P_Xk_Ci[classvalue].append( (m*p +Xk_Ci)/(m+ frequency))

         # print(P_Xk_Ci )
        else:


         # ranges = self.data[columnname[1][0]].unique()
         # # print(columnname[1][0])
         #
         # # if(ranges.dtype == 'category'):
         # flag = False
         # for i in range(len(ranges)):
         #     # print(ranges[i])
         #     c = str(ranges[i]).split(',')
         #     # if (len(c) > 1):
         #     lowwer = float(c[0][1:])
         #     upper = float(c[1][:len(c[1]) - 1])
         #
         #
         #     if (row[1][columnname[1][0]] > lowwer and row[1][columnname[1][0]] <= upper):
                  # print(row[1][columnname[1][0]])
                  # print(upper)
                  # print(lowwer)

                  Xk_Ci = self.train.loc[(self.train[columnname[1][0]] == row[1][columnname[1][0]]) & (self.train['class'] == classvalue), 'class'].count()
                  # Xk_Ci = self.data.loc[(self.data[columnname[1][0]] == ranges[i]) & (self.data['class'] == classvalue ), 'class'].count()
                  P_Xk_Ci[classvalue].append((m * p + Xk_Ci) / (m + frequency) )
                  # flag =True


         # if(flag == False):
         #             pass

      dic = {}
      for classvalue in ClassValues:
       dic[classvalue] = np.prod(P_Xk_Ci[classvalue])

      srtd = {k: v for k, v in sorted(dic.items(), key=lambda item: item[1] , reverse = True )}
      # print(srtd)
      theclassified = list(srtd.items())[:1][0][0]

      # if (theclassified == 'Y'):
      f.write("%d %s\r\n" % ((i + 1), (theclassified)))
      results.append(theclassified)
      # else:
      #     f.write("%d %s\r\n" % ((i + 1), ('no')))
      #     results.append('N')
      dic = {}

     f.close()

     accuracy = metrics.accuracy_score(results, y)

     # Print Accuracy
     print ("the Accuracy of the test set : %s" % "{0:.3%}".format(accuracy))

     #until here



              # print(float(c[0]))
              # if(row[1][columnname[1][0]] > c and row[1][columnname[1][0]]<= d):
              #     Xk_Ci = self.data.loc[self.data[columnname[1][0]] == ranges[i] , 'class'].count()
              #     P_Xk_Ci[classvalue].append((m * p + Xk_Ci) / m + frequency)

              # Xk_Ci = self.data.loc[self.data[columnname[1][0]] == row[1][columnname[1][0]], 'class'].count()
              # P_Xk_Ci[classvalue].append((m * p + Xk_Ci) / m + frequency)


         # print(row[1][columnname])


     # self.X = self.test.iloc[:, : self.data.shape[1] - 1]
     # self.Y = self.test.iloc[:, self.data.shape[1] - 1]
     #
     # #print(self.X.head())
     # # prediction
     # predictions = self.fit.predict(self.X)
     # accuracy = metrics.accuracy_score(predictions, self.Y)
     #
     # # Print Accuracy
     # print ("the Accuracy of the test set : %s" % "{0:.3%}".format(accuracy))
     #
     # # export the results to output.txt
     # f = open(pth+'/output.txt', "w+")
     #
     # for i, val in enumerate(predictions):
     #
     #     # val == predictions[i] and
     #     # if( val == 1) :
     #      f.write("%d %s\r\n" % ((i + 1),(val)) )
     #     # else:
     #     #     f.write("%d %s\r\n" % ((i + 1),('no')) )
     # f.close()
