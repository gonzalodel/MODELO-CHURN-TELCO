# Taken from https://github.com/dmnfarrell/pandastable/wiki/Code-Examples

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
import joblib
from tkinter import *
from pandastable import Table, TableModel
#import tkinter as tk
from tkinter import *
from tkinter import scrolledtext as st
import sys
from tkinter import filedialog as fd
from tkinter import messagebox as mb
from tkinter import messagebox as MessageBox
import os

class TestApp(Frame):
    """Basic test frame for the table"""
    def __init__(self, parent=None):
        self.parent = parent
        Frame.__init__(self)
        self.main = self.master
        self.main.geometry('600x400+200+100')
        self.main.title('Table app')
        
        
        f = Frame(self.main)
        f.pack(fill=BOTH,expand=1)

        modelo = joblib.load("modelo_sna.pkl")
        
        dft = pd.read_csv("sna_churn.csv")
        df = dft.drop('NUMERO', axis=1)
        
        pred0 = []
        pred1 = []
        for y0, y1 in modelo.predict_proba(df):
            pred0.append(y0)
            pred1.append(y1)

        df['PROB_NO_CHURN'] = pred0
        df['PROB_CHURN'] = pred1
        df['ID'] = dft['NUMERO']
        result = df[['ID','PROB_NO_CHURN','PROB_CHURN']]
        
        self.table = pt = Table(f, dataframe=result,
                                showtoolbar=True, showstatusbar=True)
        pt.show()
        return

app = TestApp()
#launch the app
app.mainloop()