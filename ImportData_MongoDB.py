# -*- coding: utf-8 -*-


#import the data
#import pymongo
import pandas as pd
import numpy as np
from pymongo import MongoClient

#Connect to Database
Client = MongoClient('localhost',27017)
db = Client.tripadvisor
collection = db.REVIEW

#Create Dataframe
df = pd.DataFrame(list(collection.find()))
df.shape
df = pd.DataFrame(np.array(df['Review']), columns=['Review'])
df.shape

#export to json
df.to_json('RealData.json',orient = 'records')
df.to_csv('RealData2.csv',sep=',')
