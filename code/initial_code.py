import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from fuzzywuzzy import process
from fuzzywuzzy import fuzz
import streamlit as st

movies=pd.read_csv('movies.csv')
ratings=pd.read_csv('ratings.csv')

movies.shape
movies.head()
ratings.shape
ratings.head()

ratings.groupby(by='movieId')['rating'].mean()
ratings.drop('timestamp',axis='columns',inplace=True)
ratings.drop_duplicates(subset='movieId',inplace=True)
ratings.shape
ratings['rating'].value_counts()

movies['genres']=movies['genres'].str.replace('|',' ')
movies['genres']=movies['genres'].apply(lambda x:x.lstrip().split(' ')[0])
movies['genres'].value_counts()

sns.countplot(ratings['rating'])
sns.boxplot(ratings['rating'],ratings['userId'])
movies['genres'].value_counts()[:10].plot.pie(cmap='Set3')

movies.isnull().sum()
ratings.isnull().sum()
