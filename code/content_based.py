import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

df=pd.read_csv('movie_metadata.csv')
df.isnull().sum()
df=df.loc[:,['movie_title','genres','actor_1_name','actor_2_name','actor_3_name','director_name','language','imdb_score','num_voted_users','plot_keywords']]
df.head(10)

df['genres']=df['genres'].str.replace('|',' ')
df['plot_keywords']=df['plot_keywords'].str.replace('|',' ')
df['genres']=df['genres'].apply(lambda x:x.lstrip().split(' ')[0])
df['actor_1_name']=df['actor_1_name'].replace(np.nan,'unknown')
df['actor_2_name']=df['actor_2_name'].replace(np.nan,'unknown')
df['actor_3_name']=df['actor_3_name'].replace(np.nan,'unknown')
df['language']=df['language'].replace(np.nan,'unknown')
df['director_name']=df['director_name'].replace(np.nan,'unknown')
df['plot_keywords']=df['plot_keywords'].replace(np.nan,'unknown')
df['actor_1_name']=df['actor_1_name'].str.replace(' ','')
df['actor_2_name']=df['actor_2_name'].str.replace(' ','')
df['actor_3_name']=df['actor_3_name'].str.replace(' ','')
df['director_name']=df['director_name'].str.replace(' ','')

df['genres']=df['genres'].str.lower()
df['movie_title']=df['movie_title'].str.lower()
df['movie_title'][2]
for x in range(len(df['movie_title'])):
    df['movie_title'][x]=df['movie_title'][x][:-1]

df['movie_title'][4]=df['movie_title'][4][:-12]
#df['movie_title'][4]
df.head()

df['movie_title'].value_counts().sort_values(ascending=False)
df['genres'].value_counts()

df['cast']=df['actor_1_name']+' '+df['actor_2_name']+' '+df['actor_3_name']
df['info']=df['plot_keywords']+' '+df['genres']+' '+df['director_name']+' '+df['cast']
df.head()

vect=TfidfVectorizer(stop_words='english')
matrix=vect.fit_transform(df['info'])
#matrix.shape
sim=linear_kernel(matrix,matrix)
#print(sim)

rev=pd.Series(df.index, index=df['movie_title']).drop_duplicates()
