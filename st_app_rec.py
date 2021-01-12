from scipy.sparse import csr_matrix
import pickle
import pandas as pd
import sklearn
from sklearn.neighbors import NearestNeighbors
import streamlit as st
import fuzzywuzzy
from fuzzywuzzy import process
combined=pd.read_csv(r'data/combined.csv')
knn = pickle.load(open('model/knnpickle_file', 'rb'))

df=combined.pivot_table(values='rating',columns='userId',index='title').fillna(0)

csr_mat=csr_matrix(df.values)
#print(csr_mat)
#csr_mat.shape


menu = ['Welcome', 'Recommendations-Collaborative filtering','Recommendations-Content based filtering']

option = st.sidebar.selectbox('Choose', menu)

def recommend_cf(moviename):
    idx=process.extractBests(moviename,combined['title'])
    st.write('Movies found: ')
    options=[]
    for i in range(len(idx)-1):
        options.append(idx[i][0])
    inp1=st.radio('Choose one to get recommendations: ',options)
    for n in range(len(options)):
        if options[n]==inp1:
            inp=n
    #inp= [x for x in range(len(options)) if options[x]==inp1]
    print(inp)
    st.write('Movie chosen by you: ',idx[inp+1][0])
    x=idx[inp+1][2]
    #st.write('Found at index: ',x)

    dist,ind=knn.kneighbors(csr_mat[x],n_neighbors=10)

    st.write('Recommendations: ')

    for j in ind:
        st.write(combined['title'][j].where(j!=x))

def recommend_cont(title):
    if title not in df['movie_title'].unique():
        print('Title unavailable')
    else:
        i= rev[title]
        l1=list(enumerate(sim[i]))
        l1=sorted(l1, key=lambda x:x[1],reverse=True)
        l1=l1[1:11]
        mov_idx=[x[0] for x in l1]
        return df['movie_title'].iloc[mov_idx]

if option=='Welcome':
    st.title('FILM RECOMMENDATION SYSTEM')
    st.subheader('Collaborative filtering based movie recommendation system.')
    st.image('data/img1.jpg')
    st.write('A user based collaborative filtering algorithm has been used to recommend movies similar to the input provided by the user.')
    st.write('Here, the movies are recommended using the Nearest Neighbors clustering algorithm.')
    st.write('The similarity metric used is cosine similarity which calculates the angle between 2 movie vectors.')


elif option=='Recommendations-Collaborative filtering':
    st.subheader('Collaborative filtering - Recommendations')
    st.write('Recommendations based on user similarity and nearest neighbors algorithm for clustering similar movies.')
    st.write('The different genres of movies in our database is shown below: ')
    st.image('data/pie1.jpg')
    st.write('The number of ratings given for each category i.e. on a scale of 0-5 is represented as a histogram: ')
    st.image('data/hist.jpg')
    st.write("The options shown below, are for your reference. Enter any movie's name to get respective recommendations.")
    name=st.text_input('Enter the name of a movie and get recommendations instantaneously.')
    recommend_cf(name)

elif option=='Recommendations-Content based filtering':
    st.subheader('Collaborative filtering - Recommendations')
    st.write('Recommendations based on plot, genre, cast and director. It makes use of various information given in the dataset to make recommendations')
    st.write('The image below represents the various ratings assigned to different genres.')
    st.image('images/genres2.jpg')
    st.write('The image below represents a pie chart based representation of the 10 most popular genres.')
    st.image('images/pie2.jpg')
    st.write('The image below represents the number of movies present in various genres.')
    st.image('images/count1.jpg')
    film=st.text_input('Enter the name of a movie and get recommendations instantaneously.')
    recommend_cont(film)
