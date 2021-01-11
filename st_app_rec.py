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


menu = ['Welcome', 'Get recommendations']

option = st.sidebar.selectbox('Choose', menu)

def recommend(moviename):
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

if option=='Welcome':
    st.title('FILM RECOMMENDATION SYSTEM')
    st.subheader('Collaborative filtering based movie recommendation system.')
    st.image('data/img1.jpg')
    st.write('A user based collaborative filtering algorithm has been used to recommend movies similar to the input provided by the user.')
    st.write('Here, the movies are recommended using the Nearest Neighbors clustering algorithm.')
    st.write('The similarity metric used is cosine similarity which calculates the angle between 2 movie vectors.')
elif option=='Get recommendations':
    st.subheader('Recommendation system')
    st.write("Welcome. The options shown below are for your reference. Enter any movie's name to get recommendations.")
    name=st.text_input('Enter the name of a movie and get recommendations instantaneously.')
    recommend(name)
