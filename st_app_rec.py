from scipy.sparse import csr_matrix
import pickle
import pandas as pd
import sklearn
from sklearn.neighbors import NearestNeighbors
import streamlit as st
import fuzzywuzzy
from fuzzywuzzy import process
import numpy as np

combined=pd.read_csv(r'data/combined.csv')
knn = pickle.load(open('model/knnpickle_file', 'rb'))
df=combined.pivot_table(values='rating',columns='userId',index='title').fillna(0)
csr_mat=csr_matrix(df.values)
#print(csr_mat)
#csr_mat.shape


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

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

df=pd.read_csv('data/movie_metadata.csv')
#df.isnull().sum()
df=df.loc[:,['movie_title','genres','actor_1_name','actor_2_name','actor_3_name','director_name','language','imdb_score','num_voted_users','plot_keywords']]
#df.head(10)

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
#df['movie_title'][2]
for x in range(len(df['movie_title'])):
    df['movie_title'][x]=df['movie_title'][x][:-1]

df['movie_title'][4]=df['movie_title'][4][:-12]
#df['movie_title'][4]
#df.head()

df['movie_title'].value_counts().sort_values(ascending=False)
df.drop_duplicates(subset='movie_title',inplace=True)
#df['genres'].value_counts()

df['cast']=df['actor_1_name']+' '+df['actor_2_name']+' '+df['actor_3_name']
df['info']=df['plot_keywords']+' '+df['genres']+' '+df['director_name']+' '+df['cast']
#df.head()

vect=TfidfVectorizer(stop_words='english')
matrix=vect.fit_transform(df['info'])
#matrix.shape
sim=linear_kernel(matrix,matrix)
#print(sim)

rev=pd.Series(df.index, index=df['movie_title']).drop_duplicates()


def recommend_cont(film):
    if film not in df['movie_title'].unique():
        print('Title unavailable')
    else:
        i = rev[film]
        l1=list(enumerate(sim[i]))
        l1=sorted(l1, key=lambda x:x[1],reverse=True)
        l1=l1[1:11]
        mov_idx=[x[0] for x in l1]
        st.write(df['movie_title'].iloc[mov_idx])


html = """
	<style>
	.sidebar .sidebar-content {
		background-image: linear-gradient(#33ccff 0%, #ff99ff 100%);
		color: white;
	}
	</style>
	"""
st.markdown(html, unsafe_allow_html=True)


menu = ['Welcome', 'Collaborative filtering','Content based filtering']
with st.sidebar.beta_expander('Menu',expanded=False):
    option = st.selectbox('Choose', menu)
c1,c2=st.beta_columns(2)

if option=='Welcome':
    st.title('FILM RECOMMENDATION SYSTEM')
    st.subheader('Movie recommendation system which uses 2 different approaches to recommend movies similar to one that the user enters.')
    c1.image('images/img1.jpg',use_column_width=True)
    st.write('A user based collaborative filtering algorithm as well as a content based algorithm have been used to recommend movies similar to the input provided by the user.')
    st.write('Here, the movies are recommended using - Nearest Neighbors clustering algorithm or the similarity score.')
    st.write('The Nearest Neighbors clustering algorithm is an unsupervised algorithm which finds training samples nearest to the input using specified distance metric.')
    st.write('The similarity metric used in both cases is called cosine similarity which calculates the similarity in the cosine angle between 2 movie vectors in space.')
    st.write('The cosine similarity is preferred over euclidean distance as it takes the angle into account. So this means that even if 2 data points are placed far apart distance-wise, the angle between the 2 can be taken into consideration.')
    st.write('2 different recommendation algorithms have been provided as options for the user to choose from. Depending on which one they pick, respective recommendations will be made.')

elif option=='Collaborative filtering':
    st.subheader('Collaborative filtering - Recommendations')
    st.write('Recommendations are based on ratings given by users with similar tastes and the nearest neighbors algorithm suggests similar movies.')
    st.write('A pie chart representing the different popular genres of movies in the database is shown below: ')
    st.image('images/pie1.jpg')
    st.write('The number of ratings given for each category i.e. on a scale of 0-5 is represented as a count plot: ')
    st.image('images/hist.jpg')
    st.write("The options shown below are for your reference. Enter any movie to view the model's recommendations.")
    st.write('Choose from the list of 4 movies (which are also recommendations) to get detailed and interesting recommendations.')
    name=st.text_input('Enter the name of a movie and get recommendations instantaneously.')
    recommend_cf(name)

elif option=='Content based filtering':
    st.subheader('Content based filtering - Recommendations')
    st.write('Recommendations are based on plot, genre, cast and director. It makes use of various information given in the dataset to make recommendations by converting such words into vectors using Term Frequency-Inverse Document Frequency method.')
    st.write('The plot below (referred to as the box plot) represents the various range of ratings assigned to different genres.')
    st.image('images/genres2.jpg')
    st.write('The diagram below represents a pie chart based representation of the 10 most popular genres.')
    st.image('images/pie2.jpg')
    st.write('The image below represents the number of movies present in various genres.')
    st.image('images/count1.jpg')
    film=st.text_input('Enter the exact name of any movie and get recommendations instantaneously.')
    recommend_cont(film)
