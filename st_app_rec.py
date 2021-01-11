from fuzzywuzzy import process
menu = ['Welcome', 'Get recommendations']
st.sidebar.beta_expander("Menu", expanded=False):
option = st.selectbox('Choose', menu)
if option=='Welcome':
    st.title('FILM RECOMMENDATION SYSTEM')
    st.subheader('Collaborative filtering based movie recommendation system.')
    st.write('A user based collaborative filtering algorithm has been used to recommend movies similar to the input provided by the user.
    st.write('Here, the movies are recommended using the Nearest Neighbors clustering algorithm.')
    st.write('The similarity metric used is cosine similarity which calculates the angle between 2 movie vectors.')
else if option=='Get recommendations':
    name=st.text_input('Enter the name of a movie and get recommendations instantaneously.')
    recommend(name)

    def recommend(moviename):
        idx=process.extractBests(moviename,combined['title'])
        print('Movies found: ')
        for i in range(len(idx)):
            print(i+1,' -> ',idx[i][0])
        inp=st.number_input('Choose one option to get recommendations: '))
        print('Movie chosen by you: ',idx[inp][0])
        x=idx[inp][2]
        print('Found at index: ',x)

        dist,ind=knn.kneighbors(csr_mat[x],n_neighbors=10)

        print('Recommendations: ')

        for j in ind:
            print(combined['title'][j].where(j!=x))
