# Movie-Recommender-System
A recommender system is an application that takes in the name of a certain movie, book, tv show, song, etc. and using certain mathematical/machine learning based algorithms to find similar titles.
![Movie_posters](images/img1.jpg)
These titles are then suggested to the user. Different algorithms have been used in this particular application, to recommend movies similar to the input provided by the user.

Here, the movies are recommended using 2 different approaches:
1. Nearest Neighbors clustering algorithm. The similarity metric used is cosine similarity which calculates the angle between 2 movie vectors.
2. Content based recommendations that filter user preferences based on movie's genre, plot description, cast/crew and director.

How does collaborative filtering work?
A large dataset consisting of different movies and their details, is used as a reference for predicting what other movies the user might enjoy watching.
The algorithm used here is based on unsupervised clustering of movies according to their cosine similarities.

How does content based filtering work?
A dataset with different movies, their genres, cast, etc. is present and recommendations are picked by the algorithm based on the value of similarity between the user's input and movies in the table.
