csr_mat=csr_matrix(df.values)
print(csr_mat)
csr_mat.shape

knn=NearestNeighbors(n_neighbors=10,algorithm='brute',metric='cosine')
knn.fit(csr_mat)
