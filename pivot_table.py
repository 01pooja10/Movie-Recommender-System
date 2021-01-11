combined=pd.merge(ratings,movies,on='movieId')
combined.head()
combined.shape
combined.drop('genres',axis='columns',inplace=True)

totalCount=(combined.groupby(by='title')['rating'].count().reset_index())
totalCount.rename(columns={'rating':'count'},inplace=True)
totalCount.head()
combined=combined.merge(totalCount,on='title')
print(combined['count'].describe())


df=combined.pivot_table(values='rating',columns='userId',index='title').fillna(0)
df.shape
df.head()
df.reset_index()
