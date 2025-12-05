# Step 1: Veriyi Hazırlama

import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)

df_movie = pd.read_csv("/Users/esrag/Desktop/a/datasets/movie.csv")
df_rating = pd.read_csv("/Users/esrag/Desktop/a/datasets/rating.csv")

df = df_movie.merge(df_rating, how = "left", on = "movieId")
df.head()

# Toplam oy sayısı 1000'in altında olanlar çıkartılıyor.

rating_count = pd.DataFrame(df["title"].value_counts())
rare_movies = rating_count[rating_count["count"] <= 1000].index
common_movies = df[~df["title"].isin(rare_movies)]

### USER BASED RECOMMENDATION ##################

# Kullanıcılar, filmer ve ratingler için pivot table

df_pivot = pd.pivot_table(data = common_movies, index = "userId", columns = "title", values = "rating")
df_pivot.head()

# Rastgele ID seçme.

random_user = int(pd.Series(df_pivot.index).sample(1, random_state = 123).iloc[0])

# Random kullanıcı için yeni data frame oluşturulması.

random_user_df = df_pivot[df_pivot.index == random_user]
random_user_df.shape

# Random kullanıcının izlediği filmler.

random_user_movies = random_user_df.columns[random_user_df.notna().any()].tolist()

# Datadan sadece random kullanıcıların izlediği verilerin çekilmesi.

movies_watched_df = df_pivot[random_user_movies]
movies_watched_df.head()
movies_watched_df.shape

# Her bir kullanıcının random kullanıcının izlediği filmlerden kaçını izlediği bilgisinin tutulması.

common_watched_movies_count = movies_watched_df.T.notnull().sum()
common_watched_movies_count = common_watched_movies_count.reset_index()
common_watched_movies_count.columns = ["userId", "count"]

# %60 üzerinde ortak izlenmiş film olan userları al.

perc = 0.60 * len(random_user_movies)
user_list = common_watched_movies_count.loc[common_watched_movies_count["count"] > perc, "userId"].tolist()
final_df = movies_watched_df[movies_watched_df.index.isin(user_list)]

# Kullanıcıların birbirleri ile korelasyonlarına bakılması.

corr_df = final_df.T.corr().unstack().sort_values()
corr_df = pd.DataFrame(corr_df)
corr_df.index.names = ["user_1", "user_2"]
corr_df.columns = ["corr"]
corr_df = corr_df.reset_index()
corr_df.head()

# Random Kullanıcımız ile %55 üzeri korelasyon gösteren kullanıcıları top users olarak ayırıyoruz.

top_users = corr_df[(corr_df["user_1"] == random_user) & (corr_df["corr"] > 0.55)][["user_2", "corr"]]
top_users = top_users.sort_values(by='corr', ascending=False)
top_users.rename(columns={"user_2": "userId"}, inplace=True)

# Kendisi ile olan korelasyonundan ayırmak için çıkartıyoruz.
top_users = top_users[~(top_users["userId"] == random_user)]
top_users.shape
top_users.head()

# rating bilgileri ile birleştirilmesi.
top_users_ratings = top_users.merge(df_rating[["userId", "movieId", "rating"]], how = "inner" )
top_users_ratings.head()

# Weighted Average Recommendation hesaplamak için yeni bir metrik tanımlıyoruz.
# weighted_rating = rating * corr

top_users_ratings["weighted_rating"] = top_users_ratings["corr"] * top_users_ratings["rating"]
recommendation_df = top_users_ratings.groupby('movieId').agg({"weighted_rating": "mean"})

# weighted ratingi 2.5 üzeri olan filmleri alıyoruz.

movies_to_be_recommend = recommendation_df[recommendation_df["weighted_rating"] > 2.5].sort_values("weighted_rating", ascending = False)
movies_to_be_recommend = movies_to_be_recommend.reset_index()

top_5_movies_reccomended = movies_to_be_recommend.merge(df_movie[["movieId", "title"]])["title"][:5]

### ITEM BASED RECOMMENDATION ##################

# Seçilen kullanıcının tam puan verdiği filmlerden en güncel olanını alıyoruz.

movie_id = int(df[(df["userId"] == random_user) & (df["rating"] == 5)].sort_values("timestamp", ascending = False)["movieId"].iloc[0])
movie_name = df_movie.loc[df_movie["movieId"] == movie_id, "title"].iloc[0]

# bu filme göre datamızı filtreliyoruz.

movie_df = df_pivot[movie_name]
corr_df_movie = df_pivot.corrwith(movie_df).sort_values(ascending=False).head(10)
corr_df_movie
