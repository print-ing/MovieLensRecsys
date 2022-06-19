import json
import numpy as np
import pandas as pd
import hashlib
import unittest

from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# User ----------------------------------
class User:

    def __init__(self):
        self.userId = 0
        self.df_user = pd.read_csv("user.csv", index_col=0)
        pass


    def signup(self):
        user_info = []
        print("회원가입 진행합니다")

        # userId 부여
        userId_new = int(self.df_user['userId'].iloc[self.df_user.shape[0] - 1]) + 1
        user_info.append(userId_new)
        while True:
            # id 중복확인 및 추가
            id = input("id:")
            if (self.df_user['id'] != id).all():
                break
            print("중복된 아이디입니다. 다시 입력해주세요.")
        user_info.append(id)

        # pwd 암호화 및 추가
        string = input("pwd:")
        encoded_string = string.encode('utf-8')
        pwd = hashlib.sha256(encoded_string).hexdigest()
        user_info.append(pwd)

        # Dataframe으로 만들기
        df_user_new = pd.DataFrame([user_info], columns=['userId', 'id', 'pwd'])
        self.df_user = pd.concat([self.df_user, df_user_new], ignore_index=True, axis = 0)
        self.df_user = self.df_user.reset_index(drop = True)
        self.df_user.to_csv("user.csv", encoding = "utf-8-sig")

        return userId_new

    def login(self):
        while True:
            id = input("아이디를 입력해주세요:")
            if (self.df_user['id'] == id).any():
                self.userId = self.df_user.index[self.df_user['id'] == id][0]
                break
            else:
                print("아이디가 존재하지 않습니다. 다시 입력해주세요.")
        while True:
            string = input("비밀번호를 입력해주세요:")
            encoded_string = string.encode('utf-8')
            pwd = hashlib.sha256(encoded_string).hexdigest()
            if self.df_user['pwd'][self.userId] == pwd:
                break
            else:
                print("비밀번호가 맞지 않습니다. 다시 입력해주세요.")
        print("로그인 완료했습니다.")
        now_userId = self.df_user['userId'][self.userId]
        return now_userId

    def logout(self):
        self.userId = 0

# Movie -------------------------------
class Movie:
    with open('final_movie_dict.json') as file:
        movie = json.load(file)
    with open('genres_dict.json') as file:
        genre = json.load(file)  # {movieId : [title, year,genres,story,director,[actors]}

    def __init__(self):
        self.temp_data = TempData()
        # self.temp_data2 = TempData()

    def print_10_movie(self):
        # movieId 목록을 받아와서 10개씩 출력
        ten_movie_list = self.temp_data.get_n_movies()
        cnt = 0
        for i in ten_movie_list:
            cnt += 1
            print('[{}] {}_movieId : {}'.format(cnt, Movie.movie[i][0], i))

        # pickMe 더보기, repet 증가
        # pickMe topMovie 종료
        # break

    def print_recys_movie(self, recys_movie_list):
        self.temp_data.set_movie_list(recys_movie_list)
        self.print_10_movie()

    def print_movies(self):
        movie_id = input("movieId")
        print("[{}] {}_{} <{}>".format(movie_id, Movie.movie[movie_id][0], Movie.movie[movie_id][1],
                                       Movie.movie[movie_id][2]))

    def all_imformation(self):
        # movie 정보 출력
        movie_id = input("\tmovieId: ")
        print("")
        print("title:" + Movie.movie[movie_id][0])
        print("year:" + Movie.movie[movie_id][1])
        print("genres:" + Movie.movie[movie_id][2])
        print("director:" + Movie.movie[movie_id][4])
        print("actors:" + Movie.movie[movie_id][5])
        print("story:" + Movie.movie[movie_id][3])
    # pickMe 평점 입력

    def print_genres(self):
        for i, genre_name in enumerate(Movie.genre):
            print('[{}] {}'.format(i, genre_name))
        self.input_genres()

    def input_genres(self):
        index = int(input("genre index"))
        now_genre = list(Movie.genre.keys())[index]
        self.temp_data.set_movie_list(Movie.genre[now_genre])
        print("")

    # movieId list 받아서 abc, 연도순 정렬
    def sort_movie_by_title(self, movie_list):
        # {movieId : [title, year,genres,story,derector,[actors]}의 형태일 때
        # 장르 내 영화 딕셔너리
        genre_movie = {}
        for movie_id in movie_list:
            genre_movie[movie_id] = Movie.movie[movie_id]
        genre_movie = dict(sorted(genre_movie.items(), key=lambda x: x[1][0]))
        self.temp_data.set_movie_list(list(genre_movie.keys()))
        self.print_10_movie()

    def sort_movie_by_year(self, movie_list):
        # 최신순 정렬
        genre_movie = {}
        for movie_id in movie_list:
            genre_movie[movie_id] = Movie.movie[movie_id]
        genre_movie = dict(sorted(genre_movie.items(), key=lambda x: x[1][1], reverse=True))
        self.temp_data.set_movie_list(list(genre_movie.keys()))
        self.print_10_movie()

# Data -------------------------------
class TempData:
    movie_list = []
    score_list = []
    user_list = []
    user_id = ''

    def init_movie_list(self):
        TempData.movie_list = []

    def init_score_list(self):
        TempData.score_list = []

    # 정렬되거나 선택한 장르의 영화 목록 등 임시 데이터
    def set_movie_list(self, received_movie_list):
        TempData.movie_list.append(received_movie_list)

    def get_movie_list(self, index=0):
        index_movie_list = self.movie_list[index]
        del(self.movie_list[index])
        return index_movie_list

    def get_n_movies(self, n=10, index=0):
        n_movies = []
        for i in range(n):
            if len(TempData.movie_list[index]) != 0:
                n_movies.append(TempData.movie_list[index][0])
                del(TempData.movie_list[index][0])
        return n_movies

    def set_user_list(self, received_user_list):
        TempData.user_list = received_user_list

    def get_user_id(self):
        return TempData.user_id

    def set_score_list(self, received_score_list):
        TempData.score_list.append(received_score_list)

    def get_n_score(self, n=10, index=0):
        n_scores = []
        for i in range(n):
            if len(TempData.score_list[index]) != 0:
                n_scores.append(TempData.score_list[index][0])
                del(TempData.score_list[index][0])
        return n_scores

# Rating -------------------------------
class Rating:
    with open("rating_dict.json") as file:
        rating = json.load(file)
    ranking = {}
    rating_csv = pd.read_csv("ratings.csv")

    def __init__(self):
        self.movie = Movie()
        self.temp_data = TempData()

    def init_ranking(self, received_movie_list):
        for movie_id in received_movie_list:
            Rating.ranking[movie_id] = [0, 0]

    def set_ranking(self, received_movie_list):
        # ranking = {movieId:[rating, view]}
        # 전체 평가
        for i in Rating.rating.values():
            for movie_id, score in i.items():
                if movie_id in received_movie_list:
                    Rating.ranking[movie_id][0] += score
                    Rating.ranking[movie_id][1] += 1

    # 아직 데이터 받아오는 부분을 구현하지 않아서 기본값 설정해뒀음
    def input_rating(self, user_id='1'):
        movie_id = str(input("movieId 입력"))
        movie_rating = float(input("평점 입력"))
        if movie_rating > 0 and movie_rating <= 5:
            self.set_movie_rating(movie_rating, user_id, movie_id)

    def sort_movie_by_ranking(self, received_movie_list, min_views=50):
        # rating : {userId: {movieId: rating}}
        # ranking : {movieId: [sum(movie_rating), num_ratings]}
        # 전체 영화
        self.init_ranking(received_movie_list)
        self.set_ranking(received_movie_list)
        removing_key = []

        # 평균 평점 구하기, 기준 조회수 아래 랭킹에서 삭제하기
        for movie_id in Rating.ranking.keys():
            if Rating.ranking[movie_id][1] < min_views:
                removing_key.append(movie_id)
            if Rating.ranking[movie_id][1] != 0:
                Rating.ranking[movie_id][0] /= Rating.ranking[movie_id][1]

        for movie_id in removing_key:
            del (Rating.ranking[movie_id])

        # 평점, 조회수로 내림차순 정렬
        Rating.ranking = dict(sorted(Rating.ranking.items(), key=lambda x: x[1], reverse=True))
        self.temp_data.set_movie_list(list(Rating.ranking.keys()))
        self.movie.print_10_movie()

    def set_user_history(self, user_id):
        self.temp_data.set_movie_list(list(Rating.rating[user_id].keys()))
        self.temp_data.set_score_list(list(Rating.rating[user_id].values()))

    def print_10_user_history(self, n=10):
        my_movies = self.temp_data.get_n_movies(n)
        my_scores = self.temp_data.get_n_score(n)
        for i in range(len(my_movies)):
            print("★X{}, [{}] {}- {} <{}>".format(my_scores[i], my_movies[i], Movie.movie[my_movies[i]][0], Movie.movie[my_movies[i]][1], Movie.movie[my_movies[i]][2]))

    def set_movie_rating(self, rating, user_id, movie_id):
        r_dict = {'userId': [user_id], 'movieId': [movie_id], 'rating': [rating]}
        r_df = pd.DataFrame(r_dict)
        Rating.rating_csv = pd.concat([Rating.rating_csv, r_df])
        Rating.rating_csv.to_csv("ratings.csv", encoding="utf-8-sig",index = False)
        if user_id not in Rating.rating.keys():
            Rating.rating[user_id] = {movie_id: rating}
        else:
            Rating.rating[user_id].setdefault(movie_id, rating)
        with open('rating_dict.json', 'w') as f:
            json.dump(Rating.rating, f, indent=4)

    '''
        def print_history(self, user_id):
            cnt = 0
            user_history = self.rating[user_id]
            for movie_id in user_history.keys():
                # 영화 평점 출력
                print(user_history[movie_id])
                # 영화 제목, 장르 출력
                movie.print_title_genres(movie_id)
                cnt += 1

                if cnt % 10 == 0:
                    pickme.extensions()
    '''
    '''
    def print_history(self, user_id):

        user_history = self.rating[user_id]
        movie_id = user_history.keys()
        self.temp_data.set_movie_list
            # 영화 평점 출력
            print(user_history[movie_id])
            # 영화 제목, 장르 출력
            self.movie.print_title_genres(movie_id)

    '''

# Search --------------------------------
class Search:

    def __init__(self):
        self.df = pd.read_csv("movie_real_final.csv")
        self.option = 0
        self.keyword = ""
        self.sim_scores = []

        self.movie = Movie()
        self.temp_data = TempData()
        pass  # import movie DB

    def set_search_option(self):
        self.option = int(input("1) 제목 검색 2) 본문 검색"))
        self.set_keyword()

    def set_keyword(self):
        word = input("검색어를 입력해주세요")
        self.keyword = word
        self.get_movielist()
        # print(self.get_movielist())

    def get_movielist(self):
        opt = self.option
        if opt == 1:
            sim_scores_list = self.cal_similarity_1()
            self.sim_scores = [str(self.df['movieId'].iloc[idx]) for idx, score in sim_scores_list]
        elif opt == 2:
            sim_scores_list = self.cal_similarity_2()
            self.sim_scores = [str(self.df['movieId'].iloc[idx]) for idx, score in sim_scores_list]
        else:
            print("GO BACK")
        self.temp_data.set_movie_list(self.sim_scores)
        self.movie.print_10_movie()


    def cal_similarity_1(self):
        data = self.df[['title']]
        data_keyword = pd.DataFrame([self.keyword], columns=['title'])
        data_new = pd.concat([data, data_keyword], ignore_index = True)
        tfidf = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf.fit_transform(data_new['title'])
        cosine_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

        # movie title와 id를 매핑할 dictionary를 생성해줍니다.
        title2idx = {}
        for i, c in enumerate(data_new['title']):
            title2idx[i] = c

        # id와 movie title를 매핑할 dictionary를 생성해줍니다.
        idx2title = {}
        for i, c in title2idx.items():
            idx2title[c] = i

        word = self.keyword
        idx = idx2title[word]
        sim_scores = [[i, c] for i, c in enumerate(cosine_matrix[idx]) if i != idx]  # 자기 자신을 제외한 영화들의 유사도 및 인덱스를 추출
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)  # 유사도가 높은 순서대로 정렬
        return sim_scores

    def cal_similarity_2(self):
        data = self.df[['story']]
        data_keyword = pd.DataFrame([self.keyword], columns=['story'])
        data_new = pd.concat([data, data_keyword], ignore_index = True)
        tfidf = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf.fit_transform(data_new['story'])
        cosine_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

        # movie title와 id를 매핑할 dictionary를 생성해줍니다.
        story2idx = {}

        for i, c in enumerate(data_new['story']):
            story2idx[i] = c

        # id와 movie title를 매핑할 dictionary를 생성해줍니다.
        idx2story = {}
        for i, c in story2idx.items():
            idx2story[c] = i

        word = self.keyword
        idx = idx2story[word]
        sim_scores = [[i, c] for i, c in enumerate(cosine_matrix[idx]) if i != idx]  # 자기 자신을 제외한 영화들의 유사도 및 인덱스를 추출
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)  # 유사도가 높은 순서대로 정렬
        return sim_scores

# Recys ----------------------------------
'''
class RecsysSeries():
    """유저 기반"""
    # movie 데이터프레임
    movies = pd.read_csv("movie_real_final.csv", encoding='utf-8')

    # rating 데이터프레임
    ratings = pd.read_csv("ratings.csv", encoding='utf-8')

    x = ratings.copy()
    y = ratings['userId']

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, stratify=y)

    # rating 함수 행렬 분해
    rating_matrix = x_train.pivot(index='userId',
                                  columns='movieId',
                                  values='rating')

    # 사용자 간의 유사도- user similarity
    matrix_dummy = rating_matrix.copy().fillna(0)
    user_similarity = cosine_similarity(matrix_dummy, matrix_dummy)
    user_similarity = pd.DataFrame(user_similarity,
                                   index=rating_matrix.index,
                                   columns=rating_matrix.index)

    rating_matrix_t = np.transpose(rating_matrix)
    matrix_dummy_t = rating_matrix_t.copy().fillna(0)
    item_similarity = cosine_similarity(matrix_dummy_t, matrix_dummy_t)
    item_similarity = pd.DataFrame(item_similarity,
                                   index=rating_matrix_t.index,
                                   columns=rating_matrix_t.index)


    def CF_Knn(self, userId, movieId, neighbor_size=0):
        if movieId in RecsysSeries.rating_matrix.columns:
            sim_scores = RecsysSeries.user_similarity[userId].copy()
            movie_ratings = RecsysSeries.rating_matrix[movieId].copy()
            none_rating_idx = movie_ratings[movie_ratings.isnull()].index  # 평점을 매기지 않은 사용자를 제외
            movie_ratings = movie_ratings.dropna()
            sim_scores = sim_scores.drop(none_rating_idx)  # 평점을 매기지 않은 사람을 유사도에서도 제외

            if neighbor_size == 0 and sim_scores.sum() != 0:
                mean_rating = np.dot(sim_scores, movie_ratings) / sim_scores.sum()
            else:
                if len(sim_scores) > 1:
                    neighbor_size = min(neighbor_size, len(sim_scores))
                    sim_scores = np.array(sim_scores)
                    movie_ratings = np.array(movie_ratings)
                    user_idx = np.argsort(sim_scores)
                    sim_scores = sim_scores[user_idx][-neighbor_size:]
                    movie_ratings = movie_ratings[user_idx][-neighbor_size:]
                    if sim_scores.sum() != 0:
                        mean_rating = np.dot(sim_scores, movie_ratings) / sim_scores.sum()
                    else:
                        mean_rating = 2.5
                else:
                    mean_rating = 2.5
        else:
            mean_rating = 2.5

        return mean_rating

    """아이템 기반"""

    def CF_item(self, user_id, movie_id):
        if movie_id in RecsysSeries.item_similarity.columns:
            sim_scores = RecsysSeries.item_similarity[movie_id]
            user_rating = RecsysSeries.rating_matrix_t[user_id]
            none_rating_idx = user_rating[user_rating.isnull()].index
            user_rating = user_rating.dropna()
            sim_scores = sim_scores.drop(none_rating_idx)
            if sim_scores.sum() != 0:
                mean_rating = np.dot(sim_scores, user_rating) / sim_scores.sum()
            else:
                mean_rating = 3.0
        else:
            mean_rating = 3.0

        return mean_rating

    def similar_user_list(self, user = 1):
        user_similarity_copy = RecsysSeries.user_similarity.copy()
        np.fill_diagonal(user_similarity_copy.values, 0)

        sim_list = list(user_similarity_copy.iloc[user - 1])
        sim_list_sort = sorted(sim_list, reverse=True)

        sim_userid_list = []
        for i in range(10):
            sim_user = str(sim_list.index(sim_list_sort[i]) + 1)
            sim_userid_list.append(sim_user)
        return sim_userid_list

class ListMakingSeries(RecsysSeries):

    def recom_movie_1(self, userId, n_items, neighbor_size=30):
        user_movie = RecsysSeries.rating_matrix.loc[userId].copy()

        recsys_list = []
        for movie in RecsysSeries.rating_matrix.columns:
            if pd.notnull(user_movie.loc[movie]):
                user_movie.loc[movie] = 0
            else:
                user_movie.loc[movie] = RecsysSeries.CF_Knn(self, userId, movie, neighbor_size)

            movie_sort = user_movie.sort_values(ascending=False)[:n_items]
            for i in range(n_items):
                index = movie_sort.index[i]

                index_str = index.astype(str)
                recsys_list.append(index_str)

            break

        return recsys_list


    def recom_movie_2(self, userId, n_items):
        # user_movie = rating_matrix_t.loc[userId].copy()
        user_movie = RecsysSeries.rating_matrix_t[userId].copy()

        recsys_list = []
        for movie in RecsysSeries.rating_matrix_t.index:
            if pd.notnull(user_movie.loc[movie]):
                user_movie[movie] = 0
            else:
                user_movie[movie] = RecsysSeries.CF_item(self, userId, movie)

            movie_sort = user_movie.sort_values(ascending=False)[:n_items]
            for i in range(n_items):
                index = movie_sort.index[i]

                index_str = index.astype(str)
                recsys_list.append(index_str)

            break

        return recsys_list


class MakeFinalList(ListMakingSeries):

    def select_recsys(self, option, userid=1):
        if option == 1:
            rec_list1 = ListMakingSeries.recom_movie_1(self, userId=userid, n_items=100, neighbor_size=30)
            return rec_list1

        elif option == 2:
            rec_list2 = ListMakingSeries.recom_movie_2(self, userId=userid, n_items=100)
            return rec_list2

        else:
            pass
'''

class RecsysSeries:
    """유저 기반"""
    # movie 데이터프레임
    movies = pd.read_csv("movie_real_final.csv", encoding='utf-8')

    # rating 데이터프레임
    ratings = pd.read_csv("ratings.csv", encoding='utf-8')

    x = ratings.copy()
    y = ratings['userId']

    # rating 함수 행렬 분해
    rating_matrix = x.pivot(index='userId',
                            columns='movieId',
                            values='rating')

    # 사용자 간의 유사도- user similarity
    matrix_dummy = rating_matrix.copy().fillna(0)
    user_similarity = cosine_similarity(matrix_dummy, matrix_dummy)
    user_similarity = pd.DataFrame(user_similarity,
                                   index=rating_matrix.index,
                                   columns=rating_matrix.index)

    rating_matrix_t = np.transpose(rating_matrix)
    matrix_dummy_t = rating_matrix_t.copy().fillna(0)
    item_similarity = cosine_similarity(matrix_dummy_t, matrix_dummy_t)
    item_similarity = pd.DataFrame(item_similarity,
                                   index=rating_matrix_t.index,
                                   columns=rating_matrix_t.index)

    def CF_Knn(self, userId, movieId, neighbor_size=0):
        if movieId in RecsysSeries.rating_matrix.columns:
            sim_scores = RecsysSeries.user_similarity[userId].copy()
            movie_ratings = RecsysSeries.rating_matrix[movieId].copy()
            none_rating_idx = movie_ratings[movie_ratings.isnull()].index  # 평점을 매기지 않은 사용자를 제외
            movie_ratings = movie_ratings.dropna()
            sim_scores = sim_scores.drop(none_rating_idx)  # 평점을 매기지 않은 사람을 유사도에서도 제외

            if neighbor_size == 0 and sim_scores.sum() != 0:
                mean_rating = np.dot(sim_scores, movie_ratings) / sim_scores.sum()
            else:
                if len(sim_scores) > 1:
                    neighbor_size = min(neighbor_size, len(sim_scores))
                    sim_scores = np.array(sim_scores)
                    movie_ratings = np.array(movie_ratings)
                    user_idx = np.argsort(sim_scores)
                    sim_scores = sim_scores[user_idx][-neighbor_size:]
                    movie_ratings = movie_ratings[user_idx][-neighbor_size:]
                    if sim_scores.sum() != 0:
                        mean_rating = np.dot(sim_scores, movie_ratings) / sim_scores.sum()
                    else:
                        mean_rating = 2.5
                else:
                    mean_rating = 2.5
        else:
            mean_rating = 2.5

        return mean_rating

    """아이템 기반"""

    def predict_rating(self, ratings_arr, item_sim_arr):  # 평점데이터, item유사도 데이터
        # (평점 x item 유사도) / (item 유사도의 합)
        ratings_pred = ratings_arr.dot(item_sim_arr) / np.array([np.abs(item_sim_arr).sum(axis=1)])
        return ratings_pred

    def get_unseen_movies(self, ratings_matrix, userId):

        user_rating = ratings_matrix.loc[userId, :]

        already_seen = user_rating[user_rating > 0].index.tolist()

        movies_list = ratings_matrix.columns.tolist()

        unseen_list = [movie for movie in movies_list if movie not in already_seen]

        return unseen_list

    def recomm_movie_by_userid(self, pred_df, userId, unseen_list, top_n=10):
        recomm_movies = pred_df.loc[userId, unseen_list].sort_values(ascending=False)[:top_n]
        recomm_movies_list = recomm_movies.index.astype('str').tolist()  # --------!!!!!!!!!! 수정~~~~~~!!!!!!!!!!!!!!
        return recomm_movies_list

    def item_list(self, user_id=1):
        ratings_pred = self.predict_rating(self.matrix_dummy.values, self.item_similarity.values)

        ratings_pred_matrix = pd.DataFrame(data=ratings_pred, index=self.matrix_dummy.index,
                                           columns=self.matrix_dummy.columns)
        unseen_list = self.get_unseen_movies(self.matrix_dummy, user_id)
        recomm_movies = self.recomm_movie_by_userid(ratings_pred_matrix, user_id, unseen_list, top_n=100)
        return recomm_movies

    def similar_user_list(self, user=1):
        user_similarity_copy = RecsysSeries.user_similarity.copy()
        np.fill_diagonal(user_similarity_copy.values, 0)

        sim_list = list(user_similarity_copy.iloc[user - 1])
        sim_list_sort = sorted(sim_list, reverse=True)

        sim_userid_list = []
        for i in range(10):
            sim_user = str(sim_list.index(sim_list_sort[i]) + 1)
            sim_userid_list.append(sim_user)
        return sim_userid_list


class ListMakingSeries(RecsysSeries):

    def recom_movie_1(self, userId, n_items, neighbor_size=30):
        user_movie = RecsysSeries.rating_matrix.loc[userId].copy()

        recsys_list = []
        for movie in RecsysSeries.rating_matrix.columns:
            if pd.notnull(user_movie.loc[movie]):
                user_movie.loc[movie] = 0
            else:
                user_movie.loc[movie] = RecsysSeries.CF_Knn(self, userId, movie, neighbor_size)

            movie_sort = user_movie.sort_values(ascending=False)[:n_items]
            for i in range(n_items):
                index = movie_sort.index[i]
                index_str = index.astype(str)
                recsys_list.append(index_str)
            break
        return recsys_list

    def select_recsys(self, userid=1):
        rec_list1 = self.recom_movie_1(userId=userid, n_items=100, neighbor_size=30)
        return rec_list1


# PicK Me-----------------------------------
class PickMe:
    def __init__(self):
        self.rating = Rating()
        # self.rating.init_ranking()
        self.movie = Movie()
        self.temp_data = TempData()
        # self.list_making_series = ListMakingSeries()
        self.recsys_series = RecsysSeries()
        self.list_making_series = ListMakingSeries()
        self.search = Search()
        self.user = User()
        self.userId = ''

    def main_entrance(self):
        option_ = int(input("\t[1]: sign up  [2]: login >>"))
        if option_ == 1:
            self.userId = str(self.user.signup())
            self.rating.sort_movie_by_ranking(self.movie.movie.keys())
            for i in range(10):
                print("남은 영화", 10-i)
                self.rating.input_rating(self.userId)
            self.temp_data.init_movie_list()

        elif option_ == 2:
            self.userId = str(self.user.login())

        while 1:
            option = int(input("\t[1]: recsys  [2]: top movie  [3]: genre  [4]: history  [5]: search  [6]: exit >>"))
            print("")

            if option == 1:
                option2 = int(input("\t[1]: movie user  [2]: movie item  [3]: similar user >>"))
                print("")
                if option2 == 1:
                    recsys = self.list_making_series.select_recsys(int(self.userId))
                    self.movie.print_recys_movie(recsys)
                    print("")
                elif option2 == 2:
                    recsys2 = self.recsys_series.item_list(int(self.userId))
                    self.movie.print_recys_movie(recsys2)
                    # print(recsys2)
                    print("")

                elif option2 == 3:
                    now_user = 1
                    recsys = self.recsys_series.similar_user_list(now_user)
                    # """" 유저 id 받고 유저 아이디 출력 """"
                    print(recsys)
                    # """ 유저 히스토리
                    userid = str(input("userID >>"))
                    #user ID 입력받고 출력
                    self.rating.set_user_history(userid)
                    self.rating.print_10_user_history()
                    print("")
                    self.extensions(1)


            elif option == 2:
                self.rating.sort_movie_by_ranking(self.movie.movie.keys())
                print("")
                self.extensions()

            elif option == 3:
                self.movie.print_genres()
                self.movie_sorting_option()
                print("")
                self.extensions()

            elif option == 4:
                self.rating.set_user_history(str(self.userId))
                self.rating.print_10_user_history()
                #히스토리
                self.extensions(1)
                # pass

            elif option == 5:
                self.search.set_search_option()
                self.search.set_keyword
                self.search.get_movielist
                self.extensions()
                # pass

            elif option == 6:
                #종료
                break

            #self.extensions()

    def extensions(self, entrance_option=0):
        while 1:
            option = int(input("\t[1]: initial\t[2]: more\t[3]: movie information >>"))
            if option == 1:
                # 처음으로 선택시에 저장된 임시 영화 리스트 초기화
                print("")
                self.temp_data.init_movie_list()
                self.temp_data.init_score_list()
                break
            elif option == 2:
                if entrance_option == 1:
                    self.rating.print_10_user_history()
                else:
                    self.movie.print_10_movie()
                print("")
            elif option == 3:
                self.movie.all_imformation()
                print("")
                self.movie_rating_option()



    def movie_rating_option(self):
        option = int(input("\t[1]: initial\t[2]: rate movie >>"))
        print("")
        if option == 2:
            self.rating.input_rating()

    def movie_sorting_option(self):
        option = int(input(("\t[1]: rating\t[2]: ABC\t[3]: latest >>")))
        if option == 1:
            self.rating.sort_movie_by_ranking(self.temp_data.get_movie_list())
        elif option == 2:
            self.movie.sort_movie_by_title(self.temp_data.get_movie_list())
        elif option == 3:
            self.movie.sort_movie_by_year(self.temp_data.get_movie_list())

class TestMovie(unittest.TestCase):
    # def setup(self):

    def test_start(self):
        self.PickMe = PickMe()
        self.PickMe.main_entrance()

# rating = Rating()
# rating.init_ranking()
# rating.calculate_ranking()
# movie = Movie()
if __name__ == "__main__":
    #
    # user = User()
    # userid = user.enter()
    # print(userid)

    pickme = PickMe()
    pickme.main_entrance()

# while True:
#     option = int(input("선택지 고르세요: "))
#     if option == 1:
#         print("다음과 같은 영화를 출력합니다.")
#         recsys1 = make_final_list(option)
#         recsys1.select_recsys()
#     else:
#         print("유사한 사용자가 본 히스토리를 출력합니다.")
#         pass



