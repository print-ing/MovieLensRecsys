import unittest

import pandas as pd
import numpy as np
from abc import abstractclassmethod,ABC
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity

# movie 데이터프레임
movies = pd.read_csv("movies.csv", encoding='cp949')

# rating 데이터프레임
ratings = pd.read_csv("ratings.csv", encoding='cp949')


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


class recsys_series():
    """유저 기반"""
    def CF_Knn(self, userId, movieId, neighbor_size=0):
        if movieId in rating_matrix.columns:
            sim_scores = user_similarity[userId].copy()
            movie_ratings = rating_matrix[movieId].copy()
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
                        mean_rating = 3.0
                else:
                    mean_rating = 3.0
        else:
            mean_rating = 3.0

        return mean_rating

    """아이템 기반"""
    def CF_item(self, user_id, movie_id):
        if movie_id in item_similarity.columns:
            sim_scores = item_similarity[movie_id]
            user_rating = rating_matrix_t[user_id]
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


class list_making_series(recsys_series):
    def recom_movie_1(self, userId, n_items, neighbor_size=30):
        user_movie = rating_matrix.loc[userId].copy()

        recsys_list = []
        for movie in rating_matrix.columns:
            if pd.notnull(user_movie.loc[movie]):
                user_movie.loc[movie] = 0
            else:
                user_movie.loc[movie] = recsys_series.CF_Knn(self, userId, movie, neighbor_size)

            movie_sort = user_movie.sort_values(ascending=False)[:n_items]
            for i in range(n_items):
                index = movie_sort.index[i]
                movie = movies[movies['movieId'] == index]['title']
                recsys_list.append(index)

            break

        return recsys_list

    def recom_movie_2(self, userId, n_items):
        user_movie = rating_matrix_t.loc[userId].copy()

        recsys_list = []
        for movie in rating_matrix_t.columns:
            if pd.notnull(user_movie.loc[movie]):
                user_movie.loc[movie] = 0
            else:
                user_movie.loc[movie] = recsys_series.CF_item(self, userId, movie)

            movie_sort = user_movie.sort_values(ascending=False)[:n_items]
            for i in range(n_items):
                index = movie_sort.index[i]
                movie = movies[movies['movieId'] == index]['title']
                recsys_list.append(index)

            break

        return recsys_list


class make_final_list(list_making_series):
    def __init__(self, option):
        self._option = option

    def select_recsys(self):
        if self._option == 1:
            rec_list1 = list_making_series.recom_movie_1(self, userId=1, n_items=20, neighbor_size=30)
            print("비슷한 사용자가 본 영화 추천목록")
            print(rec_list1)

            rec_list2 = list_making_series.recom_movie_2(self, userId=1, n_items=20)
            print("이미 본 영화와 관련된 영화 추천목록")
            print(rec_list2)

        else:
            pass

# user parameter은 현재 유저
def similar_user_list(user):
    user_similarity_copy = user_similarity.copy()
    np.fill_diagonal(user_similarity_copy.values, 0)

    sim_list = list(user_similarity_copy.iloc[user - 1])
    sim_list_sort = sorted(sim_list, reverse=True)

    sim_userid_list = []
    for i in range(5):
        sim_userid_list.append(sim_list.index(sim_list_sort[i]) + 1)

    return sim_userid_list

class test_rec(unittest.TestCase):
    def test_recsys(self):

        print("    <<<영화 추천 방법>>>")
        print("1: 영화 추천")
        print("2: 비슷한 유저의 히스토리 출력")


        while True:
            option = int(input("선택지 고르세요: "))
            if option == 1:
                print("다음과 같은 영화를 출력합니다.")
                recsys1 = make_final_list(option)
                recsys1.select_recsys()
            elif option == 0:
                break
            else:
                print("유사한 사용자가 본 히스토리를 출력합니다.")
                print(similar_user_list(1))
                # 이후 pickme -> movie 컴포넌트에서 출력
                pass

if __name__ == "__main__":
    test = test_rec()
    test.test_recsys()