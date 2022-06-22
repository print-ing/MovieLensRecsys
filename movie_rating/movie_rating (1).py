import json
import unittest

# Movie -------------------------------
class Movie:
    with open('final_movie_dict.json') as file:
        movie = json.load(file)
    with open('genres_dict.json') as file:
        genre = json.load(file)  # {movieId : [title, year,genres,story,director,[actors]}

    def __init__(self):
        self.temp_data = TempData()

    def print_10_movie(self):
        # movieId 목록을 받아와서 10개씩 출력
        ten_movie_list = self.temp_data.get_n_movies()
        for i in range(10):
            print('[{}] {} / movieId : {}'.format(i + 1, Movie.movie[ten_movie_list[i]][0], ten_movie_list[i]))

        # pickMe 더보기, repet 증가
        # pickMe topMovie 종료
        # break

    def print_movies(self):
        movie_id = input("movieId")
        print("[{}] {}_{} <{}>".format(movie_id, Movie.movie[movie_id][0], Movie.movie[movie_id][1],
                                       Movie.movie[movie_id][2]))

    def all_imformation(self):
        # movie 정보 출력
        movie_id = input("\tmovieId: ")
        print("title:" + Movie.movie[movie_id][0])
        print("year:" + Movie.movie[movie_id][1])
        print("genres:" + Movie.movie[movie_id][2])
        #print("director:" + Movie.movie[movie_id][4])
        #print("actors:" + Movie.movie[movie_id][5])
        #print("story:" + Movie.movie[movie_id][3])
    # pickMe 평점 입력

    def print_genres(self):
        for i, genre_name in enumerate(Movie.genre):
            print('[{}] {}'.format(i, genre_name))
        self.input_genres()

    def input_genres(self):
        index = int(input("genre index"))
        now_genre = list(Movie.genre.keys())[index]
        self.temp_data.set_movie_list(Movie.genre[now_genre])
        self.print_10_movie()

    # movieId list 받아서 abc, 연도순 정렬
    def sort_movie_by_title(self, movie_id_list):
        # {movieId : [title, year,genres,story,derector,[actors]}의 형태일 때
        # 장르 내 영화 딕셔너리
        genre_movie = {}
        for movie_id in movie_id_list:
            genre_movie[movie_id] = Movie.movie[movie_id]
        genre_movie = dict(sorted(genre_movie.items(), key=lambda x: x[1][0]))
        self.temp_data.set_movie_list(list(genre_movie.keys()))

    def sort_movie_by_year(self, movie_id_list):
        # 최신순 정렬
        genre_movie = {}
        for movie_id in movie_id_list:
            genre_movie[movie_id] = Movie.movie[movie_id]
        genre_movie = dict(sorted(genre_movie.items(), key=lambda x: x[1][1], reverse=True))
        self.temp_data.set_movie_list(list(genre_movie.keys()))

# Data-------------------------------
class TempData:
    movie_list = []
    user_id = ''

    # 정렬되거나 선택한 장르의 영화 목록 등 임시 데이터
    def set_movie_list(self, received_movie_list):
        TempData.movie_list = received_movie_list

    def get_n_movies(self, n=10):
        user_id = 0
        n_movies = []
        for i in range(n):
            if len(TempData.movie_list) == 0:
                break
            n_movies.append(TempData.movie_list[0])
            del(TempData.movie_list[0])
        return n_movies

    def set_user_id(self, received_user_id):
        self.user_id = received_user_id


# Rating -------------------------------
class Rating:
    with open("rating_dict.json") as file:
        rating = json.load(file)

    def __init__(self):
        self.movie = Movie()
        self.temp_data = TempData()
        self.ranking = {}

    def init_ranking(self):
        # ranking = {movieId:[rating, view]}
        for i in Rating.rating.values():
            for movie_id in i.keys():
                self.ranking[movie_id] = [0, 0]

    def set_movie_rating(self, movie_rating, user_id, movie_id):
        Rating.rating[user_id][movie_id] = movie_rating

    # 아직 데이터 받아오는 부분을 구현하지 않아서 기본값 설정해뒀음
    def input_rating(self, user_id='1', movie_id='1'):
        movie_rating = float(input("평점 입력"))
        self.set_movie_rating(movie_rating, user_id, movie_id)

    def calculate_ranking(self):
        # rating : {userId: {movieId: rating}}
        # ranking : {movieId: [sum(movie_rating), num_ratings]}
        self.init_ranking()
        removing_key = []
        for i in Rating.rating.values():
            for movie_id, score in i.items():
                self.ranking[movie_id][0] += score
                self.ranking[movie_id][1] += 1

        # 평균 평점 구하기, 조회수 50 아래 랭킹에서 삭제하기
        for movie_id in self.ranking.keys():
            if self.ranking[movie_id][1] < 50:
                removing_key.append(movie_id)
            self.ranking[movie_id][0] /= self.ranking[movie_id][1]

        for movie_id in removing_key:
            del (self.ranking[movie_id])

        # 평점, 조회수로 내림차순 정렬
        self.ranking = dict(sorted(self.ranking.items(), key=lambda x: x[1], reverse=True))

        self.temp_data.set_movie_list(list(self.ranking.keys()))
        self.movie.print_10_movie()

    def print_score(self, movie_id, user_id):
        print(Rating.rating[user_id][movie_id])

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

# PickMe -------------------------------
class PickMe:
    def __init__(self):
        self.rating = Rating()
        self.rating.init_ranking()
        self.movie = Movie()
        self.temp_data = TempData()

    def main_entrance(self):
        while 1:
            option = int(input("\t[2]: top movie\t[3]: genre\t[6]: exit >>"))
            if option == 2:
                self.rating.calculate_ranking()
            elif option == 3:
                self.movie.print_genres()
            elif option == 6:
                break

            self.extensions()

    def extensions(self):
        while 1:
            option = int(input("\t[1]: initial\t[2]: more\t[3]: movie information >>"))
            if option == 1:
                # 처음으로 선택시에 저장된 임시 영화 리스트 초기화
                self.temp_data.set_movie_list([])
                break
            elif option == 2:
                self.movie.print_10_movie()
            elif option == 3:
                self.movie.all_imformation()
                self.movie_rating_option()

    def movie_rating_option(self):
        option = int(input("\t[1]: initail\t[2]: rate movie >>"))
        if option == 2:
            self.rating.input_rating()


    def print_history(self, user_id):
        Rating.rating[user_id]
# -------------------------------

#rating = Rating()
#rating.init_ranking()
#rating.calculate_ranking()
#rating.check_views()
#rating.scatter_views()

class test(unittest.TestCase):
    def test_movie_rating(self):
        movie = Movie()
        movie.sort_movie_by_year(["1", "111", "3", "61", "5"])

if __name__ == '__main__':
    movie = Movie()
    movie.sort_movie_by_year(["1", "111", "3", "61", "5"])