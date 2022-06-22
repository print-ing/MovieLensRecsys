import unittest

import pandas as pd
import hashlib

class User:

    def __init__(self):
        self.userId = 0
        self.df_user = pd.read_csv("user.csv")
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
        self.df_user.to_csv("user.csv", encoding = "utf-8-sig")

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

    def logout(self):
        pass

class TestUser(unittest.TestCase):
    def test_start1(self):
        user = User()
        user.signup()

    def test_start2(self):
        user = User()
        user.login()

    def test_start3(self):
        user = User()
        user.logout()