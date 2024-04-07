# import pyrebase 
# config= {
#     'apiKey': "AIzaSyBNz0_BSuSQnapQZEwyEu9xDJJ6d02DEz4",
#     'authDomain': "disease-49866.firebaseapp.com",
#     'projectId': "disease-49866",
#     'storageBucket': "disease-49866.appspot.com",
#     'messagingSenderId': "943790514985",
#     'appId': "1:943790514985:web:b87388106c152cc81d8d38",
#     'measurementId': "G-5EX7Y7GBFF",
#     "databaseURL":"https://disease-49866-default-rtdb.firebaseio.com/"
#   }

# firebase =pyrebase.initialize_app(config)
# auth=firebase.auth()

# # email='test@gmail.com'
# # password='123456'
# # user=auth.create_user_with_email_and_password(email,password)
# # print(user)


import sqlite3
conn=sqlite3.connect('database.db')
print("Connected to database successfully")
conn.execute('''CREATE TABLE IF NOT EXISTS users 
                 (name text, sex text, phone text, city text, email text, password text)''')
print("Created table ")

ins='''INSERT INTO users (name, sex, phone, city, email, password)VALUES ('swati', 'f', '12344', 'shi','swati@gmail.com', '463737')'''
conn.execute(ins)
conn.commit
conn.close()
