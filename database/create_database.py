import sqlite3
from DataLoader import DataLoader
import json

dataset_path = "C:/Users/Fabian/OneDrive - bwedu/Privat/MotionDetection_Dataset/SensorCollector_Time_Gravity/"

connection = sqlite3.connect("data.db")
cursor = connection.cursor()

cursor.execute("""DROP TABLE data;""")

sql_create = """CREATE TABLE data (id INTEGER PRIMARY KEY, type INTEGER, class INTEGER, data TEXT);"""
cursor.execute(sql_create)

loader = DataLoader(dataset_path, 4, 1, [0.8, 0.15])

for type in range(3):
    for i in range(loader.length(type)):
        x, y = loader.next_batch(type)
        json_data = json.dumps(x[0].tolist())
        sql_insert = "INSERT INTO data (id, type, class, data) VALUES (NULL, ?, ?, ?);"
        cursor.execute(sql_insert, (type, int(y[0]), json_data))

connection.commit()
connection.close()
