import pymongo
import json
import os
from pymongo import MongoClient, InsertOne

client = pymongo.MongoClient("mongodb://localhost:27017")
db = client.runs
collection = db.all_runs
requesting = []
directory = 'testruns'
for filename in os.listdir(directory):
    name = os.path.join(directory, filename)
    print(name)
    try:
        with open(name) as f:
            myDict = json.loads(f.read())
            requesting.append(InsertOne(myDict))
    except Exception as e:
        print(f"This file is bad: {name}")

result = collection.bulk_write(requesting)
client.close()