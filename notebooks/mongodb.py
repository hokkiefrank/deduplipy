from datetime import datetime

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
        ts = float(filename.replace(".json", ""))
        timeobj = datetime.utcfromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
    except Exception as e:
        print(f"Error: {e}")
        continue
    try:
        with open(name) as f:
            myDict = json.loads(f.read())
            myDict['timestamp'] = timeobj
            requesting.append(InsertOne(myDict))
    except Exception as e:
        print(f"This file is bad: {name}")

result = collection.bulk_write(requesting)
client.close()