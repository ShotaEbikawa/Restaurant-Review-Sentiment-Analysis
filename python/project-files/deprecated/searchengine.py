# -*- coding: utf-8 -*-
"""
searchengine.py is py file that contains createSearchableData() function.
createSearchableData() is a function that enables the program to store the given
dataset (which will restaurant-review.json or restaurant-review.csv in our case) to 
external directories.
I made a directory called index_file, where it stores all datas
for restaurant-review.csv.
I used restaurant-review.csv for whoosh because I wasn't able to get the JSON file working
with the library. 
HOWEVER, I got the JSON file working using elasticsearch dsl library in search_elastic.py

GLOSSARY:
    index = pretty much means the directory or database that stores the given file's data
    document = file that is going to be stored in the index
    schema = I'm still not really sure what schema is, but I think it is an object that defines which field
             of the document to be stored in the index(database or folder)
"""


"""
ACTUALLY, OMIT SEARCHENGINE.PY AND WOOSHENGINE.PY BECAUSE
THERE'S FEW ERROR THAT I MADE.
HOWEVER, THESE ERRORS ARE RESOLVED IN SEARCH_ELASTIC.PY
"""

import os
from whoosh.index import create_in
from whoosh.fields import Schema, TEXT, ID
from whoosh.qparser import QueryParser
import whoosh.index as index
import pandas as pd
import codecs
import json
import sys


# data stores the restaurant-review.csv's content
data = pd.read_csv(r"C:\Users\shota\Desktop\Yelp-Sentiment-Analysis\python\project-files\src\restaurant-review.csv")
 

def createSearchableData(root):   

    schema = Schema(title=TEXT(stored=True),path=ID(stored=True),\
              content=TEXT(stored=True),textdata=TEXT(stored=True))
    if not os.path.exists("index_file"):
        os.mkdir("index_file")
    # Creating a index writer to add document as per schema
    ix = create_in("index_file",schema)
    writer = ix.writer()
    id_num = 0
    columns = ['id', 'name', 'rating', 'price', 'reviews__rating', 'reviews__text']
    with codecs.open(r"C:\Users\shota\Desktop\Yelp-Sentiment-Analysis\python\project-files\src\restaurant-review.csv", "r","utf-8") as f:
        for line in f:
            _id, name, rating, price, reviews__rating, reviews__text = line.split(',', 5)
            k = [_id, name, rating, price, reviews__rating, reviews__text]
            idd = 0
            doc = {}
            for c in columns:
                doc[c] = k[idd]
                print(k[idd])
                idd += 1
            doc = json.dumps(doc)
            writer.add_document(title= name, content = u''+doc)
                
#       content = f.readlines()
#       print(content)
#       id_num += 1
#       title = ''+str(id_num)
#       writer.add_document(title=u''+title, content=content)
    writer.commit()
#    filepaths = [os.path.join(root,i) for i in os.listdir(root)]
#    for path in filepaths:
#        fp = open(path,'r')
#        print(path)
#        text = fp.read()
#        writer.add_document(title=path.split("\\")[1], path=path,\
#          content=text,textdata=text)
#        fp.close()
#    writer.commit()

        
    
    
    
root = "src\restaurant-review.json"
createSearchableData(root)



