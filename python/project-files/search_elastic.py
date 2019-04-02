"""
search_elastic.py is a file that enables to store file contents (restaurant-review.json in our case)
to the index (which is like a database that stores file contents).
"""




# -*- coding: utf-8 -*-
from elasticsearch_dsl import connections
from elasticsearch_dsl import (Index, Document, Text, analyzer)
from datetime import datetime
from elasticsearch_dsl import Document, Date, Nested, Boolean, \
    analyzer, InnerDoc, Completion, Keyword, Text
from elasticsearch_dsl import FacetedSearch, TermsFacet, DateHistogramFacet
import pandas as pd
import codecs
import json
import sys

# data contains the content of restaurant-review.json
# this search engine connects to localhost:9200
# I created an index called yelp.
data = pd.read_json(r"C:\Users\shota\Desktop\Yelp-Sentiment-Analysis\python\project-files\src\restaurant-review.json")
connections.create_connection(hosts=['http://localhost:9200'], timeout=60)
yelp = Index('yelp')
yelp.settings(
        # still not too sure what shard is but it 
        # essentially splits the index into given amount of number(s)
        # I have no idea what replica does, but I did it because it was 
        # implemented in the documentation
    number_of_shards=1,
    number_of_replicas=0
)

# this method enables to choose and aliase to our index
yelp.aliases(
    connection_1 = {}
)
# create the index (yelp in our case)
yelp.create()

# html_strip contains all the preprocessing features that
# I added (standard, lowercase, eliminating stopwords, and snowball stemmers)
# I'm not too sure what char_filter["html_strip"] does
html_strip = analyzer('html_strip',
    tokenizer="standard",
    filter=["standard", "lowercase", "stop", "snowball"],
    char_filter=["html_strip"]
)
# close the index (yelp in our case)
yelp.close()




"""
In elastice search, there's two term:
    1) Document = pretty much file or file content that is going to be added to the index
    2) Inner Document = hard to explain the concept, but it's pretty much a document that will be
                        implemented INSIDE the original document.
"""

"""
ReviewList is the innerDoc that is implemented in Post document.
ReviewList is essentially reviews list key of restaurant-review.json.
It stores rating key and text key.
def Index dictates which index it will belong to
def save ovverides the current state of ReviewList object
"""
class ReviewList(InnerDoc):
    rating =Text()
    text = Text()
    class Index:
        name = 'yelp'

    def save(self, ** kwargs):
        self.created_at = datetime.now()
        return super().save(** kwargs)
    
    
    
"""
Post is the Document that essentially acts as json object in restaurant-review.json
idd stores value of the json object's id key
name stores value of the json object's name key
rating stores value of the json object's rating key
review stores a list of ReviewList. (Nested() is a function in elastic search library where it acts like a list)
I am still not sure what category does to the Post Document but it doesn't affect the code itself.
class Index dictates which index the document belongs to
def add_review increments new ReviewList object in the reviews list
def clear_review clears the reviews list
def save overrides the current state of Post
"""
class Post(Document):
    idd = Text()
    name = Text()
    rating = Text()
    price = Text()
    reviews = Nested(ReviewList)
    published = Boolean()
    category = Text(
        analyzer=html_strip,
        fields={'raw': Keyword()}
    )


    class Index:
        name = 'yelp'

    def add_review(self, r, t):
        self.reviews.append(
          ReviewList(rating=r, text=t))
        
    def clear_review(self):
        self.reviews.clear()
    
    def save(self, ** kwargs):
        self.created_at = datetime.now()
        return super().save(** kwargs)


# initialize Post
Post.init()
# open the yelp index
yelp.open()
# reviewliststorage is a list that stores ReviewList
reviewliststorage = []

"""
iterate through data(content of restaurant-review.json)
ids store value of ids key
resname stores value of name key
rating stores value of rating key
resprice stores value of price key
reviewstext stores value of reviews key 
"""
for key, value in data.items():
    if (key == 'id'):
        ids = value 
    elif (key == 'name'):
        resname = value
    elif (key == 'rating'):
        rating = value
    elif (key == 'price'):
        resprice = value
    elif (key == 'reviews'):
        reviewstext = value

#check makes sure if all the key are passed before calling add_review
check = 0

"""
Since I couldn't find json library where it fetches every json object of restaurant-review.json,
each variable (ids, resname, rating, resprice, and reviewstext) contains over 1000 elements.
I called for loop that iterates until it reaches length of ids.
I initialized Post called first and I stored ids, resname, rating and resprice to it.
I called another for loop in order to get every key in the reviewstext object (rating key and text key)
"""
for i in range(len(ids)):                    
    first = Post(idd=str(ids[i]), name=str(resname[i]), rating=str(rating[i]),price=str(resprice[i]))
    first.clear_review()
    for ii in range(len(reviewstext[i])):
        for key, value in reviewstext[i][ii].items():
            #every time if key is either rating or text, check inrements by 1
            if (key == 'rating'):
                rrating = value
                check += 1
            elif (key == 'text'):
                rtext = value
                check += 1
            # if both keys' values are initialized in the given variable (rrating and rtext),
            # it will call add_review() to append new ReviewList to first
            if (check == 2):
                first.add_review(rrating, rtext)
                check = 0
    # overrides the current state of first
    first.save()
# close the yelp index
yelp.close()    


# open the yelp index
yelp.open()
# not too sure what client does but followed the documentation in this part.
s = Search(using=client)
# s stores the query that one makes
s = Search(index="yelp").using(client).query("match", name = "The")
# response stores the query results
response = s.execute()
yelp.close()
