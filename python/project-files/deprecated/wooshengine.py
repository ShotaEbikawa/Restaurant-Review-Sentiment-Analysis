# -*- coding: utf-8 -*-

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

s = Schema(title=TEXT(stored=True),path=ID(stored=True), content=TEXT(stored=True),textdata=TEXT(stored=True))
print(s)
x = index.open_dir("index_file", schema = s)
print(x)
print('Please enter something')
query = input()
query_uni = u''+query
qp = QueryParser("content", schema=s)
q = qp.parse(query_uni)
with x.searcher() as searcher:
    result = searcher.search(q)
    print(result)
    if (result):
        for i in result:
            print(i.highlights("content")+"\n")
