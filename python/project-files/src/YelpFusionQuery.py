# -*- coding: utf-8 -*-
"""
Yelp Fusion API code sample.
This program demonstrates the capability of the Yelp Fusion API
by using the Search API to query for businesses by a search term and location,
and the Business API to query additional information about the top result
from the search query.
Please refer to http://www.yelp.com/developers/v3/documentation for the API
documentation.
This program requires the Python requests library, which you can install via:
`pip install -r requirements.txt`.
Sample usage of the program:
`python sample.py --term="bars" --location="San Francisco, CA"`
"""
from __future__ import print_function

import Restaurant
import argparse
import json
import pprint
import requests
import sys
import urllib
import time


# This client code can run on Python 2.x or 3.x.  Your imports can be
# simpler if you only need one of those.
try:
    # For Python 3.0 and later
    from urllib.error import HTTPError
    from urllib.parse import quote
    from urllib.parse import urlencode
except ImportError:
    # Fall back to Python 2's urllib2 and urllib
    from urllib3 import HTTPError
    from urllib import quote
    from urllib import urlencode


# Yelp Fusion no longer uses OAuth as of December 7, 2017.
# You no longer need to provide Client ID to fetch Data
# It now uses private keys to authenticate requests (API Key)
# You can find it on
# https://www.yelp.com/developers/v3/manage_app
API_KEY = 'C6pC9w1vsH3y_YzWlB3j6aRAWxqmv4lVE4cS1FU1RSEKq6FnWojZf-s4izZ15vBIKXnLJU3R7BZbD3mu1u6rya1Y2Bx790lT9lCfydBqY8lDTZTzxFFrkY-cpOJtXHYx'


# API constants, you shouldn't have to change these.
API_HOST = 'https://api.yelp.com'
SEARCH_PATH = '/v3/businesses/search'
BUSINESS_PATH = '/v3/businesses/'


# Defaults for our simple example.
DEFAULT_CATEGORY = 'restaurant'
DEFAULT_LOCATION = 'sanfrancisco'
SEARCH_LIMIT = 50


restaurants = []


def request(host, path, api_key, url_params=None):
    """Given your api_key, send a GET request to the API.
    Args:
        host (str): The domain host of the API.
        path (str): The path of the API after the domain.
        api_key (str): Your API Key.
        url_params (dict): An optional set of query parameters in the request.
    Returns:
        dict: The JSON response from the request.
    Raises:
        HTTPError: An error occurs from the HTTP request.
    """
    url_params = url_params or {}
    url = '{0}{1}'.format(host, quote(path.encode('utf8')))
    headers = {
        'Authorization': 'Bearer %s' % api_key,
    }

    print(u'Querying {0} ...'.format(url))

    response = requests.request('GET', url, headers=headers, params=url_params)

    return response.json()


def search(api_key, category, location, limit, offset):
    url_params = {
        'category': category,
        'location': location,
        'limit': limit,
        'offset': offset
    }
    return request(API_HOST, SEARCH_PATH, api_key, url_params=url_params)


def get_reviews(api_key, business_id):
    reviews_path = BUSINESS_PATH + business_id + '/reviews'

    return request(API_HOST, reviews_path, api_key)


def get_business(api_key, business_id):
    business_path = BUSINESS_PATH + business_id

    return request(API_HOST, business_path, api_key)


def main():
    offset = 0
    while offset < 1000:
        rest_json = search(API_KEY, DEFAULT_CATEGORY, DEFAULT_LOCATION, SEARCH_LIMIT, offset)
        #time.sleep(5)
        parse(rest_json)
        offset += 50


def parse(rest_json):
    businesses = rest_json.get('businesses')
    if (businesses):
        for bus in businesses:
            rev_list = []
            reviews = get_reviews(API_KEY, bus.get('id')).get('reviews')
            #time.sleep(3)
            if (reviews):
                for review in reviews:
                    rev = Restaurant.RestaurantReview(
                        review.get('rating'),
                        review.get('text')
                    )
                    rev_list.append(rev)
            else:
                pass
            restaurant = Restaurant.Restaurant(
                bus.get('id'),
                bus.get('name'),
                bus.get('rating'),
                bus.get('price'),
                rev_list
            )
            restaurants.append(restaurant)
            print('working')
    else:
        pass

if __name__ == '__main__':
    main()
