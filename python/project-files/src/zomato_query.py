"""
Created on Sun Mar 31 10:54:24 2019

@author: anastasiosgrigoriou
"""
from __future__ import print_function

import restaurant as res
import argparse
import json
import pprint
import requests
import sys
import urllib
import time


API_KEYS = ['bb29578461d9b138f0ff1694e5c254e3', 'd5ee4dd0e5aa0f7d1f720d5b17876f11']

# [San Francisco, South San Francisco, Oakland, San Jose, Daly City]
city_ids = [306, 10854, 10773, 10883, 10841]
orders = ['asc', 'desc']

restaurants = []


class Api(object):
    def __init__(self, user_key, host="https://developers.zomato.com/api/v2.1",
                 content_type='application/json'):
        self.host = host
        self.user_key = user_key
        self.headers = {
            "User-agent": "curl/7.43.0",
            'Accept': content_type,
            'X-Zomato-API-Key': self.user_key
        }

    def get(self, endpoint, params):
        url = self.host + endpoint + "?"
        for (k, v) in params.items():
            url = url + "{}={}&".format(k, v)
        url = url.rstrip("&")
        response = requests.get(url, headers=self.headers)
        return response.json()


class ZomatoQuery(object):
    def __init__(self, user_key):
        self.api = Api(user_key)

    def get_city_details(self, **kwargs):
        """
        :param q: query by city name
        :param lat: latitude
        :param lon: longitude
        :param city_ids: comma separated city_id values
        :param count: number of max results to display
        """
        params = {}
        available_keys = ["q", "lat", "lon", "city_ids", "count"]
        for key in available_keys:
            if key in kwargs:
                params[key] = kwargs[key]
        cities = self.api.get("/cities", params)
        return cities

    def get_cuisines(self, city_id, **kwargs):
        """
        :param city_id: id of the city for which collections are needed
        :param lat: latitude
        :param lon: longitude
        """
        params = {"city_id": city_id}
        optional_params = ["lat", "lon"]

        for key in optional_params:
            if key in kwargs:
                params[key] = kwargs[key]
        cuisines = self.api.get("/cuisines", params)
        return cuisines

    def get_by_geocode(self, lat, lon):
        """
        :param lat: latitude
        :param lon: longitude
        """
        params = {"lat": lat, "lon": lon}
        response = self.api.get("/geocode", params)
        return response

    def get_location_details(self, entity_id, entity_type):
        """
        :param entity_id: location id obtained from locations api
        :param entity_type: location type obtained from locations api
        :return:
        """
        params = {"entity_id": entity_id, "entity_type": entity_type}
        location_details = self.api.get("/location_details", params)
        return location_details

    def get_locations(self, query, **kwargs):
        """
        :param query: suggestion for location name
        :param lat: latitude
        :param lon: longitude
        :param count: number of max results to display
        :return: json response
        """
        params = {"query": query}
        optional_params = ["lat", "lon", "count"]

        for key in optional_params:
            if key in kwargs:
                params[key] = kwargs[key]
        locations = self.api.get("/locations", params)
        return locations

    def get_restaurant_details(self, restaurant_id):
        """
        :param restaurant_id: id of restaurant whose details are requested
        :return: json response
        """
        params = {"res_id": restaurant_id}
        restaurant_details = self.api.get("/restaurant", params)
        return restaurant_details

    def get_restaurant_reviews(self, restaurant_id, **kwargs):
        """
        :param restaurant_id: id of restaurant whose details are requested
        :param start: fetch results after this offset
        :param count: max number of results to retrieve
        :return: json response
        """
        params = {"res_id": restaurant_id}
        optional_params = ["start", "count"]

        for key in optional_params:
            if key in kwargs:
                params[key] = kwargs[key]
        reviews = self.api.get("/reviews", params)
        return reviews

    def search(self, **kwargs):
        """
        :param entity_id: location id
        :param entity_type: location type (city, subzone, zone, lanmark, metro , group)
        :param q: search keyword
        :param start: fetch results after offset
        :param count: max number of results to display
        :param lat: latitude
        :param lon: longitude
        :param radius: radius around (lat,lon); to define search area, defined in meters(M)
        :param cuisines: list of cuisine id's separated by comma
        :param establishment_type: estblishment id obtained from establishments call
        :param collection_id: collection id obtained from collections call
        :param category: category ids obtained from categories call
        :param sort: sort restaurants by (cost, rating, real_distance)
        :param order: used with 'sort' parameter to define ascending / descending
        :return: json response
        """
        params = {}
        available_params = [
            "entity_id", "entity_type", "q", "start",
            "count", "lat", "lon", "radius", "cuisines",
            "establishment_type", "collection_id",
            "category", "sort", "order"]

        for key in available_params:
            if key in kwargs:
                params[key] = kwargs[key]
        results = self.api.get("/search", params)
        return results


def main():
    for order in orders:
        if order == 'asc':
            api_key = API_KEYS[0]
        else:
            api_key = API_KEYS[1]
        z = ZomatoQuery(api_key)

        for city_id in city_ids:
            offset = 0
            while offset < 100:
                zomato_json = z.search(
                    entity_id=city_id,
                    entity_type='city',
                    start=offset,
                    count=20,
                    sort='rating',
                    order=order
                )
                z_parse(z, zomato_json)
                offset += 20

    print_json(restaurants)


def z_parse(z, rest_json):
    z_restaurants = rest_json.get('restaurants')
    if z_restaurants:
        for bus in z_restaurants:
            rev_list = []
            res = bus.get('restaurant')
            res_id = res.get('R').get('res_id')
            reviews = z.get_restaurant_reviews(res_id).get('user_reviews')
            if reviews:
                for review in reviews:
                    rev = res.RestaurantReview(
                        review.get('review').get('rating'),
                        review.get('review').get('review_text')
                    )
                    rev_list.append(rev)
            else:
                pass

            restaurant = res.Restaurant(
                res.get('id'),
                res.get('name'),
                float(res.get('user_rating').get('aggregate_rating')),
                str(res.get('price_range')),
                rev_list
            )
            restaurants.append(restaurant)
    else:
        pass


def serialize_reviews(obj):
    if isinstance(obj, res.RestaurantReview):
        serial = obj.__dict__
        return serial
    else:
        raise TypeError("Type not serializable")


def print_json(rests):
    with open("../datasets/restaurant-review-zomato-2.json", "w+", encoding="utf-8") as fp:
        for restaurant in rests:
            json.dump(restaurant.__dict__, fp, default=serialize_reviews)
            fp.write(",")


if __name__ == '__main__':
    main()
