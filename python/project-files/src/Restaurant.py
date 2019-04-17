class Restaurant(object):
    """
    Args:
        id (str): The business id.
        name (str): The name of the business.
        rating (float): Overall rating of the restaurant.
        price (str): Overall price of restaurant (ex: '$$$')
        reviews ([RestaurantReview]): Array of reviews for given restaurant.
    """
    def __init__(self, id, name, rating, price, reviews):
        self.id = id
        self.name = name
        self.rating = rating
        self.price = price
        self.reviews = reviews


class RestaurantReview(object):
    """
    Args:
        rating (float): The given rating by the user review.
        text (str): The text review by the user.
    """
    def __init__(self, rating, text):
        self.rating = rating
        self.text = text
