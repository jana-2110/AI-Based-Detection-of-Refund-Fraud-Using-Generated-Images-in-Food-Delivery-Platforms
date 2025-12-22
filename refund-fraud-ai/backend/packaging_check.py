def packaging_match(restaurant, detected):
    packaging = {
        "Dominos": "pizza_box",
        "KFC": "bucket",
        "McDonalds": "paper_bag"
    }
    return packaging.get(restaurant) == detected
