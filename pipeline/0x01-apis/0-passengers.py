#!/usr/bin/env python3
"""[summary]

Returns:
    [type]: [description]
"""
import requests


def availableShips(passengerCount):
    """[summary]

    Args:
        passengerCount ([type]): [description]

    Returns:
        [type]: [description]
    """
    starships_with_capacity = []
    next_page_url = 'https://swapi-api.hbtn.io/api/starships'

    while next_page_url:
        r = requests.get(next_page_url)
        r_json = r.json()
        next_page_url = r_json['next']
        starships = r_json['results']
        for starship in starships:
            capacity = starship.get('passengers', 0).replace(',', '')
            try:
                capacity = int(capacity)
            except ValueError:
                continue
            if capacity >= passengerCount:
                starships_with_capacity.append(starship['name'])
    return starships_with_capacity
