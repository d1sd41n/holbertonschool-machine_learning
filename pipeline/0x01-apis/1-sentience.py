#!/usr/bin/env python3
"""[summary]

Returns:
    [type]: [description]
"""
import requests


def sentientPlanets():
    """[summary]

    Returns:
        [type]: [description]
    """
    checked_planet_urls = {}
    sentient_planets = []
    next_page_url = 'https://swapi-api.hbtn.io/api/species/'
    r = requests.get(next_page_url)
    r_json = r.json()
    while next_page_url:
        r = requests.get(
            next_page_url)
        r_json = r.json()
        next_page_url = r_json['next']
        species = r_json['results']
        for s in species:
            designation = s.get('designation', None)
            classification = s.get('classification', None)
            if designation == 'sentient' or classification == 'sentient':
                planet_url = s['homeworld']
                if (checked_planet_urls.get(
                    planet_url, False)
                        or planet_url is None):
                    continue
                else:
                    planet = requests.get(
                        planet_url).json()['name']
                    sentient_planets.append(planet)

                    checked_planet_urls[planet_url] = True
    sentient_planets.sort()
    return sentient_planets
