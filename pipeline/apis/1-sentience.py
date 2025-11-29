#!/usr/bin/env python3
"""
Where I am?
"""
import requests


def sentientPlanets():
    """
    Returns the list of names of the home planets of all sentient species.
    Use the pagination
    'sentient' type is either in the 'classification' or 'designation'
    """
    url = 'https://swapi-api.hbtn.io/api/species'
    home_planets = []

    while url:
        response = requests.get(url)

        if response.status_code != 200:
            return []

        json_data = response.json()

        for plt in json_data.get('results', []):
            dess = plt.get('designation', '0')
            clss = plt.get('classification', '0')
            homeWorld_url = plt.get('homeworld')

            if (dess == 'sentient' or clss == 'sentient') and homeWorld_url:
                planet_response = requests.get(homeWorld_url)

                if planet_response.status_code == 200:
                    planet_data = planet_response.json()
                    planet_name = planet_data.get('name')

                    if planet_name:

                        home_planets.append(planet_name)

        url = json_data.get('next')

    return home_planets
