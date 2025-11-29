#!/usr/bin/env python3
"""
Can I join?
"""
import requests


def availableShips(passengerCount):
    """
    returns the list of ships that can hold a given number of passengers
    use the pagination
    If no ship available, return an empty list.
    """
    url = 'https://swapi-api.hbtn.io/api/starships'
    ships = []

    while url:
        response = requests.get(url)
        if response.status_code != 200:
            return []

        json_data = response.json()

        for ship in json_data.get('results', []):
            passengers = ship.get('passengers', '0')

            passengers = passengers.replace(',', '')

            if passengers.isdigit() and int(passengers) >= passengerCount:
                ships.append(ship.get('name'))

        # Check for next page
        url = json_data.get('next')

    return ships
