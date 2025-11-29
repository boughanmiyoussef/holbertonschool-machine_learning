#!/usr/bin/env python3
"""Fetch upcoming SpaceX launch data and display in required format"""
import requests


def get_upcoming_launch():
    """Get the first upcoming launch and print its details"""
    url = "https://api.spacexdata.com/v4/launches/upcoming"
    launches = requests.get(url).json()

    # Sort upcoming launches by date
    launches.sort(key=lambda x: x['date_unix'])
    first = launches[0]

    rocket_url = f"https://api.spacexdata.com/v4/rockets/{first['rocket']}"
    launchpad_url = (
        f"https://api.spacexdata.com/v4/launchpads/{first['launchpad']}"
    )

    rocket = requests.get(rocket_url).json()
    launchpad = requests.get(launchpad_url).json()

    print(
        f"{first['name']} ({first['date_local']}) {rocket['name']} - "
        f"{launchpad['name']} ({launchpad['locality']})"
    )


if __name__ == "__main__":
    get_upcoming_launch()
