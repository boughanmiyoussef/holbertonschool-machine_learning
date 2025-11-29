#!/usr/bin/env python3
"""
Displays the number of launches per rocket using the SpaceX API
"""
import requests


def fetch_launches():
    """
    Fetch all launches from SpaceX API
    """
    url = "https://api.spacexdata.com/v4/launches"

    response = requests.get(url)

    response.raise_for_status()

    return response.json()


def fetch_rockets():
    """
    Fetch all rocket names and return a dictionary {rocket_id: name}
    """
    url = "https://api.spacexdata.com/v4/rockets"

    response = requests.get(url)

    response.raise_for_status()

    rockets = response.json()

    return {rocket["id"]: rocket["name"] for rocket in rockets}


def count_launches_by_rockets(launches):
    """
    Count how many times each rocket ID appears in launches
    """
    rockets_count = {}

    for lauch in launches:

        rocket_id = lauch.get("rocket")

        if rocket_id:
            rockets_count[rocket_id] = rockets_count.get(rocket_id, 0) + 1

    return rockets_count


if __name__ == "__main__":
    launches = fetch_launches()
    rocket_names = fetch_rockets()
    rocket_counts = count_launches_by_rockets(launches)

    # Build list of tuples (rocket_name, count)
    rockets_stats = [
        (rocket_names.get(rocket_id, "Unknown"), count)
        for rocket_id, count in rocket_counts.items()
    ]

    # Sort by count descending, then by name ascending
    rockets_stats.sort(key=lambda x: (-x[1], x[0]))

    for name, count in rockets_stats:
        print(f"{name}: {count}")
