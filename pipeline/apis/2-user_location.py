#!/usr/bin/env python3
"""
Script to fetch and display GitHub user location
"""
import sys
import requests
from datetime import datetime


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} https://api.github.com/users/<username>")
        sys.exit(1)

    url = sys.argv[1]

    try:
        response = requests.get(url)
    except requests.RequestException:
        print("Network error")

    if response.status_code == 200:
        data = response.json()
        location = data.get("location")
        print(location or "No location")

    elif response.status_code == 404:
        print("Not found")

    elif response.status_code == 403:
        reset = response.headers.get("X-RateLimit-Reset")

        if reset:
            rest_time = int(reset)
            current_time = int(datetime.now().timestamp())
            minutes = ((rest_time - current_time) // 60)

            print(f"Reset in {minutes} min")

        else:
            print("Access forbidden")

    else:
        print(f"Error status: {response.status_code}")
