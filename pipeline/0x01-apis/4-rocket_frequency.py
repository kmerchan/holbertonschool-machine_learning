#!/usr/bin/env python3
"""
Uses the (unofficial) SpaceX API to print the number of launches per rocket as:
<rocket name>: <number of launches>
ordered by the number of launches in descending order or,
if rockets have the same amount of launches, in alphabetical order
"""


if __name__ == "__main__":
    rocket = None
    launches = None
    print("{}: {}".format(rocket, launches))
