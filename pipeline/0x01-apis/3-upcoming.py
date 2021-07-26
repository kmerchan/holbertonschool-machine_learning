#!/usr/bin/env python3
"""
Uses the (unofficial) SpaceX API to print the upcoming launch as:
<launch name> (<date>) <rocket name> - <launchpad name> (<launchpad locality>)

The “upcoming launch” is the one which is the soonest from now, in UTC
and if 2 launches have the same date, it's the first one in the API result.
"""


if __name__ == "__main__":
    launchName = None
    date = None
    rocket = None
    launchPad = None
    location = None
    print("{} ({}) {} - {} ({})".format(
        launchName, date, rocket, launchPad, location)
