#!/usr/bin/env python3
"""
Defines methods to ping the Star Wars API and return the list of ships
that can hold a given number of passengers
"""


def availableShips(passengerCount):
    """
    Uses the Star Wars API to return the list of ships that can hold
        passengerCount number of passengers

    parameters:
        passengerCount [int]:
            the number of passenger the ship must be able to carry

    returns:
        [list]: all ships that can hold that many passengers
    """
    if type(passengerCount) is not int:
        raise TypeError(
            "passengerCount must be a positive number of passengers")
    if passengerCount < 0:
        raise ValueError(
            "passengerCount must be a positive number of passengers")
    shipsList = []
    return shipsList
