from numpy import ndarray


def convert_initial_to_beacon_coordinates(coordinate: ndarray,
                                          beacon_coordinate: ndarray):
    """
    Converts coordinates from the initial coordinate system into the beacon
    coordinate system
    :param coordinate: Coordinates that should be converted to the beacon
    coordinates system
    :param beacon_coordinate: Coordinates of the origin of the beacon coordinate
    system
    :return: Coordinate in the beacon coordinate system
    """
    return coordinate[0] - beacon_coordinate[0], \
           coordinate[1] - beacon_coordinate[1]

