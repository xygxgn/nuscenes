import json
from http.client import responses
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import numpy as np

import urllib3

from exif import EXIF
from project import BoundaryBox
from __init__ import logger

from geopy.geocoders import Nominatim

geolocator = Nominatim(user_agent="autowise")


OSM_URL = "https://api.openstreetmap.org/api/0.6/map.json"


def get_latlon_from_address(
    prior_address: Optional[str],
) -> Optional[np.ndarray]:
    """
    Args:
        prior_address: the prior address
    Returns:
        lat, lon and alt (read if available)
    """
    location = geolocator.geocode(prior_address)
    if location is not None:
        logger.info("Using prior address '%s'", location.address)
        return np.array((location.latitude, location.longitude))
    logger.info("Could not find any location for address '%s.'", prior_address)
    return None


def get_latlon_from_exif(
    exif: Optional[EXIF],
) -> Optional[np.ndarray]:
    """
    Args:
        exif: the image EXIF metadata
    Returns:
        lat, lon and alt (read if available)
    """
    if not isinstance(exif, EXIF):
        raise TypeError("exif must be EXIF or None")
    geo = exif.extract_geo()
    if geo:
        alt = geo.get("altitude", 0)  # read if available
        logger.info("Using prior location from EXIF.")
        return np.array((geo["latitude"], geo["longitude"], alt))
    return None


def get_latlon(
    exif: Optional[EXIF] = None,
    prior_latlon: Optional[Tuple[float, float]] = None,
    prior_address: Optional[str] = None,
) -> np.ndarray:
    """
    Args:
        exif: the image EXIF metadata
        prior_latlon: the prior latitude and longitude
        prior_address: the prior address
    Returns:
        lat, lon and alt (read if available)
    """
    if exif is None and prior_latlon is None and prior_address is None:
        raise ValueError(
            "No location prior or image EXIF metadata given."
        )
    if prior_latlon is not None:
        logger.info("Using prior latlon %s.", prior_latlon)
        return prior_latlon
    if prior_address is not None:
        return get_latlon_from_address(prior_address=prior_address)
    if exif is not None and isinstance(exif, EXIF):
        return get_latlon_from_exif(exif=exif)
    raise ValueError(
        "No location prior given or found in the image EXIF metadata: "
        "maybe provide the name of a street, building or neighborhood?"
    )


def get_osm(
    boundary_box: BoundaryBox,
    cache_path: Optional[Path] = None,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """
    Args:
        boundary_box: the image EXIF metadata
        cache_path: the prior latitude and longitude
        overwrite: the prior address
    Returns:
        lat, lon and alt (read if available)
    """
    if not overwrite and cache_path is not None and cache_path.is_file():
        return json.loads(cache_path.read_text())

    # (min_latitude, min_longitude, max_latitude, max_longitude)
    (bottom, left), (top, right) = boundary_box.min_, boundary_box.max_
    # (min_longitude, min_latitude, max_longitude, max_latitude)
    query = {"bbox": f"{left},{bottom},{right},{top}"}

    logger.info("Calling the OpenStreetMap API...")
    result = urllib3.request("GET", OSM_URL, fields=query, timeout=10)
    if result.status != 200:
        error = result.info()["error"]
        raise ValueError(f"{result.status} {responses[result.status]}: {error}")

    if cache_path is not None:
        cache_path.write_bytes(result.data)
    return result.json()
