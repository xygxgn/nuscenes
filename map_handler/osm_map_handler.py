from typing import List
import os
from nuscenes.map_expansion.map_api import NuScenesMap, NuScenesMapExplorer
from shapely import affinity
from shapely.geometry import MultiLineString
import geopandas as gpd
from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt
from const import MAP_ORIGIN
from geo import TopocentricConverter

class OsmMapHandler(object):
    """
    vectorized osm map
    """
    def __init__(self, 
                 sd_map_path: str = './data/sets/osm'):
        '''
        Args:
            sd_map_path: path to OpenStreetMap(OSM) data
        '''
        super().__init__()
        self.sd_map_path = sd_map_path
        self.nusc_maps = {}

    def download(self):
        pass

    def convert(self, 
                maps: List = list(MAP_ORIGIN), 
                options: List[str] = ['living_street', 'road'], 
                save_map: bool = False, 
                output_floder: str = './output/osm'):
        '''
        Args:
            maps: list of sd_maps to be converted
        '''
        converted_sd_maps = dict()
        for map_name in maps:
            # convert
            lat, lon, alt = MAP_ORIGIN[map_name]
            converter = TopocentricConverter(lat, lon, alt)
            sd_map = gpd.read_file(os.path.join(self.sd_map_path, '{}.shp'.format(map_name)))
            sd_map = sd_map[sd_map['type'].isin(options)]
            sd_map_topo_list = [
                [converter.to_topocentric(lon, lat, 0.)[:2] for lat, lon in coords]
                for coords in sd_map.geometry.apply(lambda x: list(x.coords))
            ]
            converted_sd_maps[map_name] = MultiLineString(sd_map_topo_list)
            if save_map:
                fig, ax = plt.subplots(figsize=(10, 10))
                lines = LineCollection([list(line.coords) for line in converted_sd_maps[map_name]],
                                        colors='red',
                                        linewidths=1)
                ax.add_collection(lines)
                ax.axis('equal')
                ax.grid(True)
                output_path = os.path.join(output_floder, '{}.png'.format(map_name))
                plt.savefig(output_path)
                plt.close()
        return converted_sd_maps

if __name__ == '__main__':
    handler = OsmMapHandler(sd_map_path='./data/sets/osm')
    options = [
        'trunk', 'primary', 'secondary', 'tertiary', 'unclassified', 'residential', # road
        'trunk_link', 'primary_link', 'secondary_link', 'tertiary_link'# road link
        'living_street',  'road',  # Special road  'service'
    ]
    converted_sd_maps = handler.convert(maps=list(MAP_ORIGIN), options=options, save_map=False)