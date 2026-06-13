import os
import pystac_client
import planetary_computer
import logging
import rasterio
import dask
from dask import delayed
from typing import List, Dict, Any

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class DataIngestor:
    """
    Ingests Multi-Modal Earth Observation Data via STAC APIs.
    Supports Sentinel-1 (SAR), Sentinel-2 (Optical), Copernicus DEM, and ERA5.
    """
    def __init__(self, catalog_url: str = "https://planetarycomputer.microsoft.com/api/stac/v1"):
        self.catalog = pystac_client.Client.open(
            catalog_url,
            modifier=planetary_computer.sign_inplace
        )
        logger.info(f"Connected to STAC Catalog: {catalog_url}")

    def fetch_items(self, collection: str, bbox: List[float], time_range: str, max_items: int = 100):
        """
        Query STAC catalog for given collection.
        """
        search = self.catalog.search(
            collections=[collection],
            bbox=bbox,
            datetime=time_range,
            max_items=max_items
        )
        items = list(search.items())
        logger.info(f"Found {len(items)} items for {collection}.")
        return items

    @dask.delayed
    def download_asset(self, href: str, out_path: str):
        """
        Dask delayed function to download a single asset.
        """
        if os.path.exists(out_path):
            return out_path
            
        try:
            with rasterio.open(href) as src:
                profile = src.profile
                # Using a generic driver for raster output
                profile.update(driver='GTiff', compress='lzw')
                with rasterio.open(out_path, 'w', **profile) as dst:
                    dst.write(src.read())
            return out_path
        except Exception as e:
            logger.error(f"Failed to download {href}: {e}")
            return None

    def build_download_graph(self, items: list, asset_keys: List[str], out_dir: str):
        """
        Builds a Dask graph to parallelize downloads of specific asset keys.
        """
        os.makedirs(out_dir, exist_ok=True)
        tasks = []
        for item in items:
            for key in asset_keys:
                if key in item.assets:
                    asset = item.assets[key]
                    out_path = os.path.join(out_dir, f"{item.id}_{key}.tif")
                    tasks.append(self.download_asset(asset.href, out_path))
        return tasks

if __name__ == "__main__":
    # Example test
    ingestor = DataIngestor()
    bbox = [-122.2751, 47.5469, -121.9613, 47.7458]
    items = ingestor.fetch_items("sentinel-2-l2a", bbox, "2023-06-01/2023-06-10", max_items=2)
    # Download red and nir bands
    tasks = ingestor.build_download_graph(items, ["B04", "B08"], "data/raw/sentinel-2")
    # Uncomment to execute:
    # dask.compute(*tasks)
