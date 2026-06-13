import xarray as xr
import rioxarray
import zarr
import logging
from typing import List, Tuple
import os

logger = logging.getLogger(__name__)

class Preprocessor:
    """
    Handles reprojection, cloud masking, and conversion to Zarr formats
    for efficient multi-dimensional array access.
    """
    def __init__(self, target_crs: str = "EPSG:32612", target_res: Tuple[float, float] = (10.0, 10.0)):
        self.target_crs = target_crs
        self.target_res = target_res

    def load_raster(self, filepath: str) -> xr.DataArray:
        """
        Loads raster as an Xarray DataArray using rioxarray.
        """
        da = rioxarray.open_rasterio(filepath, chunks={'x': 512, 'y': 512})
        return da

    def reproject_and_align(self, da: xr.DataArray) -> xr.DataArray:
        """
        Reprojects the array to the target CRS and resolution.
        """
        logger.info(f"Reprojecting to {self.target_crs}")
        da_reproj = da.rio.reproject(
            self.target_crs,
            resolution=self.target_res,
            resampling=rioxarray.enums.Resampling.bilinear
        )
        return da_reproj

    def apply_cloud_mask(self, da: xr.DataArray, scl_filepath: str) -> xr.DataArray:
        """
        Applies Sentinel-2 Scene Classification (SCL) cloud mask.
        Classes to mask: 3 (Cloud shadow), 8 (Cloud medium), 9 (Cloud high), 10 (Cirrus)
        """
        scl = rioxarray.open_rasterio(scl_filepath, chunks={'x': 512, 'y': 512})
        scl_reproj = self.reproject_and_align(scl)
        
        # Mask out clouds and shadows
        mask = ~scl_reproj.isin([3, 8, 9, 10])
        # Align mask with data
        mask = mask.reindex_like(da, method='nearest')
        return da.where(mask)

    def save_to_zarr(self, ds: xr.Dataset, zarr_path: str):
        """
        Saves the xarray Dataset to a Zarr store.
        """
        logger.info(f"Writing Zarr store to {zarr_path}")
        ds.to_zarr(zarr_path, consolidated=True, mode='w')

    def create_tiles(self, da: xr.DataArray, tile_size: int = 256) -> List[xr.DataArray]:
        """
        Splits a large raster into 256x256 tiles for deep learning.
        """
        # This function generates slices for the data array based on tile_size
        tiles = []
        # Calculate number of tiles
        y_size = da.sizes['y']
        x_size = da.sizes['x']
        
        for y in range(0, y_size - tile_size + 1, tile_size):
            for x in range(0, x_size - tile_size + 1, tile_size):
                tile = da.isel(y=slice(y, y + tile_size), x=slice(x, x + tile_size))
                tiles.append(tile)
        
        return tiles

if __name__ == "__main__":
    pass
