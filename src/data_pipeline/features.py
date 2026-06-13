import xarray as xr
import numpy as np

class SpectralFeatureExtractor:
    """
    Extracts multi-spectral indices for Optical data (Sentinel-2).
    """
    @staticmethod
    def calculate_ndvi(nir: xr.DataArray, red: xr.DataArray) -> xr.DataArray:
        """Normalized Difference Vegetation Index"""
        return (nir - red) / (nir + red + 1e-8)

    @staticmethod
    def calculate_ndwi(green: xr.DataArray, nir: xr.DataArray) -> xr.DataArray:
        """Normalized Difference Water Index"""
        return (green - nir) / (green + nir + 1e-8)
        
    @staticmethod
    def calculate_mndwi(green: xr.DataArray, swir1: xr.DataArray) -> xr.DataArray:
        """Modified Normalized Difference Water Index"""
        return (green - swir1) / (green + swir1 + 1e-8)
        
    @staticmethod
    def calculate_savi(nir: xr.DataArray, red: xr.DataArray, L: float = 0.5) -> xr.DataArray:
        """Soil Adjusted Vegetation Index"""
        return ((nir - red) / (nir + red + L + 1e-8)) * (1.0 + L)

    @staticmethod
    def calculate_ndbi(swir1: xr.DataArray, nir: xr.DataArray) -> xr.DataArray:
        """Normalized Difference Built-up Index"""
        return (swir1 - nir) / (swir1 + nir + 1e-8)

class SARFeatureExtractor:
    """
    Extracts features and textures for SAR data (Sentinel-1).
    """
    @staticmethod
    def calculate_ratio(vv: xr.DataArray, vh: xr.DataArray) -> xr.DataArray:
        return vv / (vh + 1e-8)
        
    @staticmethod
    def calculate_difference(vv: xr.DataArray, vh: xr.DataArray) -> xr.DataArray:
        return vv - vh

    @staticmethod
    def extract_glcm_features(img: np.ndarray) -> np.ndarray:
        """
        Placeholder for GLCM (Gray-Level Co-occurrence Matrix) texture features
        like Entropy, Contrast, Homogeneity.
        Usually computed using skimage.feature.graycomatrix.
        """
        # In practice, this would be applied via xr.apply_ufunc
        pass

class TerrainFeatureExtractor:
    """
    Extracts terrain features from DEM.
    """
    @staticmethod
    def calculate_slope(dem: xr.DataArray) -> xr.DataArray:
        """
        Placeholder for slope calculation using xarray-spatial.
        """
        # from xrspatial import slope
        # return slope(dem)
        pass

    @staticmethod
    def calculate_aspect(dem: xr.DataArray) -> xr.DataArray:
        """
        Placeholder for aspect calculation using xarray-spatial.
        """
        # from xrspatial import aspect
        # return aspect(dem)
        pass

def generate_all_features(ds: xr.Dataset) -> xr.Dataset:
    """
    Applies all feature extractors to a dataset containing required raw bands.
    """
    features = []
    
    # Check if Sentinel-2 bands are present
    if all(b in ds.variables for b in ['nir', 'red', 'green', 'swir1']):
        features.append(SpectralFeatureExtractor.calculate_ndvi(ds['nir'], ds['red']).rename('ndvi'))
        features.append(SpectralFeatureExtractor.calculate_ndwi(ds['green'], ds['nir']).rename('ndwi'))
        features.append(SpectralFeatureExtractor.calculate_ndbi(ds['swir1'], ds['nir']).rename('ndbi'))
        
    # Check if Sentinel-1 bands are present
    if all(b in ds.variables for b in ['vv', 'vh']):
        features.append(SARFeatureExtractor.calculate_ratio(ds['vv'], ds['vh']).rename('vv_vh_ratio'))
        
    if features:
        feature_ds = xr.merge(features)
        return xr.merge([ds, feature_ds])
    return ds
