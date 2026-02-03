from importlib.resources import files

from data_gradients.assets.assets_container import AssetNotFoundException, Assets

assets = Assets(str(files("data_gradients.assets")))

__all__ = ["assets", "AssetNotFoundException"]
