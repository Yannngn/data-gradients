from pathlib import Path


class AssetNotFoundException(Exception):
    pass


class Asset:
    def __init__(self, path: str | Path):
        self.path: Path = Path(path)

    def read(self):
        with open(self.path) as f:
            return f.read()


class TextAssets:
    def __init__(self, asset_dir: str | Path):
        self.asset_dir: Path = Path(asset_dir) / "text"

    def __getattr__(self, name):
        asset_path = self.asset_dir / (name + ".txt")

        if not asset_path.exists():
            raise AssetNotFoundException(f"Asset not found: {name}")

        return Asset(asset_path).read()


class HTMLAssets:
    def __init__(self, asset_dir: str | Path):
        self.asset_dir: Path = Path(asset_dir) / "html"

    def __getattr__(self, name: str):
        asset_path = self.asset_dir / (name + ".html")

        if not asset_path.exists():
            raise AssetNotFoundException(f"Asset not found: {name}")

        return Asset(asset_path).read()


class CSSAssets:
    def __init__(self, asset_dir: str | Path):
        self.asset_dir: Path = Path(asset_dir) / "css"

    def __getattr__(self, name: str):
        asset_path = self.asset_dir / (name + ".css")

        if not asset_path.exists():
            raise AssetNotFoundException(f"Asset not found: {name}")

        return asset_path


class ImageAssets:
    def __init__(self, asset_dir: str | Path):
        self.asset_dir: Path = Path(asset_dir) / "images"

    def __getattr__(self, name: str):
        for ext in ["jpg", "jpeg", "png", "gif"]:
            asset_path = self.asset_dir / f"{name}.{ext}"

            if asset_path.exists():
                return asset_path
        raise AssetNotFoundException(f"Asset not found: {name}")


class Assets:
    """
    Assets class to allow quick access to assets.
    usage:
       call assets.text.test to get content of assets/text/test.txt asset
       call assets.htm.test to get content of assets/html/test.html asset
       call assets.css.test to get full path of assets/text/test.css asset
       call assets.image.logo to get full path of assets/images/logo.png asset (supported image formats: jpg, jpeg, png, gif)
    """

    def __init__(self, asset_dir: str | Path):
        self.asset_dir: Path = Path(asset_dir)
        self._text_assets = TextAssets(self.asset_dir)
        self._image_assets = ImageAssets(self.asset_dir)
        self._css_assets = CSSAssets(self.asset_dir)
        self._html_assets = HTMLAssets(self.asset_dir)

    @property
    def text(self):
        return self._text_assets

    @property
    def image(self):
        return self._image_assets

    @property
    def css(self):
        return self._css_assets

    @property
    def html(self):
        return self._html_assets
