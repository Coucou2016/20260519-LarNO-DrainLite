import numpy as np
import os
import torch
import torch.utils.data as data


class Dynamic2DFlood(data.Dataset):
    """Dataset for 2D urban flood events (training or testing)."""

    def __init__(self, data_root, split, location,
                 train_list="train8.txt", test_list="test12.txt",
                 wall_height=50, drainage_features=None):
        """
        Parameters
        ----------
        data_root : str
            Root directory of the benchmark data.
        split : str
            'train' or 'test'.
        location : str
            Location tag (e.g. 'region1_20m').
        train_list : str
            Filename of the event list for training, resolved from config/.
        test_list : str
            Filename of the event list for testing, resolved from config/.
        wall_height : float
            Height (metres) used for DEM border walls and building masking.
        drainage_features : list[str] or comma-separated str, optional
            Static pipe-network feature rasters to load from geodata and pass
            to the model as ``drainage_features``. Each raster must have the
            same H x W shape as dem.npy.
        """
        super().__init__()

        self.geo_root = os.path.join(data_root, "geodata")
        self.flood_root = os.path.join(data_root, "flood")
        self.locations = [location]
        self.locations_dir = [os.path.join(self.flood_root, loc) for loc in self.locations]
        self.wall_height = wall_height
        self.drainage_features = _parse_feature_list(drainage_features)

        # Event lists live in config/ relative to the working directory
        config_dir = os.path.join(os.getcwd(), "configs")
        list_file = train_list if "train" in split else test_list
        with open(os.path.join(config_dir, list_file), 'r') as f:
            self.event_names = [line.strip() for line in f if line.strip()]

        self.num_samples = len(self.event_names) * len(self.locations)
        print(
            f"Loaded Dynamic2DFlood {split} with {self.num_samples} samples "
            f"(location:{location}, locations:{len(self.locations)}, events:{len(self.event_names)}, "
            f"drainage_features:{len(self.drainage_features)})"
        )

    def _load_event(self, index):
        event_id = index // len(self.locations)
        loc_id = index % len(self.locations)
        event_dir = os.path.join(
            self.flood_root, self.locations[loc_id], self.event_names[event_id])
        event_data = self._load_event_data(event_dir, self.geo_root, self.locations[loc_id])
        return event_data, event_dir

    def _load_event_data(self, event_dir, geo_root, location):
        event_data = {}
        for attr_file in os.listdir(event_dir):
            if attr_file.endswith(".npy"):
                if "v.npy" in attr_file or "u.npy" in attr_file:
                    continue
                attr_name = os.path.splitext(attr_file)[0]
                event_data[attr_name] = np.load(
                    os.path.join(event_dir, attr_file), allow_pickle=True)

        geo_dir = os.path.join(geo_root, location)
        for attr_file in os.listdir(geo_dir):
            attr_path = os.path.join(geo_dir, attr_file)
            if not os.path.isdir(attr_path) and attr_file.endswith(".npy"):
                attr_name = os.path.splitext(attr_file)[0]
                event_data[attr_name] = np.load(attr_path, allow_pickle=True)

        return event_data

    def _prepare_input(self, event_data):
        """Build input tensors for one event.

        Returns a dict with keys: absolute_DEM, max_DEM, min_DEM,
        rainfall, cumsum_rainfall, and optionally drainage_features.
        """
        w = self.wall_height

        # DEM: replace NaN with wall_height, add border walls, clamp buildings
        dem = torch.from_numpy(event_data["dem"])
        dem = torch.nan_to_num(dem, nan=float(w))
        dem = dem.unsqueeze(0).unsqueeze(3)  # (1, H, W, 1)
        dem = _add_border_walls(dem, wall_height=w)

        # Rainfall: (H, W, T) -> (1, H, W, T)
        rainfall = torch.from_numpy(event_data["rainfall"])
        rainfall = rainfall.permute(1, 2, 0).unsqueeze(0)  # (1, H, W, T)
        cumsum_rainfall = torch.cumsum(rainfall, dim=3)

        # Zero rainfall at building pixels (DEM == wall_height)
        building_mask = (dem == w).expand_as(rainfall)
        rainfall[building_mask] = 0
        cumsum_rainfall[building_mask] = 0

        inputs = {
            "absolute_DEM": dem,
            "max_DEM": dem.max(),
            "min_DEM": dem.min(),
            "rainfall": rainfall,
            "cumsum_rainfall": cumsum_rainfall,
        }

        if self.drainage_features:
            feature_tensors = []
            dem_shape = tuple(event_data["dem"].shape)
            for name in self.drainage_features:
                if name not in event_data:
                    raise KeyError(
                        f"Drainage feature {name!r} not found in geodata. "
                        f"Available keys: {sorted(event_data.keys())}"
                    )
                feature = np.asarray(event_data[name], dtype=np.float32)
                if tuple(feature.shape) != dem_shape:
                    raise ValueError(
                        f"Drainage feature {name!r} shape {feature.shape} "
                        f"does not match dem shape {dem_shape}"
                    )
                feature = torch.from_numpy(np.nan_to_num(feature, nan=0.0, posinf=0.0, neginf=0.0))
                feature_tensors.append(feature.unsqueeze(0).unsqueeze(3))
            inputs["drainage_features"] = torch.cat(feature_tensors, dim=0)

        return inputs

    def _prepare_target(self, event_data, duration=360):
        """Build target water-depth tensor for one event.

        Returns shape [1, H, W, T] in mm.
        """
        h = torch.from_numpy(event_data["h"])  # (T, H, W)
        h = h[:duration]
        h = h.permute(1, 2, 0).unsqueeze(0) * 1000.0  # (1, H, W, T) in mm
        return h

    def __getitem__(self, index):
        event_data, event_dir = self._load_event(index)
        input_vars = self._prepare_input(event_data)
        target_vars = self._prepare_target(event_data)
        event_name = os.path.basename(event_dir)
        return input_vars, target_vars, event_name

    def __len__(self):
        return self.num_samples


# ---------------------------------------------------------------------------
# Standalone utility functions
# ---------------------------------------------------------------------------

def _add_border_walls(dem, wall_height=50):
    """Set the outer border of a DEM tensor to wall_height and clamp interior.

    Parameters
    ----------
    dem : torch.Tensor
        Shape (1, H, W, 1).
    wall_height : float
        Height of the border walls.
    """
    wall = wall_height * torch.ones_like(dem)
    wall[:, 1:-1, 1:-1] = dem[:, 1:-1, 1:-1]
    wall[wall > wall_height] = wall_height
    return wall


def MinMaxScaler(data, max_val, min_val):
    """Min-max normalise *data* to [0, 1]."""
    return (data - min_val) / (max_val - min_val)


def r_MinMaxScaler(data, max_val, min_val):
    """Inverse of :func:`MinMaxScaler`."""
    return data * (max_val - min_val) + min_val


def _parse_feature_list(value):
    if value is None:
        return []
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return [str(item).strip() for item in value if str(item).strip()]
