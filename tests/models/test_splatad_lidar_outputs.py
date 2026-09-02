"""Regression tests for SplatAD lidar rendering through the sensor API (issues #75, #79)."""
import torch

from nerfstudio.cameras.lidars import LidarType
from nerfstudio.data.datamanagers.full_images_lidar_datamanager import get_lidar_raster_params
from nerfstudio.models.ad_model import ADModel
from nerfstudio.models.splatad import SplatADModel


def test_splatad_overrides_ray_based_lidar_rendering():
    """The inherited ADModel path builds a RayBundle, which SplatADModel.get_outputs rejects."""
    assert SplatADModel.get_outputs_for_lidar is not ADModel.get_outputs_for_lidar


def test_lidar_raster_params_bracket_the_elevation_channels():
    for lidar_type in LidarType:
        try:
            boundaries, mapping, azimuth_resolution = get_lidar_raster_params(lidar_type)
        except (KeyError, NotImplementedError, ValueError):
            continue  # lidar types without a registered elevation mapping
        assert azimuth_resolution > 0
        assert torch.all(boundaries[1:] > boundaries[:-1]), "tile boundaries must be increasing"
        assert boundaries[0] < mapping.min() and boundaries[-1] > mapping.max()


def test_splatad_advertises_rasterized_lidar_to_the_viewer():
    assert SplatADModel.renders_lidar_by_rasterization is True
    assert not getattr(ADModel, "renders_lidar_by_rasterization", False)


def test_lidars_accept_azimuth_elevation_grids():
    """The viewer builds a Lidars on a regular grid; azimuths/elevations are [*num_lidars, n, 1] fields."""
    from nerfstudio.cameras.lidars import Lidars

    lidar = Lidars(
        lidar_to_worlds=torch.eye(4)[:3][None],
        times=torch.tensor([0.0]),
        azimuths=torch.linspace(0, 6.28, 720)[None, :, None],
        elevations=torch.linspace(-0.3, 0.3, 24)[None, :, None],
    )
    assert lidar.shape == (1,)
    assert lidar.azimuths.shape == (1, 720, 1) and lidar.elevations.shape == (1, 24, 1)
