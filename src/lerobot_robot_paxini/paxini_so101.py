from __future__ import annotations

from functools import cached_property

from lerobot.configs import FeatureType, PolicyFeature
from lerobot.robots.so_follower.so_follower import SOFollower
from lerobot.types import RobotObservation
from lerobot.utils.constants import OBS_TACTILE

from .config_paxini_so101 import PaxiniSO101FollowerConfig
from .paxini_reader import MockPaxiniReader, SerialPaxiniReader
from .tactile_render import TactileHeatmapRenderer, load_taxel_map_xlsx, make_mock_taxel_map

PAXINI_TACTILE_KEY = f"{OBS_TACTILE}.primary"


class PaxiniSO101Follower(SOFollower):
    config_class = PaxiniSO101FollowerConfig
    name = "paxini_so101_follower"

    def __init__(self, config: PaxiniSO101FollowerConfig):
        super().__init__(config)
        self.config = config

        if config.tactile_map_path is None:
            taxel_map = make_mock_taxel_map(config.tactile_num_taxels)
        else:
            taxel_map = load_taxel_map_xlsx(config.tactile_map_path)

        self.tactile_renderer = TactileHeatmapRenderer(
            taxel_map=taxel_map,
            image_size=config.tactile_image_size,
        )
        if config.tactile_mock:
            self.tactile = MockPaxiniReader(num_taxels=taxel_map.num_taxels)
        else:
            self.tactile = SerialPaxiniReader(
                config.tactile_port,
                config.tactile_baudrate,
                num_taxels=taxel_map.num_taxels,
                device_id=config.tactile_device_id,
                read_func_code=config.tactile_read_func_code,
                read_addr=config.tactile_read_addr,
                read_len=config.tactile_read_len,
                timeout_s=config.tactile_timeout_s,
                taxel_dtype=config.tactile_taxel_dtype,
                taxel_scale=config.tactile_taxel_scale,
            )

    @cached_property
    def observation_features(self) -> dict[str, type | tuple | PolicyFeature]:
        return {
            **self._motors_ft,
            **self._cameras_ft,
            PAXINI_TACTILE_KEY: PolicyFeature(
                type=FeatureType.TACTILE,
                shape=(self.config.tactile_image_size, self.config.tactile_image_size),
            ),
            "paxini.fx": float,
            "paxini.fy": float,
            "paxini.fz": float,
            "paxini.tx": float,
            "paxini.ty": float,
            "paxini.tz": float,
        }

    def connect(self, calibrate: bool = True) -> None:
        super().connect(calibrate=calibrate)
        self.tactile.connect()

    def get_observation(self) -> RobotObservation:
        obs = super().get_observation()
        sample = self.tactile.read_latest()
        obs[PAXINI_TACTILE_KEY] = self.tactile_renderer.render(sample.taxels)
        obs["paxini.fx"] = sample.fx
        obs["paxini.fy"] = sample.fy
        obs["paxini.fz"] = sample.fz
        obs["paxini.tx"] = sample.tx
        obs["paxini.ty"] = sample.ty
        obs["paxini.tz"] = sample.tz
        return obs

    def disconnect(self) -> None:
        self.tactile.disconnect()
        super().disconnect()
