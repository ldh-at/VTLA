from __future__ import annotations

from dataclasses import dataclass

from lerobot.robots import RobotConfig
from lerobot.robots.so_follower.config_so_follower import SOFollowerConfig


@RobotConfig.register_subclass("paxini_so101_follower")
@dataclass
class PaxiniSO101FollowerConfig(RobotConfig, SOFollowerConfig):
    tactile_port: str = "/dev/ttyUSB0"  # 실제 포트로 바꿔야 함
    tactile_baudrate: int = 115200
    tactile_device_id: int = 0x01
    tactile_read_func_code: int = 0x7B
    tactile_read_addr: int = 0x040E
    tactile_read_len: int | None = None
    tactile_timeout_s: float = 0.5
    tactile_taxel_dtype: str = "uint16"
    tactile_taxel_scale: float | None = None
    tactile_map_path: str | None = None
    tactile_image_size: int = 64
    tactile_num_taxels: int = 100
    tactile_mock: bool = True
