"""Convert jamebot's MJCF to USD using IsaacLab's MJCF converter.

The IsaacLab MJCF converter does not auto-enable the underlying
``isaacsim.asset.importer.mjcf`` extension (unlike its URDF sibling), so we
enable it manually before the conversion. Run this once after editing
``jamebot_v1.xml`` and the env loads the cached USD.

Usage::

    python convert_mjcf_to_usd.py
"""

from __future__ import annotations

import os

from isaaclab.app import AppLauncher

simulation_app = AppLauncher(headless=True).app

import omni.kit.app

# enable MJCF importer extension before constructing the converter
_manager = omni.kit.app.get_app().get_extension_manager()
if not _manager.is_extension_enabled("isaacsim.asset.importer.mjcf"):
    _manager.set_extension_enabled_immediate("isaacsim.asset.importer.mjcf", True)

from isaaclab.sim.converters import MjcfConverter, MjcfConverterCfg


def _strip_world_body_articulation_root(usd_path: str) -> None:
    """Remove the spurious ArticulationRootAPI from ``worldBody``.

    The MJCF importer marks both the world parent and the actual robot root
    body as articulation roots; PhysX then refuses to load. We keep only the
    one on the real robot body.
    """
    from pxr import Usd, UsdPhysics

    stage = Usd.Stage.Open(usd_path)
    changed = False
    for prim in stage.Traverse():
        if prim.GetName() == "worldBody" and prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)
            print(f"[fixup] removed ArticulationRootAPI from {prim.GetPath()}")
            changed = True
    if changed:
        stage.GetRootLayer().Save()


def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    mjcf_path = os.path.join(here, "robot_model", "jamebot_v1.xml")
    usd_dir = os.path.join(here, "robot_model", "usd_isaaclab")
    usd_name = "jamebot.usd"

    cfg = MjcfConverterCfg(
        asset_path=mjcf_path,
        usd_dir=usd_dir,
        usd_file_name=usd_name,
        force_usd_conversion=True,
        make_instanceable=True,
        fix_base=False,
        import_sites=True,
        self_collision=False,
    )
    converter = MjcfConverter(cfg)
    print(f"[OK] Generated USD: {converter.usd_path}")
    _strip_world_body_articulation_root(converter.usd_path)


if __name__ == "__main__":
    main()
    simulation_app.close()
