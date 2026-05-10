from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from xml.dom import minidom
from xml.etree import ElementTree as ET

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdUtils


class ConversionError(RuntimeError):
    pass


@dataclass(frozen=True)
class JointSpec:
    path: Sdf.Path
    name: str
    kind: str
    parent_body: Sdf.Path | None
    child_body: Sdf.Path
    axis: tuple[float, float, float]
    limits: tuple[float, float] | None
    driven: bool


def fmt(value: float) -> str:
    if abs(value) < 1e-12:
        value = 0.0
    return f"{value:.9g}"


def fmt_vec(values: Iterable[float]) -> str:
    return " ".join(fmt(float(value)) for value in values)


def safe_name(value: str) -> str:
    name = re.sub(r"[^0-9A-Za-z_]+", "_", value.strip("/").replace("/", "_"))
    name = re.sub(r"_+", "_", name).strip("_")
    return name or "unnamed"


def is_api_applied(prim: Usd.Prim, api_name: str) -> bool:
    return any(schema == api_name or schema.startswith(f"{api_name}:") for schema in prim.GetAppliedSchemas())


def has_rigid_body(prim: Usd.Prim) -> bool:
    return bool(UsdPhysics.RigidBodyAPI(prim))


def has_collision(prim: Usd.Prim) -> bool:
    return bool(UsdPhysics.CollisionAPI(prim))


def first_target(prim: Usd.Prim, relationship_name: str) -> Sdf.Path | None:
    relationship = prim.GetRelationship(relationship_name)
    if not relationship:
        return None
    targets = relationship.GetTargets()
    return targets[0] if targets else None


def attr_value(prim: Usd.Prim, name: str, default=None):
    attr = prim.GetAttribute(name)
    if not attr:
        return default
    value = attr.Get()
    return default if value is None else value


def axis_vector(axis) -> tuple[float, float, float]:
    token = str(axis or "X").upper()
    if token.endswith("Y"):
        return (0.0, 1.0, 0.0)
    if token.endswith("Z"):
        return (0.0, 0.0, 1.0)
    return (1.0, 0.0, 0.0)


def quat_tuple(quat) -> tuple[float, float, float, float]:
    return (
        float(quat.GetReal()),
        float(quat.GetImaginary()[0]),
        float(quat.GetImaginary()[1]),
        float(quat.GetImaginary()[2]),
    )


def transform_between(child_world: Gf.Matrix4d, parent_world: Gf.Matrix4d | None) -> Gf.Matrix4d:
    if parent_world is None:
        return child_world
    return child_world * parent_world.GetInverse()


def unresolved_dependencies(usd_path: Path) -> list[str]:
    layer = Sdf.Layer.FindOrOpen(str(usd_path))
    if layer is None:
        raise ConversionError(f"Could not open USD layer: {usd_path}")
    _layers, _assets, unresolved = UsdUtils.ComputeAllDependencies(layer.identifier)
    return [str(path) for path in unresolved]


def is_required_stage_dependency(path: str) -> bool:
    suffix = Path(path).suffix.lower()
    return suffix in {"", ".usd", ".usda", ".usdc", ".usdz"}


def load_stage(usd_path: Path) -> Usd.Stage:
    missing = unresolved_dependencies(usd_path)
    required_missing = [path for path in missing if is_required_stage_dependency(path)]
    if required_missing:
        details = "\n".join(f"  - {path}" for path in required_missing)
        raise ConversionError(
            f"{usd_path} has unresolved USD asset dependencies:\n{details}\n"
            "This file is only an overlay; convert a flattened USD or place the payloads at those paths."
        )

    stage = Usd.Stage.Open(str(usd_path))
    if stage is None:
        raise ConversionError(f"Could not open USD stage: {usd_path}")
    return stage


def is_joint_prim(prim: Usd.Prim) -> bool:
    type_name = prim.GetTypeName().lower()
    return type_name.startswith("physics") and type_name.endswith("joint")


def joint_kind(prim: Usd.Prim) -> str | None:
    type_name = prim.GetTypeName().lower()
    if "revolute" in type_name:
        return "hinge"
    if "prismatic" in type_name:
        return "slide"
    if "fixed" in type_name:
        return "fixed"
    if "spherical" in type_name:
        return "ball"
    return None


def nearest_body_path(target: Sdf.Path | None, body_paths: set[Sdf.Path]) -> Sdf.Path | None:
    if target is None:
        return None
    current = target
    while current != Sdf.Path.absoluteRootPath:
        if current in body_paths:
            return current
        current = current.GetParentPath()
    return None


def collect_body_paths(stage: Usd.Stage) -> list[Sdf.Path]:
    body_paths = [prim.GetPath() for prim in stage.Traverse() if has_rigid_body(prim)]
    if body_paths:
        return body_paths

    default_prim = stage.GetDefaultPrim()
    if default_prim and default_prim.IsValid():
        robot = default_prim.GetChild("simbot")
        if robot and robot.IsValid():
            return [robot.GetPath()]
        return [default_prim.GetPath()]

    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Xform):
            return [prim.GetPath()]
    raise ConversionError("USD stage does not contain a convertible body prim.")


def collect_joints(stage: Usd.Stage, body_paths: set[Sdf.Path]) -> tuple[list[JointSpec], list[str]]:
    joints: list[JointSpec] = []
    warnings: list[str] = []

    for prim in stage.Traverse():
        if not is_joint_prim(prim):
            continue

        kind = joint_kind(prim)
        if kind is None:
            warnings.append(f"Skipped unsupported joint {prim.GetPath()} of type {prim.GetTypeName()}.")
            continue

        parent_body = nearest_body_path(first_target(prim, "physics:body0"), body_paths)
        child_body = nearest_body_path(first_target(prim, "physics:body1"), body_paths)
        if child_body is None:
            warnings.append(f"Skipped joint {prim.GetPath()} because physics:body1 is not a rigid body.")
            continue
        if child_body == parent_body:
            warnings.append(f"Skipped joint {prim.GetPath()} because body0 and body1 resolve to the same body.")
            continue

        lower = attr_value(prim, "physics:lowerLimit")
        upper = attr_value(prim, "physics:upperLimit")
        limits = None
        if lower is not None and upper is not None and abs(float(lower)) != float("inf") and abs(float(upper)) != float("inf"):
            limits = (float(lower), float(upper))

        spec = JointSpec(
            path=prim.GetPath(),
            name=safe_name(str(prim.GetPath())),
            kind=kind,
            parent_body=parent_body,
            child_body=child_body,
            axis=axis_vector(attr_value(prim, "physics:axis", "X")),
            limits=limits,
            driven=is_api_applied(prim, "PhysicsDriveAPI"),
        )
        joints.append(spec)

    return joints, warnings


def reversed_axis(axis: tuple[float, float, float]) -> tuple[float, float, float]:
    return (-axis[0], -axis[1], -axis[2])


def reversed_limits(limits: tuple[float, float] | None) -> tuple[float, float] | None:
    if limits is None:
        return None
    lower, upper = limits
    return -upper, -lower


def choose_root_body(body_paths: list[Sdf.Path], joints: list[JointSpec]) -> Sdf.Path:
    for body_path in body_paths:
        if body_path.name == "body":
            return body_path

    degree = {body_path: 0 for body_path in body_paths}
    for joint in joints:
        if joint.parent_body in degree:
            degree[joint.parent_body] += 1
        if joint.child_body in degree:
            degree[joint.child_body] += 1
    return max(body_paths, key=lambda path: (degree.get(path, 0), -len(str(path)), str(path)))


def orient_joint_tree(body_paths: list[Sdf.Path], joints: list[JointSpec]) -> tuple[list[JointSpec], list[str]]:
    if not joints:
        return [], []

    adjacency: dict[Sdf.Path, list[tuple[Sdf.Path, JointSpec, bool]]] = {path: [] for path in body_paths}
    for joint in joints:
        if joint.parent_body is None:
            continue
        adjacency.setdefault(joint.parent_body, []).append((joint.child_body, joint, False))
        adjacency.setdefault(joint.child_body, []).append((joint.parent_body, joint, True))

    root = choose_root_body(body_paths, joints)
    visited = {root}
    queue = [root]
    oriented: list[JointSpec] = []
    warnings: list[str] = []

    while queue:
        current = queue.pop(0)
        for neighbor, joint, reverse in sorted(adjacency.get(current, []), key=lambda item: str(item[0])):
            if neighbor in visited:
                continue
            visited.add(neighbor)
            queue.append(neighbor)
            oriented.append(
                JointSpec(
                    path=joint.path,
                    name=joint.name,
                    kind=joint.kind,
                    parent_body=current,
                    child_body=neighbor,
                    axis=reversed_axis(joint.axis) if reverse else joint.axis,
                    limits=reversed_limits(joint.limits) if reverse else joint.limits,
                    driven=joint.driven,
                )
            )

    oriented_paths = {joint.path for joint in oriented}
    for joint in joints:
        if joint.parent_body is not None and joint.path not in oriented_paths:
            warnings.append(f"Skipped joint {joint.path} because it would create a kinematic loop.")

    return oriented, warnings


def collect_geoms(stage: Usd.Stage, body_paths: set[Sdf.Path]) -> dict[Sdf.Path, list[Usd.Prim]]:
    geoms: dict[Sdf.Path, list[Usd.Prim]] = {path: [] for path in body_paths}
    for prim in Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies()):
        type_name = prim.GetTypeName()
        if type_name not in {"Capsule", "Cube", "Cylinder", "Mesh", "Sphere"} and not has_collision(prim):
            continue
        body_path = nearest_body_path(prim.GetPath(), body_paths)
        if body_path is not None and prim.GetPath() != body_path:
            geoms[body_path].append(prim)
    return geoms


def local_bounds(bbox_cache: UsdGeom.BBoxCache, prim: Usd.Prim) -> tuple[Gf.Vec3d, Gf.Vec3d] | None:
    bbox = bbox_cache.ComputeLocalBound(prim)
    aligned_range = bbox.ComputeAlignedRange()
    if aligned_range.IsEmpty():
        return None
    lower = aligned_range.GetMin()
    upper = aligned_range.GetMax()
    half = (upper - lower) * 0.5
    center = lower + half
    if max(abs(float(half[0])), abs(float(half[1])), abs(float(half[2]))) < 1e-9:
        return None
    return center, half


def add_inertial(body_el: ET.Element, prim: Usd.Prim) -> bool:
    mass = attr_value(prim, "physics:mass")
    diagonal = attr_value(prim, "physics:diagonalInertia")
    if mass is None or diagonal is None:
        return False

    attrs = {
        "mass": fmt(float(mass)),
        "diaginertia": fmt_vec(diagonal),
    }
    center_of_mass = attr_value(prim, "physics:centerOfMass")
    if center_of_mass is not None:
        attrs["pos"] = fmt_vec(center_of_mass)
    principal_axes = attr_value(prim, "physics:principalAxes")
    if principal_axes is not None and abs(float(principal_axes.GetReal())) + sum(abs(float(x)) for x in principal_axes.GetImaginary()) > 1e-12:
        attrs["quat"] = fmt_vec(quat_tuple(principal_axes))
    ET.SubElement(body_el, "inertial", attrs)
    return True


def mass_props_from_prim(
    prim: Usd.Prim,
    body_world: Gf.Matrix4d,
    xform_cache: UsdGeom.XformCache,
) -> tuple[float, Gf.Vec3d, tuple[float, float, float]] | None:
    mass = attr_value(prim, "physics:mass")
    diagonal = attr_value(prim, "physics:diagonalInertia")
    if mass is None or diagonal is None:
        return None

    center_of_mass = attr_value(prim, "physics:centerOfMass", Gf.Vec3d(0, 0, 0))
    prim_world = xform_cache.GetLocalToWorldTransform(prim)
    rel = transform_between(prim_world, body_world)
    local_com = rel.Transform(center_of_mass)
    return float(mass), local_com, tuple(max(float(value), 1e-9) for value in diagonal)


def add_inertial_from_geoms(
    body_el: ET.Element,
    body_world: Gf.Matrix4d,
    geoms: list[Usd.Prim],
    xform_cache: UsdGeom.XformCache,
) -> bool:
    props = [prop for geom in geoms if (prop := mass_props_from_prim(geom, body_world, xform_cache)) is not None]
    if not props:
        return False

    total_mass = sum(prop[0] for prop in props)
    if total_mass <= 0:
        return False

    center = Gf.Vec3d(0, 0, 0)
    diagonal = [0.0, 0.0, 0.0]
    for mass, local_com, local_diagonal in props:
        center += local_com * (mass / total_mass)
        for index, value in enumerate(local_diagonal):
            diagonal[index] += value

    ET.SubElement(
        body_el,
        "inertial",
        {
            "mass": fmt(total_mass),
            "pos": fmt_vec(center),
            "diaginertia": fmt_vec(max(value, 1e-9) for value in diagonal),
        },
    )
    return True


def add_fallback_inertial(body_el: ET.Element) -> None:
    ET.SubElement(
        body_el,
        "inertial",
        {
            "mass": "0.01",
            "diaginertia": "1e-5 1e-5 1e-5",
        },
    )


def add_geom(
    body_el: ET.Element,
    prim: Usd.Prim,
    body_world: Gf.Matrix4d,
    xform_cache: UsdGeom.XformCache,
    bbox_cache: UsdGeom.BBoxCache,
) -> None:
    bounds = local_bounds(bbox_cache, prim)
    if bounds is None:
        return

    local_center, half = bounds
    geom_world = xform_cache.GetLocalToWorldTransform(prim)
    rel = transform_between(geom_world, body_world)
    pos = rel.Transform(local_center)
    quat = rel.RemoveScaleShear().ExtractRotationQuat()

    type_name = prim.GetTypeName()
    attrs = {
        "name": safe_name(str(prim.GetPath())),
        "pos": fmt_vec(pos),
        "quat": fmt_vec(quat_tuple(quat)),
        "rgba": "0.75 0.76 0.78 1",
    }
    if type_name == "Sphere":
        attrs["type"] = "sphere"
        attrs["size"] = fmt(max(float(half[0]), float(half[1]), float(half[2])))
    elif type_name in {"Capsule", "Cylinder"}:
        attrs["type"] = "capsule" if type_name == "Capsule" else "cylinder"
        attrs["size"] = f"{fmt(max(float(half[0]), float(half[1])))} {fmt(float(half[2]))}"
    else:
        attrs["type"] = "box"
        attrs["size"] = fmt_vec(half)

    ET.SubElement(body_el, "geom", attrs)


def make_body_tree(body_paths: list[Sdf.Path], joints: list[JointSpec]) -> dict[Sdf.Path | None, list[Sdf.Path]]:
    parent_by_child = {joint.child_body: joint.parent_body for joint in joints}
    children: dict[Sdf.Path | None, list[Sdf.Path]] = {None: []}
    for body_path in body_paths:
        parent = parent_by_child.get(body_path)
        children.setdefault(parent, []).append(body_path)
        children.setdefault(body_path, [])
    return children


def write_body(
    parent_el: ET.Element,
    stage: Usd.Stage,
    body_path: Sdf.Path,
    parent_body_path: Sdf.Path | None,
    joint_by_child: dict[Sdf.Path, JointSpec],
    children_by_parent: dict[Sdf.Path | None, list[Sdf.Path]],
    geoms_by_body: dict[Sdf.Path, list[Usd.Prim]],
    xform_cache: UsdGeom.XformCache,
    bbox_cache: UsdGeom.BBoxCache,
) -> None:
    prim = stage.GetPrimAtPath(body_path)
    body_world = xform_cache.GetLocalToWorldTransform(prim)
    parent_world = xform_cache.GetLocalToWorldTransform(stage.GetPrimAtPath(parent_body_path)) if parent_body_path else None
    local = transform_between(body_world, parent_world)

    body_el = ET.SubElement(
        parent_el,
        "body",
        {
            "name": safe_name(str(body_path)),
            "pos": fmt_vec(local.ExtractTranslation()),
            "quat": fmt_vec(quat_tuple(local.RemoveScaleShear().ExtractRotationQuat())),
        },
    )
    geoms = geoms_by_body.get(body_path, [])
    if not add_inertial(body_el, prim):
        if not add_inertial_from_geoms(body_el, body_world, geoms, xform_cache):
            add_fallback_inertial(body_el)

    joint = joint_by_child.get(body_path)
    if joint is None and parent_body_path is None:
        ET.SubElement(body_el, "freejoint", {"name": f"{safe_name(str(body_path))}_free"})
    elif joint is not None and joint.kind != "fixed":
        attrs = {
            "name": joint.name,
            "type": joint.kind,
            "axis": fmt_vec(joint.axis),
        }
        if joint.limits is not None:
            attrs["range"] = fmt_vec(joint.limits)
            attrs["limited"] = "true"
        ET.SubElement(body_el, "joint", attrs)

    for geom_prim in geoms:
        add_geom(body_el, geom_prim, body_world, xform_cache, bbox_cache)

    for child_path in sorted(children_by_parent.get(body_path, []), key=str):
        write_body(
            body_el,
            stage,
            child_path,
            body_path,
            joint_by_child,
            children_by_parent,
            geoms_by_body,
            xform_cache,
            bbox_cache,
        )


def build_mjcf(stage: Usd.Stage, model_name: str) -> tuple[ET.Element, list[str]]:
    body_paths = collect_body_paths(stage)
    body_path_set = set(body_paths)
    joints, warnings = collect_joints(stage, body_path_set)
    joints, orientation_warnings = orient_joint_tree(body_paths, joints)
    warnings.extend(orientation_warnings)
    geoms_by_body = collect_geoms(stage, body_path_set)
    children_by_parent = make_body_tree(body_paths, joints)
    joint_by_child = {joint.child_body: joint for joint in joints}

    root = ET.Element("mujoco", {"model": safe_name(model_name)})
    ET.SubElement(root, "compiler", {"angle": "degree"})
    ET.SubElement(root, "option", {"timestep": "0.001", "gravity": "0 0 -9.81"})

    default = ET.SubElement(root, "default")
    ET.SubElement(default, "geom", {"friction": "1 0.005 0.0001", "density": "500"})
    ET.SubElement(default, "joint", {"damping": "0.05", "armature": "0.001"})

    asset = ET.SubElement(root, "asset")
    ET.SubElement(
        asset,
        "texture",
        {
            "name": "grid",
            "type": "2d",
            "builtin": "checker",
            "rgb1": "0.1 0.2 0.3",
            "rgb2": "0.2 0.3 0.4",
            "width": "300",
            "height": "300",
            "mark": "edge",
            "markrgb": "0.2 0.3 0.4",
        },
    )
    ET.SubElement(
        asset,
        "material",
        {
            "name": "grid",
            "texture": "grid",
            "texrepeat": "2 2",
            "texuniform": "true",
            "reflectance": "0.2",
        },
    )

    worldbody = ET.SubElement(root, "worldbody")
    ET.SubElement(worldbody, "light", {"name": "key", "pos": "1 -3 4", "dir": "-1 3 -4"})
    ET.SubElement(worldbody, "geom", {"name": "floor", "type": "plane", "size": "3 3 0.05", "material": "grid"})

    xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    bbox_cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
        useExtentsHint=True,
    )

    for body_path in sorted(children_by_parent.get(None, []), key=str):
        write_body(
            worldbody,
            stage,
            body_path,
            None,
            joint_by_child,
            children_by_parent,
            geoms_by_body,
            xform_cache,
            bbox_cache,
        )

    actuated = [joint for joint in joints if joint.kind in {"hinge", "slide"} and joint.driven]
    if actuated:
        actuator = ET.SubElement(root, "actuator")
        for joint in actuated:
            ET.SubElement(actuator, "motor", {"name": f"{joint.name}_motor", "joint": joint.name, "gear": "1"})

    return root, warnings


def pretty_xml(root: ET.Element) -> str:
    rough = ET.tostring(root, encoding="utf-8")
    pretty = minidom.parseString(rough).toprettyxml(indent="  ")
    return "\n".join(line for line in pretty.splitlines() if line.strip())


def convert(usd_path: Path, output_path: Path, validate: bool = True) -> list[str]:
    stage = load_stage(usd_path)
    root, warnings = build_mjcf(stage, usd_path.stem)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(pretty_xml(root) + "\n", encoding="utf-8")

    if validate:
        import mujoco

        mujoco.MjModel.from_xml_path(str(output_path))
    return warnings


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert a composed USD physics stage to a MuJoCo MJCF XML.")
    parser.add_argument("usd", type=Path, help="Input USD/USDZ/USDA/USDC file.")
    parser.add_argument("output", type=Path, nargs="?", default=Path("simbot.xml"), help="Output MJCF XML path.")
    parser.add_argument("--no-validate", action="store_true", help="Write XML without compiling it with MuJoCo.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        warnings = convert(args.usd, args.output, validate=not args.no_validate)
    except ConversionError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(f"wrote {args.output}")
    for warning in warnings:
        print(f"warning: {warning}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
