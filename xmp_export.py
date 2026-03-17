"""
XMP preset export for Adobe Lightroom / Camera Raw.

Generates .xmp preset files that reference a DCP camera profile by name,
allowing Lightroom users to apply channel-swap profiles via the preset panel.
"""

import os
import shutil
import uuid
from pathlib import Path

XMP_TEMPLATE = """\
<x:xmpmeta xmlns:x="adobe:ns:meta/" x:xmptk="DNG Channel Tool">
 <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about=""
    xmlns:crs="http://ns.adobe.com/camera-raw-settings/1.0/"
    crs:PresetType="Normal"
    crs:Cluster=""
    crs:UUID="{preset_uuid}"
    crs:SupportsAmount="False"
    crs:SupportsColor="True"
    crs:SupportsMonochrome="True"
    crs:SupportsHighDynamicRange="True"
    crs:SupportsNormalDynamicRange="True"
    crs:SupportsSceneReferred="True"
    crs:SupportsOutputReferred="True"
    crs:CameraModelRestriction="{camera_model}"
    crs:Copyright="{copyright}"
    crs:Version="15.0"
    crs:ProcessVersion="11.0"
    crs:CameraProfile="{profile_name}"
    crs:Group{{Name}}="{group_name}"
    />
 </rdf:RDF>
</x:xmpmeta>
"""


def write_xmp_preset(
    filepath: str,
    profile_name: str,
    camera_model: str = "",
    group_name: str = "Channel Swap",
    copyright: str = "DNG Channel Tool",
) -> str:
    """Write an XMP preset file that references a DCP camera profile.

    Parameters
    ----------
    filepath : str
        Destination path for the .xmp file.
    profile_name : str
        Name of the DCP camera profile (must match the embedded profile name
        in the corresponding .dcp file).
    camera_model : str, optional
        Restrict the preset to a specific camera model.  Leave empty to allow
        the preset to appear for every camera.
    group_name : str, optional
        Lightroom preset group / folder name.
    copyright : str, optional
        Copyright string embedded in the preset metadata.

    Returns
    -------
    str
        The absolute path of the written file.
    """
    preset_uuid = str(uuid.uuid4()).upper()

    xml = XMP_TEMPLATE.format(
        preset_uuid=preset_uuid,
        camera_model=camera_model,
        copyright=copyright,
        profile_name=profile_name,
        group_name=group_name,
    )

    filepath = os.path.abspath(filepath)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    with open(filepath, "w", encoding="utf-8") as fh:
        fh.write(xml)

    return filepath


def get_lightroom_preset_dir() -> Path:
    """Return the default Adobe Camera Raw preset directory on Windows.

    The path is ``%APPDATA%/Adobe/CameraRaw/Settings/``.

    Returns
    -------
    pathlib.Path
        Resolved preset directory (may not exist yet).
    """
    appdata = os.environ.get("APPDATA", "")
    if not appdata:
        raise EnvironmentError(
            "APPDATA environment variable is not set. "
            "Cannot locate the Lightroom preset directory."
        )
    return Path(appdata) / "Adobe" / "CameraRaw" / "Settings"


def install_xmp_to_lightroom(
    xmp_path: str,
    subfolder: str = "Channel Swap",
) -> str:
    """Copy an XMP preset into the Lightroom Camera Raw presets directory.

    Parameters
    ----------
    xmp_path : str
        Path to the source .xmp file.
    subfolder : str, optional
        Sub-folder inside the Settings directory.  Lightroom shows this as a
        preset group.

    Returns
    -------
    str
        The absolute path of the installed preset.

    Raises
    ------
    FileNotFoundError
        If *xmp_path* does not exist.
    """
    xmp_path = os.path.abspath(xmp_path)
    if not os.path.isfile(xmp_path):
        raise FileNotFoundError(f"XMP file not found: {xmp_path}")

    dest_dir = get_lightroom_preset_dir() / subfolder
    dest_dir.mkdir(parents=True, exist_ok=True)

    dest_path = dest_dir / os.path.basename(xmp_path)
    shutil.copy2(xmp_path, dest_path)

    return str(dest_path)
