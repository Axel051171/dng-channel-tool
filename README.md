# DNG Channel Tool

**All-in-one color channel swap, IR photography toolkit, and universal camera preset converter.**

A Python GUI tool for photographers who work with infrared photography, color grading, and custom camera presets across multiple camera systems (Nikon, Canon, Sony, Fujifilm) and editing software (Adobe Lightroom, DaVinci Resolve, Capture One).

![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)
![License MIT](https://img.shields.io/badge/license-MIT-green.svg)
![Platform Windows/Mac/Linux](https://img.shields.io/badge/platform-Win%20%7C%20Mac%20%7C%20Linux-lightgrey.svg)

---

## Features

### Color Channel Swap & Mix
- All 6 RGB permutations (R↔B, R↔G, G↔B, rotations)
- Weighted 3x3 channel mixing matrix with sliders
- 8 IR-optimized presets (Blue Sky, Goldie, Aerochrome, Chocolate, etc.)
- Live image preview with before/after split view
- RGB histogram

### Infrared Photography Tools
- **IR White Balance Assistant** — click on vegetation to auto-calculate WB
- **8 False-Color Presets** — Classic Blue Sky, Kodak Aerochrome, Super Color, Goldie, Chocolate, Dream, Candy, B&W IR
- **IR Filter Simulation** — see how a photo would look with 590nm/630nm/680nm/720nm/850nm filters
- **Hotspot Detection** — detect and correct lens hotspots in IR images
- **NDVI Calculation** — Normalized Difference Vegetation Index with color-coded map
- **Custom IR DCP Profiles** — camera-specific profiles per filter type

### Camera Preset Support

| Format | Read | Write | Camera System |
|--------|------|-------|---------------|
| Nikon NPC/NP3 | Yes | Yes | Nikon D-SLR & Z-series |
| Canon PF3 | Yes | Yes | Canon EOS |
| Sony Creative Look | Yes | Yes | Sony Alpha |
| Fujifilm Recipes | Yes (text) | → XMP/NPC | Fujifilm X-series |
| Adobe DCP | Yes | Yes | Lightroom / Camera Raw profiles |
| Adobe XMP | — | Yes | Lightroom develop presets |
| 3D LUT (.cube) | — | Yes | DaVinci Resolve, Premiere, OBS, etc. |
| ICC Profile | — | Yes | Color management |

### Cross-System Conversion
Convert presets between all supported camera systems:
- Nikon ↔ Canon ↔ Sony ↔ Fujifilm
- Any camera preset → Adobe Lightroom XMP
- Any camera preset → 3D LUT (.cube)
- Any camera preset → DCP camera profile

### NEF Picture Control Extraction
- Extract embedded Picture Controls from Nikon NEF files
- Includes tone curve, contrast, brightness, saturation, clarity, sharpness
- Convert directly to Lightroom presets or Nikon NPC files
- One-click install to Adobe Camera Raw

### Style Transfer
- **Analyze** any image to extract its color style (tone curve, contrast, saturation)
- **Transfer** a reference image's style to your photos
- **Compare** original + edited versions to extract the exact transformation
- **Histogram Matching** for precise color distribution transfer

### DCP Profile Tools (NEW in v2.0)
- **DCP Camera Model Rewrite** — change camera model in DCP files (e.g. Z6 to Z6 II)
- **DCP to XML / XML to DCP** — decompile/compile profiles (dcpTool-compatible format)
- **Make Invariant** — merge LookTable into HueSatMap for predictable profiles
- **UnTwist** — remove value-dependent hue shifts from profiles
- **PGM to DNG Converter** — create DNG files from raw sensor data with CFA patterns

### Additional Tools
- **White Balance Picker** — click on neutral gray to calculate WB correction
- **Color Checker Calibration** — photograph an X-Rite ColorChecker, auto-generate DCP profile
- **Camera JPEG vs RAW comparison** — split-view of camera rendering vs neutral RAW
- **Preset Library** — browse all installed presets (Adobe, Nikon, LUTs) with search
- **Batch Processing** — apply channel swap to multiple files at once
- **Export All** — DCP + XMP + LUT + ICC with one click (Ctrl+Shift+E)
- **Drag & Drop** — drop images and profiles onto the window
- **Recent Files** — quickly reopen last 10 files
- **705 camera database** — color matrices from dnglab for DCP profile generation
- **Zoom/Pan** — mouse wheel zoom, drag to pan, double-click reset

---

## Installation

### Quick Start (Windows)
```bash
git clone https://github.com/Axel051171/dng-channel-tool.git
cd dng-channel-tool
run.bat
```

### Manual Installation
```bash
git clone https://github.com/Axel051171/dng-channel-tool.git
cd dng-channel-tool
pip install -r requirements.txt
python main.py
```

### Requirements
- **Python 3.10+**
- **numpy** — array operations
- **Pillow** — image loading/saving
- **rawpy** *(optional)* — RAW file support (NEF, CR2, ARW, DNG, etc.)
- **exifread** *(optional)* — EXIF/MakerNote parsing for NEF extraction

```bash
pip install numpy Pillow rawpy exifread
```

### Camera Database (Optional)
For the 705-camera color matrix database, clone [dnglab](https://github.com/dnglab/dnglab) as a sibling directory:
```
parent/
  dng-channel-tool/   (this repo)
  dnglab/             (clone of dnglab)
```

---

## Usage

### GUI
```bash
python main.py
```

### CLI (Headless Mode)
```bash
# Channel swap DCP profile
python main.py --swap RB --camera "NIKON Z 8" --export-dcp output.dcp

# Export all formats at once
python main.py --swap RB --camera "NIKON Z 8" \
  --export-dcp out.dcp --export-xmp out.xmp \
  --export-lut out.cube --export-icc out.icc

# Custom mix matrix
python main.py --mix "0,0,1,0,1,0,1,0,0" --camera "Canon EOS R5" --export-dcp ir.dcp

# Process image with channel swap
python main.py input.jpg --swap RB --export-image output.jpg

# Use existing DCP as base
python main.py --load-dcp "Adobe Standard.dcp" --swap RB --camera "NIKON Z 6_2" --export-dcp swapped.dcp
```

### DCP XML Tools (CLI)
```bash
# Decompile DCP to XML
python dcp_xml.py -d profile.dcp profile.xml

# Compile XML back to DCP
python dcp_xml.py -c profile.xml profile.dcp

# Make DCP invariant (merge LookTable)
python dcp_xml.py -i profile.dcp invariant.dcp

# Remove hue twists
python dcp_xml.py -u profile.dcp untwisted.dcp
```

### PGM to DNG (CLI)
```bash
python dng_writer.py input.pgm output.dng --pattern=RGGB --camera="NIKON Z 8" --wp=0.47,1.0,0.63
```

### Keyboard Shortcuts
| Key | Action |
|-----|--------|
| `Ctrl+O` | Open image |
| `Ctrl+S` | Save image |
| `Ctrl+E` | Export DCP profile |
| `Ctrl+N` | Extract NEF Picture Control |
| `Ctrl+Shift+E` | Export all (DCP+XMP+LUT+ICC) |
| `Ctrl+B` | Batch processing |
| `Ctrl+L` | Preset library |
| `V` | Toggle before/after split view |
| `W` | White balance picker |
| Mouse wheel | Zoom in/out |
| Drag | Pan image |
| Double-click | Reset zoom |

### Infrared Workflow
1. Open your IR photo (`Ctrl+O`)
2. **Infrared → IR White Balance** — click on vegetation
3. **Infrared → False-Color Presets → Classic Blue Sky**
4. Fine-tune with channel mix sliders
5. Export as:
   - **DCP** for Lightroom (`Ctrl+E`)
   - **3D LUT** for Resolve/Premiere (`File → 3D LUT`)
   - **NPC/NP3** for Nikon camera (`Extras → Nikon Preset`)

### Fujifilm Recipe Conversion
1. **Extras → Fujifilm Recipe** — paste recipe text from [fujixweekly.com](https://fujixweekly.com)
2. Click **Parse & Preview**
3. Export as Lightroom XMP, Nikon NPC, or 3D LUT

### NEF Extraction
1. **Extras → NEF Picture Control extrahieren**
2. Select your .NEF file
3. View tone curve, parameters, preview
4. **Install to Lightroom** — one click

---

## File Structure
```
dng-channel-tool/
├── main.py              # GUI application + CLI entry point
├── channel_swap.py      # Channel swap & mix logic
├── dcp_io.py            # Adobe DCP profile read/write
├── dcp_xml.py           # DCP ↔ XML conversion, make invariant, untwist
├── dng_writer.py        # PGM → DNG converter
├── xmp_export.py        # Lightroom XMP preset export
├── npc_io.py            # Nikon NPC/NP3 read/write
├── nef_extract.py       # NEF Picture Control extraction
├── fuji_recipe.py       # Fujifilm recipe parser
├── camera_presets.py    # Canon PF3 + Sony Creative Look
├── camera_db.py         # Camera color matrix database
├── ir_tools.py          # Infrared photography tools
├── style_transfer.py    # Image style analysis & transfer
├── lut_export.py        # 3D LUT (.cube) export
├── wb_picker.py         # White balance + histogram matching
├── color_checker.py     # ColorChecker calibration
├── preset_library.py    # Preset browser/manager
├── icc_export.py        # ICC v2 profile export
├── gui_dialogs.py       # Dialog windows
├── gui_widgets.py       # Reusable UI components
├── undo.py              # Undo/Redo system
├── logging_setup.py     # Logging configuration
├── tests/
│   └── test_core.py     # 119 unit tests
├── pyproject.toml
├── run.bat              # Windows quick start
└── LICENSE
```

---

## Export Formats

### 3D LUT (.cube)
Universal lookup tables for DaVinci Resolve, Adobe Premiere, Final Cut Pro, OBS, Capture One, and mobile apps. 33x33x33 grid.

### DCP (DNG Camera Profile)
Adobe Camera Raw / Lightroom camera profiles with color matrices, forward matrices, HueSatMap data, and tone curves. Installed to `%APPDATA%/Adobe/CameraRaw/CameraProfiles/`.

### XMP Presets
Lightroom develop presets with tone curve, contrast, saturation, clarity, white balance, grain, and camera profile reference. Installed to `%APPDATA%/Adobe/CameraRaw/Settings/`.

### Nikon NPC/NP3
Nikon Picture Control files for D-SLR (.NPC) and Z-series (.NP3) cameras. Copy to SD card under `/NIKON/CUSTOMPC/` and import via camera menu.

### ICC Profiles
ICC v2 Display profiles for color-managed workflows.

---

## Contributing
Pull requests welcome! Areas that could use help:
- More IR false-color presets
- Better ColorChecker auto-detection (OpenCV)
- Fujifilm recipe database
- macOS/Linux testing
- Translations

---

## License
MIT License. See [LICENSE](LICENSE).

---

## Credits
- Camera database from [dnglab](https://github.com/dnglab/dnglab)
- Fujifilm recipe format from [Fuji X Weekly](https://fujixweekly.com)
- DCP format based on Adobe DNG Specification
- Nikon NPC format reverse-engineered from Nikon Picture Control Utility 2
