"""
Set up GRASS GIS environment for running ITZI.
This script handles the environment setup that grass84.bat normally does,
but in a way that works from any Python environment.
"""
import os
import sys
import subprocess

GRASS_BASE = r"E:\Tools\GRASS-GIS-8.4.2"


def setup_grass_env():
    """Set up environment variables for GRASS GIS."""
    gisbase = GRASS_BASE

    # Set GRASS environment variables
    os.environ["GISBASE"] = gisbase
    os.environ["GRASS_PYTHON"] = os.path.join(gisbase, "extrabin", "python3.exe")
    os.environ["PYTHONHOME"] = os.path.join(gisbase, "Python312")
    os.environ["GRASS_PROJSHARE"] = os.path.join(gisbase, "share", "proj")
    os.environ["PROJ_LIB"] = os.path.join(gisbase, "share", "proj")
    os.environ["GDAL_DATA"] = os.path.join(gisbase, "share", "gdal")
    os.environ["FONTCONFIG_FILE"] = os.path.join(gisbase, "etc", "fonts.conf")

    # Set PATH - include all GRASS binary directories
    grass_paths = [
        os.path.join(gisbase, "lib"),
        os.path.join(gisbase, "bin"),
        os.path.join(gisbase, "extrabin"),
        os.path.join(gisbase, "Python312"),
        os.path.join(gisbase, "Python312", "Scripts"),
    ]
    existing_path = os.environ.get("PATH", "")
    new_path = os.pathsep.join(grass_paths + [existing_path])
    os.environ["PATH"] = new_path

    # Set Python path for GRASS modules
    grass_python_path = os.path.join(gisbase, "etc", "python")
    sys.path.insert(0, grass_python_path)

    # On Python 3.8+, add DLL search path
    if sys.version_info >= (3, 8):
        os.add_dll_directory(os.path.join(gisbase, "lib"))
        os.add_dll_directory(os.path.join(gisbase, "bin"))

    print(f"GRASS GIS environment set up:")
    print(f"  GISBASE = {gisbase}")
    return gisbase


def test_grass():
    """Test that GRASS GIS works properly."""
    setup_grass_env()

    # Import GRASS scripting
    import grass.script as gscript

    # Create a temporary location to test
    import tempfile
    tmpdir = tempfile.mkdtemp(prefix="grass_test_")
    tmp_loc = os.path.join(tmpdir, "test_loc")
    os.makedirs(tmp_loc, exist_ok=True)

    # Test that GRASS can create a location
    print(f"Testing GRASS with temp location: {tmpdir}")
    try:
        # Create a session
        gs = gscript.setup.init(
            path=tmpdir,
            location="test_loc",
            mapset="PERMANENT",
            grass_path=os.path.join(GRASS_BASE, "bin", "grass84"),
        )
        print("  GRASS session created successfully!")

        # Test basic commands
        result = gscript.parse_command("g.version", flags="g")
        print(f"  GRASS version: {result.get('version', 'unknown')}")

        # Clean up
        gs.finish()
    except Exception as e:
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)

    print("GRASS GIS test complete!")


if __name__ == "__main__":
    test_grass()
