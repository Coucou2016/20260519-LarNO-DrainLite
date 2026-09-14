import os, shutil

dst = r'E:\Tools\GRASS-GIS-8.4.2'

# List all items
items = os.listdir(dst)
print(f"Items in destination: {items}")

# Find the NSIS directory (ends with 89_)
nsis_dir = None
for item in items:
    if item.endswith('89_') and os.path.isdir(os.path.join(dst, item)):
        nsis_dir = os.path.join(dst, item)
        print(f"Found NSIS dir: {nsis_dir}")
        break

if not nsis_dir:
    print("NSIS directory not found!")
    exit(1)

# Move all files from NSIS dir to root
for item in os.listdir(nsis_dir):
    s = os.path.join(nsis_dir, item)
    d = os.path.join(dst, item)
    if os.path.exists(d):
        if os.path.isdir(d):
            shutil.rmtree(d)
        else:
            os.remove(d)
    shutil.move(s, d)
    print(f"  Moved: {item}")

# Check for $_90_ (second NSIS output)
for item in os.listdir(dst):
    if item.endswith('90_') and os.path.isdir(os.path.join(dst, item)):
        src2 = os.path.join(dst, item)
        for f2 in os.listdir(src2):
            s2 = os.path.join(src2, f2)
            d2 = os.path.join(dst, f2)
            if not os.path.exists(d2):
                shutil.move(s2, d2)
        os.rmdir(src2)
        print(f"  Merged: {item}")

# Remove empty NSIS dirs
for item in os.listdir(dst):
    p = os.path.join(dst, item)
    if os.path.isdir(p) and ('PLUGINSDIR' in item or item.endswith('89_') or item.endswith('90_')):
        try:
            shutil.rmtree(p)
            print(f"  Removed: {item}")
        except:
            pass

# Verify
print("\nFinal directory structure:")
for item in sorted(os.listdir(dst)):
    p = os.path.join(dst, item)
    is_dir = os.path.isdir(p)
    print(f"  {'DIR' if is_dir else 'FILE':5s}  {item}")
