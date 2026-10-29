"""List game sprites the Halloween mod should replace but doesn't, so new civs or DLC
graphics that slipped past the build's file patterns show up."""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(__file__))
import modkit

MOD_GFX = os.path.join(modkit.MODS_LOCAL, "Halloween", "resources", "_common", "drs", "graphics")
GROUPS = {
    "town centers": r"town_center",
    "herd animals": r"^a_herd_",
    "huntables": r"^a_hunt_",
    "fish": r"^a_fish_",
    "fish traps": r"fish_trap(?!.*foundation)",
    "pastures": r"pasture",
    "forage": r"^n_forage_",
    "trees": r"^n_tree_(?!.*(stump|felled|reeds))",
}

game = sorted(f for f in os.listdir(modkit.GAME_GFX) if f.endswith("_x1.sld"))
mod = set(os.listdir(MOD_GFX))
missing = 0
for group, pat in GROUPS.items():
    files = [f for f in game if re.search(pat, f)]
    gaps = [f for f in files if f not in mod]
    missing += len(gaps)
    print(f"{group}: {len(files) - len(gaps)}/{len(files)} replaced")
    for f in gaps:
        print("   missing", f[:-4])

cultures = sorted({m.group(1) for f in game if (m := re.match(r"^b_([a-z]+)_town_center_age", f))})
print(f"\nTown Center cultures in this game install: {len(cultures)} ({', '.join(cultures)})")
print("all covered" if not missing else f"{missing} sprite(s) not replaced")
