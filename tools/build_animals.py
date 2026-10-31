"""Build "Halloween Animals": only the creatures from the Halloween mod (spider sheep,
ghost wild animals, skeleton boars, ghost fish). Trees, bushes, farms and buildings are
left alone, so it combines with grid, tree and fish-border mods."""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import modkit
import build_halloween as hw

MOD = os.path.join(modkit.MODS_LOCAL, "Halloween Animals")

DESCRIPTION = ("Just the animals from the Halloween mod. Herd animals become spider sheep, wild game "
               "becomes ghosts of itself, boars and javelinas are skeletons, and fish are ghosts. Trees, "
               "bushes, farms and buildings are untouched, so it works alongside grid, tree and "
               "fish-border mods. Graphics only: works in multiplayer and ranked. Don't enable it together "
               "with the full Halloween mod.")


if __name__ == "__main__":
    # the Halloween steps write wherever these point
    hw.MOD = MOD
    hw.GFX_OUT = os.path.join(MOD, "resources", "_common", "drs", "graphics")
    modkit.write_info(MOD, "Halloween Animals", DESCRIPTION)
    hw.herd_spiders()
    hw.pasture_ibex()
    hw.ghost_deer()
    hw.skeleton_pigs()
    hw.haunted_water(traps=False)
    print("done ->", MOD)
