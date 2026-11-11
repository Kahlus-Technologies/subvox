# dmgbuild settings: window 660x400 matches the 660x400pt background (1320x800 @2x).
import os
app = os.environ["APP"]
format = "UDZO"
files = [app]
symlinks = {"Applications": "/Applications"}
background = "dmg/background.tiff"
icon_size = 104
text_size = 13
window_rect = ((200, 120), (660, 428))
default_view = "icon-view"
show_status_bar = False
show_tab_view = False
show_toolbar = False
show_pathbar = False
show_sidebar = False
icon_locations = {"Subvox.app": (175, 190), "Applications": (485, 190)}
