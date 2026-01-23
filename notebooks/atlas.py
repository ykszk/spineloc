# %%
import numpy as np

from spineloc.data.spine_radiograph import RadiographView, SpineRadiograph, SpineRadiographAtlas

# Create a SpineRadiograph instance
sr_frontal = SpineRadiograph.from_labelme_file("../data/radiopedia/raw/case4_frontal.json")
sr_lateral = SpineRadiograph.from_labelme_file("../data/radiopedia/raw/case4_lateral.json")

# %%
atlas = SpineRadiographAtlas(frontal=sr_frontal, lateral_left=sr_lateral)

# Locate a bounding box in anatomical coordinates
# bbox_anat = [0, 0, 10, 10]
# bbox_anat = [-2.2256317138671875, -5.746438980102539, 1.38214111328125, 13.472418785095215]
# bbox_anat = [-1.2242168188095093, -7.582393169403076, 8.611021995544434, 15.498279571533203]
bbox_anat = [5.58205343, -4.92156832, -13.40964125, 14.0433815]
bbox_anat = np.array(bbox_anat)

image_with_bbox = atlas.locate(bounding_box=bbox_anat, view=RadiographView.LATERAL_RIGHT)

# %%
from PIL import Image

img = Image.fromarray(np.array(image_with_bbox))
img
# %%
img.save("/tmp/atlas_frontal.png")
# %%
