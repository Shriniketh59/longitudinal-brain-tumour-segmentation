"""Keep tests hermetic and light: never load the real ~850 MB of model weights.

Loading them inside the test process (on top of a running API) exhausted RAM on a
7.6 GB machine. Tests build tiny random models themselves where they need one.
"""

import os

for var, name in (
    ("CLASSIFIER_WEIGHTS", "vgg16_classifier.pt"),
    ("UNET_WEIGHTS", "unet_vgg16.pt"),
    ("SWIN_WEIGHTS", "swin_unetr_brats.pt"),
):
    os.environ.setdefault(var, f"/nonexistent/test-weights/{name}")
