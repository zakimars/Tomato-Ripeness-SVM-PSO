"""Tomato ripeness classification from TCS34725 RGB-C readings with SVM tuned by PSO."""

CLASSES = ["unripe", "partially_ripe", "ripe"]
FEATURES = ["r_norm", "g_norm", "b_norm"]
