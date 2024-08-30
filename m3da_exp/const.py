RANDOM_STATE = 0xBadFace

# AMOS approx median spacing modality-wise:
AMOS_CT_SPACING = (0.675, 0.675, 5.0)
AMOS_MRI_SPACING = (1.1875, 1.1875, 3.0)
# the common spacing (min + Z adjustment) for both datasets:
AMOS_COMMON_SPACING = (0.675, 0.675, 3.0)

# CT noise parameters:
# corresponds to N ~= 4096. N -- Mean photon count pet detector bin without attenuation.
LDCT_NOISE_INTENSITY = 0.04096

# CT kernel modulation parameters:
CT_SOFT_A_RANGE = (-0.8, 0.0)
CT_SOFT_B_RANGE = (0.6, 1.0)
CT_SOFT_A = -1
CT_SOFT_B = 0.5

CT_SHARP_A_RANGE = (20.0, 50.0)
CT_SHARP_B_RANGE = (2.5, 4.0)
CT_SHARP_A = 60
CT_SHARP_B = 5

# nnUnet suggested spacing:
CC359_SPACING = (1.0, 1.0, 1.0)
LIDC_SPACING = (0.703125, 0.703125, 1.25)
