class TrackerConfig(object):
    MIN_CONFIDENCE = 0.3
    MAX_DIST = 0.32
    MAX_IOU = 0.7
    N_INIT = 10
    MAX_AGE = 2000
    NN_BUDGET = 1000

# GOOD FOR VMD:
    # MIN_CONFIDENCE = 0.3
    # MAX_DIST = 0.3
    # MAX_IOU = 0.7
    # N_INIT = 6
    # MAX_AGE = 200

    # MIN_CONFIDENCE = 0.3
    # MAX_DIST = 0.25
    # MAX_IOU = 0.7
    # N_INIT = 30
    # MAX_AGE = 2000

# GOOD FOR DNC:
#     MIN_CONFIDENCE = 0.3
#     MAX_DIST = 0.205
#     MAX_IOU = 0.8
#     N_INIT = 15
#     MAX_AGE = 200
