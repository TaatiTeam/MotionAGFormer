H36M_JOINT_TO_LABEL = {
    0: "Bottom torso",
    1: "Left hip",
    2: "Left knee",
    3: "Left foot",
    4: "Right hip",
    5: "Right knee",
    6: "Right foot",
    7: "Center torso",
    8: "Upper torso",
    9: "Neck base",
    10: "Center head",
    11: "Right shoulder",
    12: "Right elbow",
    13: "Right hand",
    14: "Left shoulder",
    15: "Left elbow",
    16: "Left hand"
}

H36M_TO_MPI = {
    0: 14,  # Sacrum
    1: 8,  # LHip
    2: 9,  # LKnee
    3: 10,  # LAnkle
    4: 11,  # RHip
    5: 12,  # RKnee
    6: 13,  # RAnkle
    7: 15,  # Spine
    8: 1,  # SpineShoulder
    9: 16,  # Neck
    10: 0,  # Head
    11: 5,  # RShoulder
    12:  6,  # RElbow
    13:  7,  # RHand
    14:  2,  # LShoulder
    15:  3,  # LElbow
    16:  4,  # LHand
}

H36M_LOWER_BODY_JOINTS = list(range(1, 7))
H36M_UPPER_BODY_JOINTS = list(range(7, 17))

H36M_0_DF = [0, 7, 8]
H36M_1_DF = [1, 4, 9, 11, 14]
H36M_2_DF = [2, 5, 10, 12, 15]
H36M_3_DF = [3, 6, 13, 16]
