import enum


# creating enumerations using class
class WeightType(enum.Enum):
    RANDOM = 0
    XAVIER = 1
    KAIMING = 2
    LAMARCKIAN = 3
    NONE = -1

WEIGHT_TYPES_STRING  = ["random", "xavier", "kaiming", "lamarckian"]
NUM_WEIGHT_TYPES = 4

def get_enum_from_string(input_string):
    if input_string == "random":
        return WeightType.RANDOM

    if input_string == "xavier":
        return WeightType.XAVIER

    if input_string == "lamarckian":
        return WeightType.LAMARCKIAN

    if input_string == "kaiming":
        return WeightType.KAIMING

    return WeightType.NONE

