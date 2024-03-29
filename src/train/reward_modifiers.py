import math

from typing import Callable, Dict, Tuple
from cereal import log

# The way this function works is that while you are loading a MsgVecDataset
# it will call this function for each message in the log.  Then, when it comes time to
# record the reward, the last result of this function will be added to it.

RewardModifierFunc = Callable[[log.Event, Dict], Tuple[float, Dict]]

def default_reward_modifier(msg: log.Event, state: Dict) -> Tuple[float, Dict]:
    return 0.0, state

def _clamp(val: float, min: float, max: float) -> float:
    if val < min:
        return min
    elif val > max:
        return max
    else:
        return val

# In this first version, we penalize moving backwards
def reward_modifier_penalize_move_backwards(msg: log.Event, state: Dict) -> Tuple[float, Dict]:
    PENALIZE_BACKWARDS_SCALE = 0.02

    if msg.which() == "odriveFeedback":
        state["lastLeftVelocity"] = msg.odriveFeedback.leftMotor.vel
        state["lastRightVelocity"] = msg.odriveFeedback.rightMotor.vel

    mod = 0.0

    # Discourage moving backwards
    if "lastLeftVelocity" in state and "lastRightVelocity" in state:
        velDif = state["lastRightVelocity"] - state["lastLeftVelocity"]
        if velDif < 0:
            mod += _clamp(velDif, -1.0, 0.0) * PENALIZE_BACKWARDS_SCALE

    return mod, state

def reward_modifier_penalize_fast_move_backwards(msg: log.Event, state: Dict) -> Tuple[float, Dict]:
    PENALIZE_BACKWARDS_SCALE = 0.02
    PENALIZE_FAST_MOTORS_SCALE = 0.05
    PENALIZE_LARGE_GYRO_SCALE = 0.05

    if msg.which() == "odriveFeedback":
        state["lastLeftVelocity"] = msg.odriveFeedback.leftMotor.vel
        state["lastRightVelocity"] = msg.odriveFeedback.rightMotor.vel
    elif msg.which() == "gyroscope" and msg.gyroscope.sensor == 3:
        state["lastGyro"] = list(msg.gyroscope.gyro.v)

    mod = 0.0

    # Discourage moving backwards
    if "lastLeftVelocity" in state and "lastRightVelocity" in state:
        velDif = state["lastRightVelocity"] - state["lastLeftVelocity"]
        if velDif < 0:
            mod += _clamp(velDif, -1.0, 0.0) * PENALIZE_BACKWARDS_SCALE

        # Discourage high motor values
        if abs(state["lastLeftVelocity"]) > 0.5:
            mod -= _clamp(abs(state["lastLeftVelocity"]), 0.0, 1.0) * PENALIZE_FAST_MOTORS_SCALE

        if abs(state["lastRightVelocity"]) > 0.5:
            mod -= _clamp(abs(state["lastRightVelocity"]), 0.0, 1.0) * PENALIZE_FAST_MOTORS_SCALE

    if "lastGyro" in state:
        mag = math.sqrt(sum([x * x for x in state["lastGyro"]]))
        if mag > 0.5:
            mod -= _clamp(mag, 0.0, 1.0) * PENALIZE_LARGE_GYRO_SCALE
    
    return mod, state


