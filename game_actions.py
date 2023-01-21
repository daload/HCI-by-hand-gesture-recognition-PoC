import logging as log

from constants import gestures


def throttle(key, flags, gamepad):
    """
    Simulate turning right input on a virtual controller

    Parameters
    ----------
    key: string - Label of the gesture (either performed or not)
    flags: dict - Dictionary with the inputs to perform
    gamepad : VX360Gamepad - Virtual controller
    """
    value = flags[key]
    gamepad.right_trigger_float(value_float=value)
    if value == 1:
        log.info("Accelerating")


def brake(key, flags, gamepad):
    """
    Simulate turning right input on a virtual controller

    Parameters
    ----------
    key: string - Label of the gesture (either performed or not)
    flags: dict - Dictionary with the inputs to perform
    gamepad : VX360Gamepad - Virtual controller
    """
    value = flags[key]
    gamepad.left_trigger_float(value_float=value)
    if value == 1:
        log.info("Breaking")


def turnLeft(key, flags, gamepad):
    """
    Simulate turning right input on a virtual controller

    Parameters
    ----------
    key: string - Label of the gesture (either performed or not)
    flags: dict - Dictionary with the inputs to perform
    gamepad : VX360Gamepad - Virtual controller
    """
    value = flags[key]
    if flags[gestures["right"]] == 0:
        gamepad.left_joystick_float(x_value_float=-value / 1.25, y_value_float=0)
    if value == 1:
        log.info("Turning left")


def turnRight(key, flags, gamepad):
    """
    Simulate turning right input on a virtual controller

    Parameters
    ----------
    key: string - Label of the gesture (either performed or not)
    flags: dict - Dictionary with the inputs to perform
    gamepad : VX360Gamepad - Virtual controller
    """
    value = flags[key]
    if flags[gestures["left"]] == 0:
        gamepad.left_joystick_float(x_value_float=value / 1.25, y_value_float=0)
    if value == 1:
        log.info("Turning right")


def turn(turning_input, gamepad):
    """
    Simulate turning input on a virtual controller

    Parameters
    ----------
    turning_input : float - Value to be turned (1/-1 -> 100% turning, 0 -> no turning)
    gamepad : VX360Gamepad - Virtual controller
    """
    gamepad.left_joystick(x_value=round(turning_input * 32767), y_value=0)
    direction = "Not turning"
    if turning_input > 0:
        direction = "Turning right {}%".format(round(turning_input * 100))
    if turning_input < 0:
        direction = "Turning left {}%".format(round(turning_input * 100) * -1)
    log.info(direction)


actions = {"throttle": throttle, "brake": brake, "right": turnRight, "left": turnLeft}
