SERVER_HOSTNAME = 'bohg-ws-14'
ROBOT_HOSTNAME_PREFIX = 'iprl-bot'
CONN_AUTHKEY = b'secret password'
REDIS_PASSWORD = 'secret password'

################################################################################
# Arm

# Arm-dependent heading compensation (set to 0 if unsure)
ARM_HEADING_COMPENSATION = {
    0: -0.7,  # Robot 1 (asset tag: none)
    1: 0.2,   # Robot 2 (asset tag: 000007 402760)
    2: 0.7,   # Robot 3 (asset tag: 000007 402746)
}

################################################################################
# Camera

CAMERA_SERIALS = {
    0: '634093BE',  # Robot 1
    1: '44251E9E',  # Robot 2
    2: '7E841E9E',  # Robot 3
}
CAMERA_FOCUS = 0
CAMERA_TEMPERATURE = 3900
CAMERA_EXPOSURE = 156
CAMERA_GAIN = 10
