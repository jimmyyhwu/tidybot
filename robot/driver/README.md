# Mobile Base Driver

## Setup

The following steps should be run on the mobile base computer:

1. Download the driver code and reset to known working commit:

    ```bash
    git clone git@github.com:stanford-iprl-lab/Mobile_Manipulation_Dev.git --branch camera_traj_tracking
    git reset --hard 081a03f9f3efee7cd2ec67c7de5bca077a5b35de
    ```

2. Apply Jimmy's driver code patch:

    ```bash
    git apply driver.patch
    ```

3. Build the driver code:

    ```bash
    cd  ~/Mobile_Manipulation_Dev
    mkdir build
    cd build
    cmake .. && make
    ```

4. Copy motor offsets to the location expected by the driver:

    ```bash
    cp motor-offsets-$(hostname | rev | cut -c1).txt ~/Mobile_Manipulation_Dev/bin/.motor_cal.txt
    ```

    Note: `motor-offsets-1.txt` goes with mobile base #1, `motor-offsets-2.txt` with mobile base #2, etc.
