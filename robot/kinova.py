# Author: Jimmy Wu
# Date: February 2023

import math
import os
import subprocess
import threading
import time

import numpy as np
from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient
from kortex_api.autogen.client_stubs.ControlConfigClientRpc import ControlConfigClient
from kortex_api.autogen.messages import ActuatorCyclic_pb2, Base_pb2, BaseCyclic_pb2, ControlConfig_pb2, Session_pb2
from kortex_api.Exceptions.KServerException import KServerException
from kortex_api.RouterClient import RouterClient, RouterClientSendOptions
from kortex_api.SessionManager import SessionManager
from kortex_api.TCPTransport import TCPTransport
from kortex_api.UDPTransport import UDPTransport

class KinovaArm:
    ACTION_TIMEOUT_DURATION = 20

    def __init__(self):
        # Check if arm is connected
        try:
            subprocess.run(['ping', '-c', '1',  '192.168.1.10'], check=True, timeout=1, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except subprocess.TimeoutExpired as e:
            raise Exception('Could not communicate with arm') from e

        # Setup
        self.tcp_connection = DeviceConnection.createTcpConnection()
        self.udp_connection = DeviceConnection.createUdpConnection()
        self.base = BaseClient(self.tcp_connection.__enter__())
        self.base_cyclic = BaseCyclicClient(self.udp_connection.__enter__())
        self.control_config = ControlConfigClient(self.base.router)
        self.num_joints = self.base.GetActuatorCount().count
        self.movej_primitive = MoveJController(self.base, self.base_cyclic)

        # Action topic notifications
        self.end_or_abort_event = threading.Event()
        self.end_or_abort_event.set()
        def check_for_end_or_abort(e):
            def check(notification, e=e):
                #print("EVENT : " + Base_pb2.ActionEvent.Name(notification.action_event))
                if notification.action_event in (Base_pb2.ACTION_END, Base_pb2.ACTION_ABORT):
                    e.set()
            return check
        self.notification_handle = self.base.OnNotificationActionTopic(
            check_for_end_or_abort(self.end_or_abort_event),
            Base_pb2.NotificationOptions()
        )

    def disconnect(self):
        self.base.Unsubscribe(self.notification_handle)
        self.tcp_connection.__exit__()
        self.udp_connection.__exit__()

    def ready(self):
        return self.end_or_abort_event.is_set()

    def wait_ready(self):
        self.end_or_abort_event.wait(KinovaArm.ACTION_TIMEOUT_DURATION)

    def set_high_level_servoing(self):
        base_servo_mode = Base_pb2.ServoingModeInformation()
        base_servo_mode.servoing_mode = Base_pb2.SINGLE_LEVEL_SERVOING
        self.base.SetServoingMode(base_servo_mode)

    def _reference_action(self, action_name, blocking=True):
        action_type = Base_pb2.RequestedActionType()
        action_type.action_type = Base_pb2.REACH_JOINT_ANGLES
        action_list = self.base.ReadAllActions(action_type)
        action_handle = None
        for action in action_list.action_list:
            if action.name == action_name:
                action_handle = action.handle
        if action_handle is None:
            return
        self.end_or_abort_event.clear()
        self.base.ExecuteActionFromReference(action_handle)
        if blocking:
            self.end_or_abort_event.wait(KinovaArm.ACTION_TIMEOUT_DURATION)

    def home(self, blocking=True):
        self._reference_action('Home', blocking=blocking)

    def retract(self, blocking=True):
        self._reference_action('Retract', blocking=blocking)

    def zero(self, blocking=True):
        self._reference_action('Zero', blocking=blocking)

    def move_angular(self, joint_positions, blocking=True):
        assert len(joint_positions) == self.num_joints
        action = Base_pb2.Action()
        for i in range(self.num_joints):
            joint_angle = action.reach_joint_angles.joint_angles.joint_angles.add()
            joint_angle.joint_identifier = i
            joint_angle.value = joint_positions[i]
        self.end_or_abort_event.clear()
        self.base.ExecuteAction(action)
        if blocking:
            self.end_or_abort_event.wait(KinovaArm.ACTION_TIMEOUT_DURATION)

    def move_cartesian(self, xyz, theta_xyz, blocking=True):
        action = Base_pb2.Action()
        cartesian_pose = action.reach_pose.target_pose
        cartesian_pose.x = xyz[0]
        cartesian_pose.y = xyz[1]
        cartesian_pose.z = xyz[2]
        cartesian_pose.theta_x = theta_xyz[0]
        cartesian_pose.theta_y = theta_xyz[1]
        cartesian_pose.theta_z = theta_xyz[2]
        self.end_or_abort_event.clear()
        self.base.ExecuteAction(action)
        if blocking:
            self.end_or_abort_event.wait(KinovaArm.ACTION_TIMEOUT_DURATION)

    def toss(self, arm_heading=0):
        joint_angles = [arm_heading, 50, 180, 250, 0, 260, 90]
        self.move_angular(joint_angles)
        if self.check_joint_angles(joint_angles):
            self.movej_primitive.execute([arm_heading, 20, 180, 325, 0, 25, 90], max_vel=140, max_accel=300, max_decel=200, gripper_release_ms=600)

    def _gripper_position_command(self, value, blocking=True):
        gripper_command = Base_pb2.GripperCommand()
        gripper_command.mode = Base_pb2.GRIPPER_POSITION
        finger = gripper_command.gripper.finger.add()
        finger.value = value
        self.base.SendGripperCommand(gripper_command)
        if blocking:
            self._gripper_wait()

    def _gripper_speed_command(self, value, blocking=True):
        gripper_command = Base_pb2.GripperCommand()
        gripper_command.mode = Base_pb2.GRIPPER_SPEED
        finger = gripper_command.gripper.finger.add()
        finger.value = value
        self.base.SendGripperCommand(gripper_command)
        if blocking:
            self._gripper_wait()

    def _gripper_wait(self):
        time.sleep(0.05)  # Wait 50 ms for gripper to begin moving
        gripper_request = Base_pb2.GripperRequest()
        gripper_request.mode = Base_pb2.GRIPPER_SPEED
        while True:
            gripper_measure = self.base.GetMeasuredGripperMovement(gripper_request)
            if len(gripper_measure.finger) == 0 or gripper_measure.finger[0].value == 0.0:
                break

    def open_gripper(self, blocking=True):
        self._gripper_position_command(0, blocking=blocking)
        #self._gripper_speed_command(1, blocking=blocking)

    def close_gripper(self, blocking=True):
        #self._gripper_position_command(1, blocking=blocking)
        self._gripper_speed_command(-1, blocking=blocking)  # Speed command enables gripper to continue closing if the object gets smaller

    def get_joint_angles(self):
        joint_angles = self.base.GetMeasuredJointAngles()
        return [joint_angle.value for joint_angle in joint_angles.joint_angles]

    def get_heading(self):
        return self.get_joint_angles()[0]

    def check_joint_angles(self, expected_joint_angles):
        diffs = np.abs(np.array(expected_joint_angles) - np.array(self.get_joint_angles()))
        max_diff = np.minimum(diffs, 360 - diffs).max()
        print(f'Check joint angles: max diff {max_diff:.4f} deg')
        return max_diff < 1  # 1 deg

    def get_tool_pose(self):
        pose = self.base.GetMeasuredCartesianPose()
        return (pose.x, pose.y, pose.z), (pose.theta_x, pose.theta_y, pose.theta_z)

    def get_gripper_position(self):
        gripper_request = Base_pb2.GripperRequest()
        gripper_request.mode = Base_pb2.GRIPPER_POSITION
        gripper_measure = self.base.GetMeasuredGripperMovement(gripper_request)
        if len(gripper_measure.finger) > 0:
            return gripper_measure.finger[0].value
        return None

    def compute_inverse_kinematics(self, xyz, theta_xyz, guess_joint_angles=None):
        ik_data = Base_pb2.IKData()
        ik_data.cartesian_pose.x = xyz[0]
        ik_data.cartesian_pose.y = xyz[1]
        ik_data.cartesian_pose.z = xyz[2]
        ik_data.cartesian_pose.theta_x = theta_xyz[0]
        ik_data.cartesian_pose.theta_y = theta_xyz[1]
        ik_data.cartesian_pose.theta_z = theta_xyz[2]
        if guess_joint_angles is None:
            guess_joint_angles = self.get_joint_angles()
        for joint_angle in guess_joint_angles:
            ik_data.guess.joint_angles.add().value = joint_angle
        try:
            computed_joint_angles = self.base.ComputeInverseKinematics(ik_data)
            return [(joint_angle.value % 360) for joint_angle in computed_joint_angles.joint_angles]
        except KServerException:
            return None

    def set_joint_limits(self, speed_limits=(60, 60, 60, 60, 60, 60, 60), acceleration_limits=(80, 80, 80, 80, 80, 80, 80), cartesian=False):
        if cartesian:
            joint_speed_soft_limits = ControlConfig_pb2.JointSpeedSoftLimits()
            joint_speed_soft_limits.control_mode = ControlConfig_pb2.CARTESIAN_TRAJECTORY
            joint_speed_soft_limits.joint_speed_soft_limits.extend(speed_limits)
            self.control_config.SetJointSpeedSoftLimits(joint_speed_soft_limits)
        else:
            joint_speed_soft_limits = ControlConfig_pb2.JointSpeedSoftLimits()
            joint_speed_soft_limits.control_mode = ControlConfig_pb2.ANGULAR_TRAJECTORY
            joint_speed_soft_limits.joint_speed_soft_limits.extend(speed_limits)
            self.control_config.SetJointSpeedSoftLimits(joint_speed_soft_limits)
            joint_acceleration_soft_limits = ControlConfig_pb2.JointAccelerationSoftLimits()
            joint_acceleration_soft_limits.control_mode = ControlConfig_pb2.ANGULAR_TRAJECTORY
            joint_acceleration_soft_limits.joint_acceleration_soft_limits.extend(acceleration_limits)
            self.control_config.SetJointAccelerationSoftLimits(joint_acceleration_soft_limits)

    def set_max_joint_limits(self):
        speed_limits = self.control_config.GetKinematicHardLimits().joint_speed_limits
        acceleration_limits = self.control_config.GetKinematicHardLimits().joint_acceleration_limits
        self.set_joint_limits(speed_limits, acceleration_limits)

    def reset_joint_limits(self):
        control_mode_information = ControlConfig_pb2.ControlModeInformation()
        for control_mode in [
            ControlConfig_pb2.ANGULAR_JOYSTICK,
            ControlConfig_pb2.CARTESIAN_JOYSTICK,
            ControlConfig_pb2.ANGULAR_TRAJECTORY,
            ControlConfig_pb2.CARTESIAN_TRAJECTORY,
            ControlConfig_pb2.CARTESIAN_WAYPOINT_TRAJECTORY,
        ]:
            control_mode_information.control_mode = control_mode
            self.control_config.ResetJointSpeedSoftLimits(control_mode_information)
        for control_mode in [
            ControlConfig_pb2.ANGULAR_JOYSTICK,
            ControlConfig_pb2.ANGULAR_TRAJECTORY,
        ]:
            control_mode_information.control_mode = control_mode
            self.control_config.ResetJointAccelerationSoftLimits(control_mode_information)

    def set_twist_linear_limit(self, limit):
        twist_linear_soft_limit = ControlConfig_pb2.TwistLinearSoftLimit()
        twist_linear_soft_limit.control_mode = ControlConfig_pb2.CARTESIAN_TRAJECTORY
        twist_linear_soft_limit.twist_linear_soft_limit = limit
        self.control_config.SetTwistLinearSoftLimit(twist_linear_soft_limit)

    def set_max_twist_linear_limit(self):
        limit = self.control_config.GetKinematicHardLimits().twist_linear  # 0.5
        self.set_twist_linear_limit(limit)

    def reset_twist_linear_limit(self):
        control_mode_information = ControlConfig_pb2.ControlModeInformation()
        control_mode_information.control_mode = ControlConfig_pb2.CARTESIAN_TRAJECTORY
        self.control_config.ResetTwistLinearSoftLimit(control_mode_information)

    def pause_action(self):
        self.base.PauseAction()

    def resume_action(self):
        self.base.ResumeAction()

    def stop_action(self):
        self.base.StopAction()

    def stop(self):
        self.base.Stop()

    def apply_emergency_stop(self):
        self.base.ApplyEmergencyStop()

    def in_fault(self):
        return self.base.GetArmState().active_state == Base_pb2.ARMSTATE_IN_FAULT

    def clear_faults(self):
        if self.in_fault():
            self.base.ClearFaults()
            while self.base.GetArmState().active_state != Base_pb2.ARMSTATE_SERVOING_READY:
                time.sleep(0.1)

class MoveJController:
    CYCLIC_STEP_SIZE = 0.001

    def __init__(self, base, base_cyclic):
        # Set real-time scheduling policy
        try:
            os.sched_setscheduler(0, os.SCHED_FIFO, os.sched_param(os.sched_get_priority_max(os.SCHED_FIFO)))
        except PermissionError:
            print('Failed to set real-time scheduling policy, please edit /etc/security/limits.d/99-realtime.conf')

        # Setup
        self.base = base
        self.base_cyclic = base_cyclic
        self.num_joints = self.base.GetActuatorCount().count
        self.base_command = BaseCyclic_pb2.Command()
        for _ in range(self.num_joints):
            self.base_command.actuators.add()
        self.motor_cmd = self.base_command.interconnect.gripper_command.motor_cmd.add()
        self.base_feedback = None
        self.send_options = RouterClientSendOptions()
        self.send_options.timeout_ms = 3
        self.cyclic_thread = None
        self.kill_the_thread = False
        self.cyclic_running = False
        self.trajectory = None
        self.gripper_release_ms = None

    @staticmethod
    def trapezoidal_motion_profile(pos_diff, max_vel=60, max_accel=80, max_decel=80, step_size=CYCLIC_STEP_SIZE):
        assert not pos_diff < 0

        # Duration of each phase
        if pos_diff < 0.5 * max_vel**2 / max_accel + 0.5 * max_vel**2 / max_decel:
            accel_duration = (pos_diff / (max_accel + max_accel**2 / max_decel))**0.5
            decel_duration = (max_accel / max_decel) * accel_duration
            const_duration = 0
        else:
            accel_duration = max_vel / max_accel
            decel_duration = max_vel / max_decel
            const_duration = pos_diff / max_vel - 0.5 * max_vel / max_accel - 0.5 * max_vel / max_decel
        total_duration = accel_duration + const_duration + decel_duration

        # Motion profile
        t = np.arange(0, total_duration + step_size, step_size, dtype=np.float32)
        pos = np.zeros_like(t)

        # Acceleration
        accel_idx = math.ceil(accel_duration / step_size)
        pos[:accel_idx] = 0.5 * max_accel * t[:accel_idx]**2

        # Constant speed
        decel_idx = math.ceil((accel_duration + const_duration) / step_size)
        pos[accel_idx:decel_idx] = 0.5 * max_accel * accel_duration**2 + max_vel * (t[accel_idx:decel_idx] - accel_duration)

        # Deceleration
        tmp = t[decel_idx:] - (accel_duration + const_duration)
        pos[decel_idx:] = 0.5 * max_accel * accel_duration**2 + max_vel * const_duration
        pos[decel_idx:] += (max_decel * decel_duration) * tmp - 0.5 * max_decel * tmp**2

        return t, pos

    def execute(self, joint_positions, max_vel=60, max_accel=80, max_decel=80, gripper_release_ms=600):
        assert len(joint_positions) == self.num_joints

        # Compute motion profile
        self.base_feedback = self.base_cyclic.RefreshFeedback()
        pos_diffs = [((joint_positions[i] - self.base_feedback.actuators[i].position) + 180) % 360 - 180 for i in range(self.num_joints)]
        max_pos_diff = max(abs(pos_diff) for pos_diff in pos_diffs)
        _, pos = self.trapezoidal_motion_profile(max_pos_diff, max_vel=max_vel, max_accel=max_accel, max_decel=max_decel)
        self.trajectory = np.zeros((pos.shape[0], self.num_joints), dtype=np.float32)
        for i, pos_diff in enumerate(pos_diffs):
            self.trajectory[:, i] = self.base_feedback.actuators[i].position + (pos_diff / max_pos_diff) * pos
        self.gripper_release_ms = gripper_release_ms

        # Start cyclic thread
        self._init_cyclic()
        while self.cyclic_running:
            try:
                time.sleep(0.1)
            except KeyboardInterrupt:
                break
        self._stop_cyclic()

    def _init_cyclic(self):
        base_servo_mode = Base_pb2.ServoingModeInformation()
        base_servo_mode.servoing_mode = Base_pb2.LOW_LEVEL_SERVOING
        self.base.SetServoingMode(base_servo_mode)

        for i in range(self.num_joints):
            self.base_command.actuators[i].flags = ActuatorCyclic_pb2.SERVO_ENABLE
            self.base_command.actuators[i].position = self.base_feedback.actuators[i].position
        self.base_feedback = self.base_cyclic.Refresh(self.base_command, 0, self.send_options)
        self.motor_cmd.position = self.base_feedback.interconnect.gripper_feedback.motor[0].position
        self.motor_cmd.velocity = 0
        self.motor_cmd.force = 100

        self.cyclic_thread = threading.Thread(target=self._run_cyclic, daemon=True)
        self.cyclic_thread.start()

    def _run_cyclic(self):
        self.cyclic_running = True

        cyclic_count = 0
        failed_cyclic_count = 0

        t_now = time.time()
        t_cyclic = t_now
        step_times = []
        # data = []
        while not self.kill_the_thread:
            if self.kill_the_thread:
                break

            t_now = time.time()
            if (t_now - t_cyclic) >= MoveJController.CYCLIC_STEP_SIZE:
                step_times.append(t_now - t_cyclic)
                t_cyclic = t_now

                for i in range(self.num_joints):
                    self.base_command.actuators[i].position = self.trajectory[cyclic_count][i]

                self.base_command.frame_id += 1
                if self.base_command.frame_id > 65535:
                    self.base_command.frame_id = 0
                for i in range(self.num_joints):
                    self.base_command.actuators[i].command_id = self.base_command.frame_id

                if cyclic_count >= self.gripper_release_ms:
                    self.motor_cmd.position = 0
                    self.motor_cmd.velocity = 100

                try:
                    self.base_feedback = self.base_cyclic.Refresh(self.base_command, 0, self.send_options)
                except:
                    failed_cyclic_count = failed_cyclic_count + 1

                # data.append({
                #     'timestamp': t_now,
                #     'position': [actuator.position for actuator in self.base_feedback.actuators],
                #     'velocity': [actuator.velocity for actuator in self.base_feedback.actuators],
                #     'torque': [actuator.torque for actuator in self.base_feedback.actuators],
                #     'current_motor': [actuator.current_motor for actuator in self.base_feedback.actuators],
                #     'gripper_position': self.base_feedback.interconnect.gripper_feedback.motor[0].position,
                # })

                cyclic_count = cyclic_count + 1
                if cyclic_count >= len(self.trajectory):
                    break

        self.cyclic_running = False
        print(f'Cyclic count: {cyclic_count}')
        bins = np.r_[np.arange(MoveJController.CYCLIC_STEP_SIZE, 11 * MoveJController.CYCLIC_STEP_SIZE, MoveJController.CYCLIC_STEP_SIZE), np.inf]
        print('Cyclic step times:', np.histogram(step_times, bins=bins))

        # import pickle
        # output_path = f'joint-states.pkl'
        # with open(output_path, 'wb') as f:
        #     pickle.dump(data, f)
        # print(f'Data saved to {output_path}')

    def _stop_cyclic(self):
        if self.cyclic_running:
            self.kill_the_thread = True
            self.cyclic_thread.join()

        base_servo_mode = Base_pb2.ServoingModeInformation()
        base_servo_mode.servoing_mode = Base_pb2.SINGLE_LEVEL_SERVOING
        self.base.SetServoingMode(base_servo_mode)

class DeviceConnection:
    IP = '192.168.1.10'
    TCP_PORT = 10000
    UDP_PORT = 10001

    @staticmethod
    def createTcpConnection():
        return DeviceConnection(port=DeviceConnection.TCP_PORT)

    @staticmethod
    def createUdpConnection():
        return DeviceConnection(port=DeviceConnection.UDP_PORT)

    def __init__(self, ip=IP, port=TCP_PORT, credentials = ('admin', 'admin')):
        self.ip = ip
        self.port = port
        self.credentials = credentials
        self.session_manager = None
        self.transport = TCPTransport() if port == DeviceConnection.TCP_PORT else UDPTransport()
        self.router = RouterClient(self.transport, RouterClient.basicErrorCallback)

    def __enter__(self):
        self.transport.connect(self.ip, self.port)
        if self.credentials[0] != '':
            session_info = Session_pb2.CreateSessionInfo()
            session_info.username = self.credentials[0]
            session_info.password = self.credentials[1]
            session_info.session_inactivity_timeout = 10000   # (milliseconds)
            session_info.connection_inactivity_timeout = 2000 # (milliseconds)
            self.session_manager = SessionManager(self.router)
            print('Logging as', self.credentials[0], 'on device', self.ip)
            self.session_manager.CreateSession(session_info)
        return self.router

    def __exit__(self, *_):
        if self.session_manager is not None:
            router_options = RouterClientSendOptions()
            router_options.timeout_ms = 1000
            self.session_manager.CloseSession(router_options)
        self.transport.disconnect()

if __name__ == '__main__':
    arm = KinovaArm()
    try:
        arm.home()
        arm.retract()
    finally:
        arm.disconnect()
