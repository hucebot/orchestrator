#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import math
import numpy as np
from std_msgs.msg import String, Bool
from geometry_msgs.msg import PoseStamped
from std_srvs.srv import Trigger
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration

from rclpy.qos import (
    QoSProfile,
    QoSReliabilityPolicy,
    QoSHistoryPolicy,
    qos_profile_sensor_data,
)

import time


# ==========================================================
# 1. HELPER CLASSES
# ==========================================================
class PoseTracker:
    """Encapsulates all the Kalman Filter math and Quaternion logic."""

    def __init__(self):
        self.kf_x = np.zeros(6)
        self.kf_P = np.eye(6) * 1.0
        self.kf_Q = np.eye(6) * 0.01
        self.kf_R = np.eye(6) * 0.1
        self.kf_initialized = False
        self.pose_buffer = []
        self.stable_pose = False

    def reset(self):
        self.kf_initialized = False
        self.pose_buffer.clear()
        self.stable_pose = False

    def update(self, msg, pub, logger):
        pos = msg.pose.position
        q = msg.pose.orientation
        r, p, y = self._euler_from_quat(q.x, q.y, q.z, q.w)
        z = np.array([pos.x, pos.y, pos.z, r, p, y])

        if not self.kf_initialized:
            self.kf_x = z
            self.kf_initialized = True
        else:
            diff_r = (r - self.kf_x[3] + math.pi) % (2 * math.pi) - math.pi
            diff_p = (p - self.kf_x[4] + math.pi) % (2 * math.pi) - math.pi
            diff_y = (y - self.kf_x[5] + math.pi) % (2 * math.pi) - math.pi

            if abs(diff_r) > 0.78 or abs(diff_p) > 0.78 or abs(diff_y) > 0.78:
                logger.warn(
                    "Large orientation jump detected. Ignoring pose.",
                    throttle_duration_sec=1.0,
                )
            else:
                z[3] = self.kf_x[3] + diff_r
                z[4] = self.kf_x[4] + diff_p
                z[5] = self.kf_x[5] + diff_y

                self.kf_P = self.kf_P + self.kf_Q
                K = self.kf_P @ np.linalg.inv(self.kf_P + self.kf_R)
                self.kf_x = self.kf_x + K @ (z - self.kf_x)
                self.kf_P = (np.eye(6) - K) @ self.kf_P

        filtered_msg = PoseStamped()
        filtered_msg.header = msg.header
        (
            filtered_msg.pose.position.x,
            filtered_msg.pose.position.y,
            filtered_msg.pose.position.z,
        ) = (self.kf_x[0], self.kf_x[1], self.kf_x[2])
        qx, qy, qz, qw = self._quat_from_euler(self.kf_x[3], self.kf_x[4], self.kf_x[5])
        (
            filtered_msg.pose.orientation.x,
            filtered_msg.pose.orientation.y,
            filtered_msg.pose.orientation.z,
            filtered_msg.pose.orientation.w,
        ) = (qx, qy, qz, qw)

        pub.publish(filtered_msg)

        self.pose_buffer.append(self.kf_x.copy())
        if len(self.pose_buffer) > 10:
            self.pose_buffer.pop(0)

        self.stable_pose = True

    def _euler_from_quat(self, x, y, z, w):
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll = math.atan2(t0, t1)
        t2 = max(min(+2.0 * (w * y - z * x), 1.0), -1.0)
        pitch = math.asin(t2)
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw = math.atan2(t3, t4)
        return roll, pitch, yaw

    def _quat_from_euler(self, roll, pitch, yaw):
        cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
        cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
        cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
        return (
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
            cr * cp * cy + sr * sp * sy,
        )


# ==========================================================
# 2. THE STATE DESIGN PATTERN
# ==========================================================
class BaseState:
    @property
    def name(self):
        return self.__class__.__name__

    def execute(self, ctx):
        return self


class StartPreNavAndMeshState(BaseState):
    def execute(self, ctx):
        t = ctx.current_task
        ctx.pub_teleop_mode.publish(String(data="replay"))

        # 1. Trigger Pre-Nav
        pre_nav_str = f"pre_{t['nav_goal']}"
        ctx.pub_nav.publish(String(data=pre_nav_str))

        # 2. Request Target Mesh (Background loading)
        if t["target_obj"] != "none":
            ctx.pub_mesh.publish(String(data="mesh_update_" + t["target_obj"]))

        return WaitPreNavState()


class WaitPreNavState(BaseState):
    def execute(self, ctx):
        if ctx.nav_done:
            ctx.nav_done = False
            return StartHomeAndNavState()
        return self


class StartHomeAndNavState(BaseState):
    def execute(self, ctx):
        t = ctx.current_task

        # 1. Trigger Final Nav
        ctx.pub_nav.publish(String(data=t["nav_goal"]))

        # 2. Trigger Homing
        srv_name = f"/home_position/{t['nav_goal']}"
        if srv_name not in ctx.home_clients:
            ctx.home_clients[srv_name] = ctx.create_client(Trigger, srv_name)

        client = ctx.home_clients[srv_name]
        if client.service_is_ready():
            req = Trigger.Request()
            future = client.call_async(req)
            future.add_done_callback(ctx._cb_home_future)
        else:
            ctx.get_logger().warn(
                f"Home service {srv_name} not ready! Marking failed to retry...",
                throttle_duration_sec=2.0,
            )
            ctx.home_trigger_failed = True

        return WaitHomeAndNavState()


class WaitHomeAndNavState(BaseState):
    def execute(self, ctx):
        t = ctx.current_task

        # If homing service rejected/failed, retry ONLY homing
        if ctx.home_trigger_failed:
            ctx.get_logger().warn(
                "Home config failed! Retrying homing...", throttle_duration_sec=2.0
            )
            ctx.home_trigger_failed = False

            srv_name = f"/home_position/{t['nav_goal']}"
            client = ctx.home_clients.get(srv_name)
            if client and client.service_is_ready():
                req = Trigger.Request()
                future = client.call_async(req)
                future.add_done_callback(ctx._cb_home_future)

        # Wait for BOTH physical motions to finish
        if ctx.nav_done and ctx.home_cfg_done:
            ctx.pub_resume.publish(Bool(data=True))
            ctx.nav_done = False
            ctx.home_cfg_done = False

            # Smart Routing: Dock, or skip to mesh verification
            if t["do_dock"]:
                ctx.tracker.reset()
                return WaitDockPoseState()
            else:
                return VerifyMeshState()

        return self


class WaitDockPoseState(BaseState):
    def execute(self, ctx):
        if ctx.tracker.stable_pose:
            ctx.tracker.stable_pose = False
            return ExecuteDockState()
        return self


class ExecuteDockState(BaseState):
    def execute(self, ctx):
        ctx.pub_dock.publish(Bool(data=True))
        return WaitDockState()


class WaitDockState(BaseState):
    def execute(self, ctx):
        if ctx.dock_done:
            ctx.dock_done = False
            return VerifyMeshState()
        return self


class VerifyMeshState(BaseState):
    def execute(self, ctx):
        t = ctx.current_task

        # If no object manipulation is needed, go straight to skill/done
        if t["target_obj"] == "none":
            return ExecuteSkillState()

        # Check if the mesh has finished loading in the background
        if ctx.mesh_loaded:
            ctx.mesh_loaded = False
            ctx.tracker.reset()
            time.sleep(4.0)
            return WaitTargetPoseState()

        return self


class WaitTargetPoseState(BaseState):
    def execute(self, ctx):
        if ctx.tracker.stable_pose:
            ctx.tracker.stable_pose = False
            return ExecuteSkillState()
        return self


class ExecuteSkillState(BaseState):
    def execute(self, ctx):
        t = ctx.current_task
        if t["skill"] != "none":
            ctx.pub_skill.publish(String(data=t["skill"]))
            return WaitSkillState()
        else:
            ctx._send_gripper("right", 0.0)
            return TaskDoneState()


class WaitSkillState(BaseState):
    def execute(self, ctx):
        if ctx.skill_done:
            ctx.skill_done = False
            return TaskDoneState()
        return self


class TaskDoneState(BaseState):
    def execute(self, ctx):
        ctx.get_logger().info(f"Task {ctx.current_task['task']} Complete.")
        if ctx.tasks:
            ctx.current_task = ctx.tasks.pop(0)
            return StartPreNavAndMeshState()
        else:
            ctx.get_logger().info("All tasks in sequence completed successfully.")
            return DoneState()


class DoneState(BaseState):
    def execute(self, ctx):
        return self


# ==========================================================
# 3. THE ORCHESTRATOR NODE (Context)
# ==========================================================
class PickPlaceOrchestrator(Node):
    def __init__(self):
        super().__init__("pick_place_orchestrator")

        # Cleaned up task queue: Removed dock_obj, added do_dock, added all Place motions
        self.tasks = [
            {
                "task": "open_fridge",
                "nav_goal": "fridge_open",
                "do_dock": True,
                "target_obj": "none",
                "skill": "open_fridge",
            },
            # MILK
            {
                "task": "pick_milk",
                "nav_goal": "table",
                "do_dock": True,
                "target_obj": "milk",
                "skill": "pick_milk",
            },
            {
                "task": "place_milk",
                "nav_goal": "fridge_door",
                "do_dock": True,
                "target_obj": "none",
                "skill": "place_milk",
            },
            {
                "task": "pick_solevita",
                "nav_goal": "table",
                "do_dock": True,
                "target_obj": "solevita",
                "skill": "pick_solevita",
            },
            {
                "task": "place_solevita",
                "nav_goal": "fridge_door",
                "do_dock": True,
                "target_obj": "none",
                "skill": "place_solevita",
            },
            # BANANA
            {
                "task": "pick_orange",
                "nav_goal": "table",
                "do_dock": True,
                "target_obj": "orange",
                "skill": "pick_orange",
            },
            {
                "task": "place_orange",
                "nav_goal": "fridge_place",
                "do_dock": True,
                "target_obj": "none",
                "skill": "place_orange",
            },
            # BAGUETTE
            {
                "task": "pick_banana",
                "nav_goal": "table",
                "do_dock": True,
                "target_obj": "banana",
                "skill": "pick_banana",
            },
            {
                "task": "place_banana",
                "nav_goal": "fridge_place",
                "do_dock": True,
                "target_obj": "none",
                "skill": "place_banana",
            },
            {
                "task": "close_fridge",
                "nav_goal": "fridge_close",
                "do_dock": True,
                "target_obj": "none",
                "skill": "close_fridge",
            },
            {
                "task": "pick_baguette",
                "nav_goal": "table",
                "do_dock": True,
                "target_obj": "baguette",
                "skill": "pick_baguette",
            },
            {
                "task": "place_baguette",
                "nav_goal": "sink",
                "do_dock": True,
                "target_obj": "none",
                "skill": "place_baguette",
            },
        ]

        # Map the Navigation Goal directly to the required AprilTag ID for docking
        self.nav_to_tag = {
            "table": "9",
            "fridge_place": "37",
            "fridge_door": "0",
            "fridge_open": "38",
            "fridge_close": "0",
            "sink": "7",
        }

        self.current_task = self.tasks.pop(0)
        self.current_state = StartPreNavAndMeshState()

        # Flags
        self.last_published_task = ""
        self.nav_done = False
        self.home_cfg_done = False
        self.home_trigger_failed = False
        self.mesh_loaded = False
        self.dock_done = False
        self.skill_done = False

        self.tracker = PoseTracker()
        self.home_clients = {}

        self._setup_ros()

    def _setup_ros(self):
        qos_cmd = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
        )
        qos_state = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # Publishers
        self.pub_nav = self.create_publisher(String, "/nav/goal/target", qos_cmd)
        self.pub_mesh = self.create_publisher(
            String, "/orchestrator/foundation_pose/target_object", qos_cmd
        )
        self.pub_skill = self.create_publisher(
            String, "/inference/execute_task", qos_cmd
        )
        self.pub_ui_task = self.create_publisher(
            String, "/orchestrator/ui/current_task", qos_state
        )
        self.pub_ui_state = self.create_publisher(
            String, "/orchestrator/ui/current_state", qos_state
        )
        self.pub_resume = self.create_publisher(Bool, "/ros_control_bridge/restart", 10)
        self.pub_teleop_mode = self.create_publisher(
            String, "/streamdeck/teleop_mode", 10
        )
        self.pub_gripper_right = self.create_publisher(
            JointTrajectory, "/gripper_right_controller/joint_trajectory", 10
        )

        self.pub_dock = self.create_publisher(Bool, "/dock/goal/start", qos_state)
        self.pub_target_pose = self.create_publisher(
            PoseStamped, "/dock/goal/target", qos_profile_sensor_data
        )

        # Static Subscribers
        self.create_subscription(Bool, "/nav/goal/done", self._cb_nav, qos_state)
        self.create_subscription(
            Bool, "/foundation_pose/mesh_status", self._cb_mesh, qos_state
        )
        self.create_subscription(Bool, "/inference/status", self._cb_skill, qos_state)
        self.create_subscription(
            Bool, "/cartesian_interface/home_done", self._cb_home, qos_state
        )
        self.create_subscription(Bool, "/dock/goal/done", self._cb_dock, qos_state)
        self.create_subscription(
            Bool, "/orchestrator/manual_override", self._cb_manual_override, 10
        )

        # Subscribe to all AprilTags defined in the configuration mapping
        unique_tags = set(self.nav_to_tag.values())
        for tag_id in unique_tags:
            self.create_subscription(
                PoseStamped,
                f"/apriltag_pose/pose_tag_{tag_id}",
                lambda msg, tid=tag_id: self._cb_pose(
                    msg, source_type="dock", tag_id=tid
                ),
                qos_profile_sensor_data,
            )

        # Target Pose Subscriber (Foundation Pose)
        self.create_subscription(
            PoseStamped,
            "/foundation_pose/object_pose",
            lambda msg: self._cb_pose(msg, source_type="target"),
            qos_profile_sensor_data,
        )

        self.timer = self.create_timer(0.1, self.tick)

    # --- Simple Callbacks ---
    def _cb_nav(self, msg):
        if msg.data and not self.nav_done:
            self.nav_done = True

    def _cb_home(self, msg):
        if msg.data:
            self.home_cfg_done = True
        else:
            self.home_trigger_failed = True

    def _cb_mesh(self, msg):
        if msg.data and not self.mesh_loaded:
            self.mesh_loaded = True

    def _cb_dock(self, msg):
        if msg.data and not self.dock_done:
            self.dock_done = True

    def _cb_skill(self, msg):
        if msg.data and not self.skill_done:
            self.skill_done = True

    # Maps each Wait/poll state to the flag(s) it is blocked on, so a manual
    # override can pretend that flag's normal source (a sensor/service
    # callback) just fired.
    _WAIT_STATE_OVERRIDES = {
        "WaitPreNavState": lambda ctx: setattr(ctx, "nav_done", True),
        "WaitHomeAndNavState": lambda ctx: (
            setattr(ctx, "nav_done", True),
            setattr(ctx, "home_cfg_done", True),
            setattr(ctx, "home_trigger_failed", False),
        ),
        "WaitDockPoseState": lambda ctx: setattr(ctx.tracker, "stable_pose", True),
        "WaitDockState": lambda ctx: setattr(ctx, "dock_done", True),
        "VerifyMeshState": lambda ctx: setattr(ctx, "mesh_loaded", True),
        "WaitTargetPoseState": lambda ctx: setattr(ctx.tracker, "stable_pose", True),
        "WaitSkillState": lambda ctx: setattr(ctx, "skill_done", True),
    }

    def _cb_manual_override(self, msg):
        if not msg.data:
            return

        state_name = self.current_state.name
        override = self._WAIT_STATE_OVERRIDES.get(state_name)
        if override is None:
            self.get_logger().warn(
                f"Manual override received, but {state_name} isn't waiting on "
                "anything right now. Ignoring."
            )
            return

        override(self)
        self.get_logger().warn(f"MANUAL OVERRIDE: forcing completion of {state_name}.")

    def _cb_home_future(self, future):
        try:
            response = future.result()
            self.home_trigger_failed = not response.success
        except Exception:
            self.home_trigger_failed = True

    def _cb_pose(self, msg, source_type, tag_id=None):
        """Routes the pose to the tracker depending on the active state and tag mapping."""
        state_name = self.current_state.name

        if source_type == "dock":
            if state_name not in [
                "WaitDockPoseState",
                "ExecuteDockState",
                "WaitDockState",
            ]:
                return
            # Ensure the tag ID matches the required tag for the current nav location
            expected_tag = self.nav_to_tag.get(self.current_task["nav_goal"])
            if expected_tag is None or tag_id != expected_tag:
                return

        elif source_type == "target":
            if state_name != "WaitTargetPoseState":
                return

        # Pass through Kalman Filter and publish
        self.tracker.update(msg, self.pub_target_pose, self.get_logger())

    def _send_gripper(self, side: str, pos: float) -> None:
        pub = self.pub_gripper_right
        traj = JointTrajectory()
        traj.joint_names = [f"gripper_{side}_finger_joint"]
        p = JointTrajectoryPoint()
        p.positions = [pos]
        p.time_from_start = Duration(sec=0, nanosec=int(2e8))
        traj.points = [p]
        pub.publish(traj)

    def tick(self):
        if not self.tasks and self.current_state.name == "DoneState":
            return

        t = self.current_task
        if self.last_published_task != t["task"]:
            self.get_logger().info(
                f"\n================ TASK: {t['task'].upper()} ================"
            )
            self.pub_ui_task.publish(String(data=t["task"]))
            self.last_published_task = t["task"]

        # Execute State and capture transition
        next_state = self.current_state.execute(self)

        if next_state.name != self.current_state.name:
            self.get_logger().info(f"--> Transitioned to: {next_state.name}")
            self.pub_ui_state.publish(String(data=next_state.name))
            self.current_state = next_state


def main(args=None):
    rclpy.init(args=args)
    node = PickPlaceOrchestrator()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
