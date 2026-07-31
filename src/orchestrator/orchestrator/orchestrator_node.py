import rclpy
from rclpy.node import Node
from enum import Enum
import math
import numpy as np
from std_msgs.msg import String, Bool
from geometry_msgs.msg import PoseStamped
from std_srvs.srv import Trigger
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration

# Import QoS modules
from rclpy.qos import (
    QoSProfile,
    QoSReliabilityPolicy,
    QoSHistoryPolicy,
    qos_profile_sensor_data,
)


class State(Enum):
    S1_START_NAV_AND_MESH = 1
    S2_WAIT_NAV = 2
    S3_SET_HOME_CFG = 3
    S4_WAIT_HOME_CFG = 4
    S5_WAIT_TARGET_MESH = 5  # Renamed from DOCK_MESH
    S6_WAIT_DOCK_POSE = 6
    S7_EXECUTE_DOCK = 7
    S8_WAIT_DOCK = 8
    # S9 and S10 were removed as they are obsolete
    S11_WAIT_TARGET_POSE = 11
    S12_EXECUTE_SKILL = 12
    S13_WAIT_SKILL = 13
    S14_TASK_DONE = 14
    DONE = 15


class PickPlaceOrchestrator(Node):
    def __init__(self):
        super().__init__("pick_place_orchestrator")

        # 1. THE TASK QUEUE
        self.tasks = [
            {"task": "pick_milk",          "nav_goal": "table", "dock_obj": "pan", "target_obj": "milk",     "skill": "pick_milk"},
            {"task": "go_home",            "nav_goal": "home",  "dock_obj": "none","target_obj": "none",     "skill": "none"},
            {"task": "pick_banana",        "nav_goal": "table", "dock_obj": "pan", "target_obj": "banana",   "skill": "pick_banana"},
            {"task": "go_home",            "nav_goal": "home",  "dock_obj": "none","target_obj": "none",     "skill": "none"},
            {"task": "pick_baguette",      "nav_goal": "table", "dock_obj": "pan", "target_obj": "baguette", "skill": "pick_baguette"},
            {"task": "go_home",            "nav_goal": "home",  "dock_obj": "none","target_obj": "none",     "skill": "none"},
            {"task": "pick_apple",         "nav_goal": "table", "dock_obj": "pan", "target_obj": "redapple", "skill": "pick_apple"},
            {"task": "go_home",            "nav_goal": "home",  "dock_obj": "none","target_obj": "none",     "skill": "none"},
        ]

        self.current_task = self.tasks.pop(0)
        self.state = State.S1_START_NAV_AND_MESH

        # UI/Logging Tracking
        self.last_published_task = ""
        self.last_published_state = None

        # Flags
        self.nav_done = False
        self.home_cfg_done = False
        self.home_trigger_failed = False
        self.mesh_loaded = False
        self.dock_done = False
        self.skill_done = False
        self.stable_pose = False

        # Kalman Filter States
        self.kf_x = np.zeros(6)
        self.kf_P = np.eye(6) * 1.0
        self.kf_Q = np.eye(6) * 0.01
        self.kf_R = np.eye(6) * 0.1
        self.kf_initialized = False
        self.pose_buffer = []

        # Service Client Cache (Stores dynamic home clients)
        self.home_clients = {}

        # ==========================================
        # QoS PROFILES
        # ==========================================
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

        qos_pose = qos_profile_sensor_data

        # ==========================================
        # PUBLISHERS
        # ==========================================
        self.pub_nav = self.create_publisher(String, "/nav/goal/target", qos_cmd)
        self.pub_mesh = self.create_publisher(String, "/orchestrator/foundation_pose/target_object", qos_cmd)
        self.pub_skill = self.create_publisher(String, "/inference/execute_task", qos_cmd)
        self.pub_toggle_fp = self.create_publisher(Bool, "/orchestrator/foundation_pose/toggle", qos_state)
        self.pub_dock = self.create_publisher(Bool, "/dock/goal/start", qos_state)
        self.pub_ui_task = self.create_publisher(String, "/orchestrator/ui/current_task", qos_state)
        self.pub_ui_state = self.create_publisher(String, "/orchestrator/ui/current_state", qos_state)
        self.pub_target_pose = self.create_publisher(PoseStamped, "/dock/goal/target", qos_pose)
        self.pub_resume = self.create_publisher(Bool, "/ros_control_bridge/restart", 10)
        self.pub_teleop_mode = self.create_publisher(String, "/streamdeck/teleop_mode", 10)
        self.pub_gripper_right = self.create_publisher(JointTrajectory, "/gripper_right_controller/joint_trajectory", 10)

        # ==========================================
        # SUBSCRIBERS
        # ==========================================
        self.create_subscription(Bool, "/nav/goal/done", self._cb_nav, qos_state)
        self.create_subscription(Bool, "/foundation_pose/mesh_status", self._cb_mesh, qos_state)
        self.create_subscription(Bool, "/dock/goal/done", self._cb_dock, qos_state)
        self.create_subscription(Bool, "/inference/status", self._cb_skill, qos_state)
        self.create_subscription(Bool, "/cartesian_interface/home_done", self._cb_home, qos_state)

        # Poses
        self.create_subscription(
            PoseStamped,
            "/foundation_pose/object_pose",
            lambda msg: self._cb_pose(msg, is_tag=False),
            qos_pose,
        )
        self.create_subscription(
            PoseStamped,
            "/apriltag_pose/pose_tag_38",  # Explicitly using tag 38
            lambda msg: self._cb_pose(msg, is_tag=True),
            qos_pose,
        )

        self.timer = self.create_timer(0.1, self.tick)
        self.get_logger().info("Orchestrator Initialized and Running.")

    def _cb_nav(self, msg):
        if msg.data and not self.nav_done:
            self.get_logger().info("Callback: Navigation Done")
        self.nav_done = msg.data

    def _cb_home(self, msg):
        if msg.data:
            if not self.home_cfg_done:
                self.get_logger().info("Callback: Home Config Physically Complete")
            self.home_cfg_done = True
        else:
            self.get_logger().error("Callback: Home Config FAILED completely! Bouncing FSM to retry...")
            self.home_trigger_failed = True

    def _cb_home_future(self, future):
        try:
            response = future.result()
            if response.success:
                self.get_logger().info("Home service request accepted. Waiting for physical motion...")
                self.home_trigger_failed = False
            else:
                self.get_logger().warn(f"Home service returned failure: {response.message}")
                self.home_trigger_failed = True
        except Exception as e:
            self.get_logger().error(f"Home service call failed: {e}")
            self.home_trigger_failed = True

    def _cb_mesh(self, msg):
        if msg.data and not self.mesh_loaded:
            self.get_logger().info("Callback: Mesh Loaded")
        self.mesh_loaded = msg.data

    def _cb_dock(self, msg):
        if msg.data and not self.dock_done:
            self.get_logger().info("Callback: Docking Complete")
        self.dock_done = msg.data

    def _cb_skill(self, msg):
        if msg.data and not self.skill_done:
            self.get_logger().info("Callback: Skill Execution Finished")
        self.skill_done = msg.data

    def _cb_pose(self, msg, is_tag):
        if self.state not in [
            State.S6_WAIT_DOCK_POSE,
            State.S11_WAIT_TARGET_POSE,
            State.S7_EXECUTE_DOCK,
            State.S8_WAIT_DOCK,
        ]:
            return

        # STRICT ROUTING: Only accept tag for docking, only accept foundation pose for targets
        if self.state in [State.S6_WAIT_DOCK_POSE, State.S7_EXECUTE_DOCK, State.S8_WAIT_DOCK] and not is_tag:
            return
        if self.state == State.S11_WAIT_TARGET_POSE and is_tag:
            return

        pos = msg.pose.position
        q = msg.pose.orientation
        r, p, y = self.euler_from_quat(q.x, q.y, q.z, q.w)
        z = np.array([pos.x, pos.y, pos.z, r, p, y])

        if not self.kf_initialized:
            self.kf_x = z
            self.kf_initialized = True
        else:
            diff_r = (r - self.kf_x[3] + math.pi) % (2 * math.pi) - math.pi
            diff_p = (p - self.kf_x[4] + math.pi) % (2 * math.pi) - math.pi
            diff_y = (y - self.kf_x[5] + math.pi) % (2 * math.pi) - math.pi

            if abs(diff_r) > 0.78 or abs(diff_p) > 0.78 or abs(diff_y) > 0.78:
                self.get_logger().warn("Large orientation jump detected. Ignoring pose.", throttle_duration_sec=1.0)
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
        (filtered_msg.pose.position.x, filtered_msg.pose.position.y, filtered_msg.pose.position.z) = (self.kf_x[0], self.kf_x[1], self.kf_x[2])
        qx, qy, qz, qw = self.quat_from_euler(self.kf_x[3], self.kf_x[4], self.kf_x[5])
        (filtered_msg.pose.orientation.x, filtered_msg.pose.orientation.y, filtered_msg.pose.orientation.z, filtered_msg.pose.orientation.w) = (qx, qy, qz, qw)
        self.pub_target_pose.publish(filtered_msg)

        self.pose_buffer.append(self.kf_x.copy())
        if len(self.pose_buffer) > 10:
            self.pose_buffer.pop(0)

        self.stable_pose = True

    def reset_pose_tracking(self):
        self.kf_initialized = False
        self.pose_buffer.clear()
        self.stable_pose = False

    def _send_gripper(self, side: str, pos: float) -> None:
        pub = self.pub_gripper_left if side == "left" else self.pub_gripper_right
        traj = JointTrajectory()
        traj.joint_names = [f"gripper_{side}_finger_joint"]
        p = JointTrajectoryPoint()
        p.positions = [pos]
        p.time_from_start = Duration(sec=0, nanosec=int(2e8))
        traj.points = [p]
        pub.publish(traj)

    def tick(self):
        if not self.tasks and self.state == State.DONE:
            return

        t = self.current_task

        if self.last_published_task != t["task"]:
            self.get_logger().info(f"\n================ STARTING TASK: {t['task'].upper()} ================")
            self.pub_ui_task.publish(String(data=t["task"]))
            self.last_published_task = t["task"]

        if self.last_published_state != self.state:
            self.get_logger().info(f"--> Transitioned to: {self.state.name}")
            self.pub_ui_state.publish(String(data=self.state.name))
            self.last_published_state = self.state

        # ==================================================
        # S1: Start Navigation and Load TARGET Mesh
        # ==================================================
        if self.state == State.S1_START_NAV_AND_MESH:
            self.pub_teleop_mode.publish(String(data="replay"))
            self.pub_nav.publish(String(data=t["nav_goal"]))

            # Request the TARGET object mesh immediately so it loads during nav/homing
            if t["target_obj"] != "none":
                mesh_update_str = "mesh_update_" + t["target_obj"]
                self.pub_mesh.publish(String(data=mesh_update_str))

            self.state = State.S2_WAIT_NAV

        elif self.state == State.S2_WAIT_NAV:
            if self.nav_done:
                self.nav_done = False
                self.state = State.S3_SET_HOME_CFG

        elif self.state == State.S3_SET_HOME_CFG:
            home_name = f"{t['nav_goal']}"
            srv_name = f"/home_position/{home_name}"

            if srv_name not in self.home_clients:
                self.home_clients[srv_name] = self.create_client(Trigger, srv_name)

            client = self.home_clients[srv_name]
            if client.service_is_ready():
                req = Trigger.Request()
                future = client.call_async(req)
                future.add_done_callback(self._cb_home_future)
                self.state = State.S4_WAIT_HOME_CFG
            else:
                self.get_logger().warn(f"Waiting for home service: {srv_name}...", throttle_duration_sec=2.0)

        elif self.state == State.S4_WAIT_HOME_CFG:
            if self.home_trigger_failed:
                self.get_logger().warn("Home config failed! Bouncing back to S3...", throttle_duration_sec=2.0)
                self.home_trigger_failed = False
                self.state = State.S3_SET_HOME_CFG
            elif self.home_cfg_done:
                self.pub_resume.publish(Bool(data=True))
                self.home_cfg_done = False
                # Go to mesh wait, or directly to dock pose if no target mesh is needed
                self.state = State.S5_WAIT_TARGET_MESH if t["target_obj"] != "none" else State.S6_WAIT_DOCK_POSE

        # ==================================================
        # S5: Wait for Target Mesh to Finish Background Loading
        # ==================================================
        elif self.state == State.S5_WAIT_TARGET_MESH:
            if self.mesh_loaded:
                self.mesh_loaded = False
                self.reset_pose_tracking()
                self.state = State.S6_WAIT_DOCK_POSE

        # ==================================================
        # S6 to S8: Docking (Using AprilTag)
        # ==================================================
        elif self.state == State.S6_WAIT_DOCK_POSE:
            if self.stable_pose:
                self.stable_pose = False
                self.state = State.S7_EXECUTE_DOCK

        elif self.state == State.S7_EXECUTE_DOCK:
            self.pub_dock.publish(Bool(data=True))
            self.state = State.S8_WAIT_DOCK

        elif self.state == State.S8_WAIT_DOCK:
            if self.dock_done:
                self.dock_done = False
                self.reset_pose_tracking()
                # Skip S9/S10 because the target mesh was already loaded in S1!
                self.state = State.S11_WAIT_TARGET_POSE if t["target_obj"] != "none" else State.S12_EXECUTE_SKILL

        # ==================================================
        # S11+: Target Pose and Skill Execution
        # ==================================================
        elif self.state == State.S11_WAIT_TARGET_POSE:
            if self.stable_pose:
                self.stable_pose = False
                self.state = State.S12_EXECUTE_SKILL

        elif self.state == State.S12_EXECUTE_SKILL:
            if t["skill"] != "none":
                self.pub_skill.publish(String(data=t["skill"]))
                self.state = State.S13_WAIT_SKILL
            else:
                self._send_gripper("right", 0.0)
                self.state = State.S14_TASK_DONE

        elif self.state == State.S13_WAIT_SKILL:
            if self.skill_done:
                self.skill_done = False
                self.state = State.S14_TASK_DONE

        elif self.state == State.S14_TASK_DONE:
            self.get_logger().info(f"Task {t['task']} Complete.")
            if self.tasks:
                self.current_task = self.tasks.pop(0)
                self.state = State.S1_START_NAV_AND_MESH
            else:
                self.get_logger().info("All tasks in sequence completed successfully.")
                self.state = State.DONE

    # --- Math Helpers ---
    def euler_from_quat(self, x, y, z, w):
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll = math.atan2(t0, t1)
        t2 = +2.0 * (w * y - z * x)
        t2 = max(min(t2, 1.0), -1.0)
        pitch = math.asin(t2)
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw = math.atan2(t3, t4)
        return roll, pitch, yaw

    def quat_from_euler(self, roll, pitch, yaw):
        cr = math.cos(roll * 0.5)
        sr = math.sin(roll * 0.5)
        cp = math.cos(pitch * 0.5)
        sp = math.sin(pitch * 0.5)
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        return (
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
            cr * cp * cy + sr * sp * sy,
        )

def main(args=None):
    rclpy.init(args=args)
    node = PickPlaceOrchestrator()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == "__main__":
    main()