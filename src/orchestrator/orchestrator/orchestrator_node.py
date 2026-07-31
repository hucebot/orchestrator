import rclpy
from rclpy.node import Node
from enum import Enum
import math
import numpy as np
from std_msgs.msg import String, Bool
from geometry_msgs.msg import PoseStamped
from std_srvs.srv import Trigger

# Import QoS modules
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, qos_profile_sensor_data

class State(Enum):
    S1_START_NAV_AND_MESH = 1
    S2_WAIT_NAV = 2
    S3_SET_HOME_CFG = 3
    S4_WAIT_HOME_CFG = 4
    S5_WAIT_DOCK_MESH = 5
    S6_WAIT_DOCK_POSE = 6
    S7_EXECUTE_DOCK = 7
    S8_WAIT_DOCK = 8
    S9_LOAD_TARGET_MESH = 9
    S10_WAIT_TARGET_MESH = 10
    S11_WAIT_TARGET_POSE = 11
    S12_EXECUTE_SKILL = 12
    S13_WAIT_SKILL = 13
    S14_TASK_DONE = 14
    DONE = 15

class PickPlaceOrchestrator(Node):
    def __init__(self):
        super().__init__('pick_place_orchestrator')

        # 1. THE TASK QUEUE


        # This tests all the picking motions with navigation and docking (Testing the pipeline up until we gradually have everything working)
        self.tasks = [
            {"task": "pick_milk",          "nav_goal": "table",         "dock_obj": "pan",        "target_obj": "milk",   "skill": "pick_milk"},
            {"task": "go_home",            "nav_goal": "home",          "dock_obj": "none",       "target_obj": "none",   "skill": "none"},
            # {"task": "pick_juice",         "nav_goal": "table",         "dock_obj": "pan",        "target_obj": "juice",  "skill": "pick_juice"},
            # {"task": "go_home",            "nav_goal": "home",          "dock_obj": "none",       "target_obj": "none",   "skill": "none"},
            {"task": "pick_banana",        "nav_goal": "table",         "dock_obj": "pan",        "target_obj": "banana",  "skill": "pick_banana"},
            {"task": "go_home",            "nav_goal": "home",          "dock_obj": "none",       "target_obj": "none",  "skill": "none"},
            {"task": "pick_baguette",        "nav_goal": "table",         "dock_obj": "pan",        "target_obj": "baguette",  "skill": "pick_baguette"},
            {"task": "go_home",            "nav_goal": "home",          "dock_obj": "none",       "target_obj":"none",   "skill":"none"},
            {"task": "pick_apple",       "nav_goal": "table",         "dock_obj": "pan",        "target_obj": "redapple",  "skill": "pick_apple"},
            {"task": "go_home",            "nav_goal": "home",          "dock_obj": "none",       "target_obj":"none",   "skill":"none"},
        ]
        # self.tasks = [
        #         {"task": "open_fridge",        "nav_goal": "fridge_open",   "dock_obj": "tag_1",      "target_obj": "tag_1",  "skill": "open_fridge"},
        #         {"task": "pick_milk",          "nav_goal": "table",         "dock_obj": "pot",        "target_obj": "milk",   "skill": "grasp_milk"},
        #         {"task": "place_milk_door",    "nav_goal": "fridge_door",   "dock_obj": "yellow_mug", "target_obj": "milk",   "skill": "place_item"},
        #         {"task": "pick_juice",         "nav_goal": "table",         "dock_obj": "pot",        "target_obj": "juice",  "skill": "grasp_juice"},
        #         {"task": "place_juice_door",   "nav_goal": "fridge_door",   "dock_obj": "yellow_mug", "target_obj": "juice",  "skill": "place_item"},
        #         {"task": "pick_apple_1",       "nav_goal": "table",         "dock_obj": "pot",        "target_obj": "apple",  "skill": "grasp_apple"},
        #         {"task": "place_apple_inside", "nav_goal": "fridge_inside", "dock_obj": "mustard",    "target_obj": "mustard",  "skill": "place_item"},
        #         {"task": "pick_banana",        "nav_goal": "table",         "dock_obj": "pot",        "target_obj": "banana", "skill": "grasp_banana"},
        #         {"task": "place_banana_inside","nav_goal": "fridge_inside", "dock_obj": "mustard",    "target_obj": "banana", "skill": "place_item"},
        #         {"task": "pick_apple_2",       "nav_goal": "table",         "dock_obj": "pot",        "target_obj": "apple",  "skill": "grasp_apple"},
        #         {"task": "place_apple_inside", "nav_goal": "fridge_inside", "dock_obj": "mustard",    "target_obj": "apple",  "skill": "place_item"},
        #         {"task": "close_fridge",       "nav_goal": "fridge_close",  "dock_obj": "yellow_mug", "target_obj": "none",   "skill": "close_fridge"},
        #         {"task": "go_home",            "nav_goal": "home",          "dock_obj": "none",       "target_obj": "none",   "skill": "none"}
        #     ]

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

        # Commands (Strings like nav goals, skills): RELIABLE, Queue 10.
        # Why: We CANNOT drop a command if the network blips, so it must be reliable.
        # A small buffer (10) ensures if nodes startup slightly out of sync, the command isn't lost.
        qos_cmd = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10
        )

        # State/Toggles (Booleans): RELIABLE, Queue 1.
        # Why: We need guaranteed delivery so we don't miss a transition, but we ONLY
        # care about the absolute freshest state. If we toggle fast, old buffered toggles cause chaos.
        qos_state = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )

        # Sensor Data (Poses): BEST_EFFORT, Queue 1. (Using ROS2's built-in preset)
        # Why: If the network lags, do NOT buffer old poses. We want the robot to move
        # to where the object is NOW, not where it was 2 seconds ago. Drops old frames.
        qos_pose = qos_profile_sensor_data

        # ==========================================
        # PUBLISHERS
        # ==========================================
        self.pub_nav = self.create_publisher(String, '/nav/goal/target', qos_cmd)
        self.pub_mesh = self.create_publisher(String, '/orchestrator/foundation_pose/target_object', qos_cmd)
        self.pub_skill = self.create_publisher(String, '/inference/execute_task', qos_cmd)

        self.pub_toggle_fp = self.create_publisher(Bool, '/orchestrator/foundation_pose/toggle', qos_state)
        self.pub_dock = self.create_publisher(Bool, '/dock/goal/start', qos_state)
        self.pub_ui_task = self.create_publisher(String, '/orchestrator/ui/current_task', qos_state)
        self.pub_ui_state = self.create_publisher(String, '/orchestrator/ui/current_state', qos_state)

        self.pub_target_pose = self.create_publisher(PoseStamped, '/dock/goal/target', qos_pose)

        # ==========================================
        # SUBSCRIBERS
        # ==========================================
        self.create_subscription(Bool, '/nav/goal/done', self._cb_nav, qos_state)
        self.create_subscription(Bool, '/foundation_pose/mesh_status', self._cb_mesh, qos_state)
        self.create_subscription(Bool, '/dock/goal/done', self._cb_dock, qos_state)
        self.create_subscription(Bool, '/inference/status', self._cb_skill, qos_state)
        self.create_subscription(
            Bool, '/cartesian_interface/home_done', self._cb_home, qos_state
        )
        self.create_subscription(PoseStamped, '/foundation_pose/object_pose', lambda msg: self._cb_pose(msg, is_tag=False), qos_pose)
        self.create_subscription(PoseStamped, '/apriltag_pose/pose', lambda msg: self._cb_pose(msg, is_tag=True), qos_pose)

        self.timer = self.create_timer(0.1, self.tick)
        self.get_logger().info("Orchestrator Initialized and Running.")

    def _cb_nav(self, msg):
        if msg.data and not self.nav_done: self.get_logger().info("Callback: Navigation Done")
        self.nav_done = msg.data

    def _cb_home(self, msg):
        if msg.data:
            if not self.home_cfg_done:
                self.get_logger().info("Callback: Home Config Physically Complete")
            self.home_cfg_done = True
        else:
            self.get_logger().error("Callback: Home Config FAILED completely! Bouncing FSM to retry...")
            # We reuse the future failure mechanism we built earlier to bounce back to S3
            self.home_trigger_failed = True

    # Callback for async Home Service Call
    def _cb_home_future(self, future):
        try:
            response = future.result()
            if response.success:
                self.get_logger().info("Home service request accepted by Cartesian Interface. Waiting for physical motion...")
                self.home_trigger_failed = False
            else:
                self.get_logger().warn(f"Home service returned failure: {response.message}")
                self.home_trigger_failed = True
        except Exception as e:
            self.get_logger().error(f"Home service call failed: {e}")
            self.home_trigger_failed = True

    def _cb_mesh(self, msg):
        if msg.data and not self.mesh_loaded: self.get_logger().info("Callback: Mesh Loaded")
        self.mesh_loaded = msg.data

    def _cb_dock(self, msg):
        if msg.data and not self.dock_done: self.get_logger().info("Callback: Docking Complete")
        self.dock_done = msg.data

    def _cb_skill(self, msg):
        if msg.data and not self.skill_done: self.get_logger().info("Callback: Skill Execution Finished")
        self.skill_done = msg.data

    def _cb_pose(self, msg, is_tag):
        if self.state not in [State.S6_WAIT_DOCK_POSE, State.S11_WAIT_TARGET_POSE, State.S7_EXECUTE_DOCK, State.S8_WAIT_DOCK]:
            return

        active_obj = self.current_task["dock_obj"] if self.state == State.S6_WAIT_DOCK_POSE else self.current_task["target_obj"]
        if (is_tag and "tag" not in active_obj) or (not is_tag and "tag" in active_obj):
            return

        pos = msg.pose.position
        q = msg.pose.orientation
        r, p, y = self.euler_from_quat(q.x, q.y, q.z, q.w)
        z = np.array([pos.x, pos.y, pos.z, r, p, y])

        if not self.kf_initialized:
            self.kf_x = z
            self.kf_initialized = True
        else:
            self.kf_P = self.kf_P + self.kf_Q
            K = self.kf_P @ np.linalg.inv(self.kf_P + self.kf_R)
            self.kf_x = self.kf_x + K @ (z - self.kf_x)
            self.kf_P = (np.eye(6) - K) @ self.kf_P

        filtered_msg = PoseStamped()
        filtered_msg.header = msg.header
        filtered_msg.pose.position.x, filtered_msg.pose.position.y, filtered_msg.pose.position.z = self.kf_x[0], self.kf_x[1], self.kf_x[2]
        qx, qy, qz, qw = self.quat_from_euler(self.kf_x[3], self.kf_x[4], self.kf_x[5])
        filtered_msg.pose.orientation.x, filtered_msg.pose.orientation.y, filtered_msg.pose.orientation.z, filtered_msg.pose.orientation.w = qx, qy, qz, qw
        self.pub_target_pose.publish(filtered_msg)

        self.pose_buffer.append(self.kf_x.copy())
        if len(self.pose_buffer) > 10:
            self.pose_buffer.pop(0)

        # FIXME: pose is never stable upon change
        # if len(self.pose_buffer) == 10:
        #     std_devs = np.std(self.pose_buffer, axis=0)
        #     if np.all(std_devs[:3] < 0.1) and np.all(std_devs[3:] < 0.2):
        #         if not self.stable_pose:
        #             self.get_logger().info("Pose stabilized.")
        #         self.stable_pose = True
        #     else:
        #         self.get_logger().info("Pose not stable yet.", throttle_duration_sec=1.0)
        self.stable_pose = True
    def reset_pose_tracking(self):
        self.kf_initialized = False
        self.pose_buffer.clear()
        self.stable_pose = False

    def tick(self):
        if not self.tasks and self.state == State.DONE:
            return

        t = self.current_task

        # UI & Logging Updates
        # This state just publishes the current state and task to the UI, so we can see what the orchestrator is doing.
        if self.last_published_task != t["task"]:
            self.get_logger().info(f"\n================ STARTING TASK: {t['task'].upper()} ================")
            self.pub_ui_task.publish(String(data=t["task"]))
            self.last_published_task = t["task"]

        if self.last_published_state != self.state:
            self.get_logger().info(f"--> Transitioned to: {self.state.name}")
            self.pub_ui_state.publish(String(data=self.state.name))
            self.last_published_state = self.state

        # FSM Logic
        # This state is responsible for calling the navigation goal, and loading the docking mesh if needed.
        if self.state == State.S1_START_NAV_AND_MESH:
            self.pub_nav.publish(String(data=t["nav_goal"]))
            if t["dock_obj"] != "none":
                mesh_update_str = "mesh_update_" + t["dock_obj"]
                self.pub_mesh.publish(String(data=mesh_update_str))
            self.state = State.S2_WAIT_NAV

        # Here we wait for navigation to be done.
        elif self.state == State.S2_WAIT_NAV:
            if self.nav_done:
                self.nav_done = False
                self.state = State.S3_SET_HOME_CFG

        # Trigger the home configuration for the nav goal ( different home for pick, place, and fridge door )
        elif self.state == State.S3_SET_HOME_CFG:
            home_name = f"{t['nav_goal']}"
            #XXX: MAKE SURE THAT THIS MATCHES THE SERVICE NAME IN THE HOME CONFIGURATION NODE  (cartesian_interface in our case)
            srv_name = f"/home_position/{home_name}"

            # Cache the client if we haven't seen it yet
            if srv_name not in self.home_clients:
                self.home_clients[srv_name] = self.create_client(Trigger, srv_name)

            client = self.home_clients[srv_name]

            # Non-blocking service check (so we don't freeze the tick loop)
            if client.service_is_ready():
                req = Trigger.Request()
                future = client.call_async(req)
                future.add_done_callback(self._cb_home_future)
                self.state = State.S4_WAIT_HOME_CFG
            else:
                self.get_logger().warn(f"Waiting for home service: {srv_name}...", throttle_duration_sec=2.0)
                # Failsafe: Stays in S3_SET_HOME_CFG until service is up

        elif self.state == State.S4_WAIT_HOME_CFG:
            if self.home_trigger_failed:
                self.get_logger().warn("Home configuration failed! Bouncing back to S3 to retry...", throttle_duration_sec=2.0)
                self.home_trigger_failed = False
                self.state = State.S3_SET_HOME_CFG  # Retry the service call

            elif self.home_cfg_done:
                self.home_cfg_done = False
                self.state = State.S5_WAIT_DOCK_MESH if t["dock_obj"] != "none" else State.S12_EXECUTE_SKILL

        # This make sure the docking mesh is loaded before we start looking for the docking pose.
        elif self.state == State.S5_WAIT_DOCK_MESH:
            if self.mesh_loaded:
                self.mesh_loaded = False
                self.reset_pose_tracking()
                self.state = State.S6_WAIT_DOCK_POSE

        # Wait for the docking pose to stabilize
        elif self.state == State.S6_WAIT_DOCK_POSE:
            if self.stable_pose:
                self.stable_pose = False
                self.state = State.S7_EXECUTE_DOCK
        # Trigger the docking sequence, and wait for it to finish. Once done, we either go to the target mesh or execute the skill directly.
        elif self.state == State.S7_EXECUTE_DOCK:
            self.pub_dock.publish(Bool(data=True))
            self.state = State.S8_WAIT_DOCK

        # Wait for docking to be done
        elif self.state == State.S8_WAIT_DOCK:
            if self.dock_done:
                self.dock_done = False
                if t["target_obj"] == t["dock_obj"]:
                    self.reset_pose_tracking()
                    self.state = State.S11_WAIT_TARGET_POSE
                elif t["target_obj"] != "none":
                    self.state = State.S9_LOAD_TARGET_MESH
                else:
                    self.state = State.S12_EXECUTE_SKILL

        elif self.state == State.S9_LOAD_TARGET_MESH:
            mesh_update_str = "mesh_update_" + t["target_obj"]
            self.pub_mesh.publish(String(data=mesh_update_str))
            self.state = State.S10_WAIT_TARGET_MESH

        elif self.state == State.S10_WAIT_TARGET_MESH:
            if self.mesh_loaded:
                self.mesh_loaded = False
                self.reset_pose_tracking()
                self.state = State.S11_WAIT_TARGET_POSE

        elif self.state == State.S11_WAIT_TARGET_POSE:
            if self.stable_pose:
                self.stable_pose = False
                self.state = State.S12_EXECUTE_SKILL

        elif self.state == State.S12_EXECUTE_SKILL:
            if t["skill"] != "none":
                self.pub_skill.publish(String(data=t["skill"]))
                self.state = State.S13_WAIT_SKILL
            else:
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
        t0 = +2.0 * (w * x + y * z); t1 = +1.0 - 2.0 * (x * x + y * y); roll = math.atan2(t0, t1)
        t2 = +2.0 * (w * y - z * x); t2 = max(min(t2, 1.0), -1.0); pitch = math.asin(t2)
        t3 = +2.0 * (w * z + x * y); t4 = +1.0 - 2.0 * (y * y + z * z); yaw = math.atan2(t3, t4)
        return roll, pitch, yaw

    def quat_from_euler(self, roll, pitch, yaw):
        cr = math.cos(roll * 0.5); sr = math.sin(roll * 0.5)
        cp = math.cos(pitch * 0.5); sp = math.sin(pitch * 0.5)
        cy = math.cos(yaw * 0.5); sy = math.sin(yaw * 0.5)
        return sr*cp*cy - cr*sp*sy, cr*sp*cy + sr*cp*sy, cr*cp*sy - sr*sp*cy, cr*cp*cy + sr*sp*sy

def main(args=None):
    rclpy.init(args=args)
    node = PickPlaceOrchestrator()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()