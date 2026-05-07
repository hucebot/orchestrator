
import rclpy
from rclpy.node import Node
from enum import Enum
import math
import time

# ROS 2 Messages
from std_msgs.msg import String, Bool
from geometry_msgs.msg import PoseStamped
from std_srvs.srv import Trigger  # Using Trigger as a placeholder for the Home Pose service

class State(Enum):
    INIT = 1
    WAIT_FRIDGE_REACHED = 2
    WAIT_APRIL_TAG_POSE = 3
    WAIT_OPEN_FRIDGE_DONE = 4
    
    # Item Loop States
    GOING_TO_TABLE = 5
    WAIT_TABLE_REACHED = 6
    WAIT_ITEM_POSE = 7
    WAIT_ITEM_GRASPED = 8
    GOING_TO_PLACE_FRIDGE = 9
    WAIT_PLACE_FRIDGE_REACHED = 10
    WAIT_ITEM_PLACED = 11
    
    # Close Fridge States
    GOING_TO_INT_FRIDGE_CLOSE = 12
    WAIT_INT_FRIDGE_CLOSE = 13
    WAIT_APRIL_TAG_POSE_2 = 14
    WAIT_CLOSE_FRIDGE = 15
    DONE = 16

class PickPlaceOrchestrator(Node):
    def __init__(self):
        super().__init__('pick_place_orchestrator')
        
        # Items List
        self.items_to_process = ['mustard', 'bowl', 'gavottes', 'juice', 'milk', 'plate']
        self.current_item = None

        # State Management
        self.state = State.INIT
        self.timer = self.create_timer(0.1, self.tick) # 10 Hz state machine evaluation
        
        # Flags & Buffers
        self.nav_is_done = False
        self.motion_status = ""
        self.pose_buffer = []
        self.stable_pose_achieved = False

        # --- Publishers ---
        self.pub_toggle_fp = self.create_publisher(Bool, '/orchestrator/pose/toggle_fp', 10)
        self.pub_nav_target = self.create_publisher(String, '/nav/goal/target', 10)
        self.pub_target_skill = self.create_publisher(String, '/motion_recorder/target_skill', 10)
        self.pub_target_object = self.create_publisher(String, '/orchestrator/pose/target_object', 10)

        # --- Subscribers ---
        self.sub_nav_done = self.create_subscription(Bool, '/nav/goal/is_done', self.nav_done_cb, 10)
        self.sub_motion_status = self.create_subscription(String, '/motion_recorder/status', self.motion_status_cb, 10)
        self.sub_object_pose = self.create_subscription(PoseStamped, '/object_pose', self.object_pose_cb, 10)
        self.sub_april_tag_pose = self.create_subscription(PoseStamped, 'apriltag_pose/pose_tag_2', self.object_pose_cb, 10) # Assuming same processing for simplicity

        # --- Services ---
        # Placeholder for your home pose service # TODO: replace with self.srv_home = self.create_service(Trigger, 'home_position', self._home_service_cb)
        self.client_home_pose = self.create_client(Trigger, '/trigger_home_pose')

        self.get_logger().info("Orchestrator initialized. Starting state machine...")

    # ===============================
    # CALLBACKS
    # ===============================
    def nav_done_cb(self, msg: Bool):
        self.nav_is_done = msg.data

    def motion_status_cb(self, msg: String):
        self.motion_status = msg.data

    def object_pose_cb(self, msg: PoseStamped):
        """Buffers poses and evaluates stability (5 frames, <5s, <0.05m, <0.05rad)"""
        # Only process if we are actually waiting for a stable pose
        if self.state not in [State.WAIT_APRIL_TAG_POSE, State.WAIT_ITEM_POSE, State.WAIT_APRIL_TAG_POSE_2]:
            return
            
        timestamp = time.time() # Using system time for frame arrival delta
        pos = msg.pose.position
        quat = msg.pose.orientation
        roll, pitch, yaw = self.euler_from_quaternion(quat.x, quat.y, quat.z, quat.w)
        
        self.pose_buffer.append((timestamp, pos.x, pos.y, pos.z, roll, pitch, yaw))
        
        # Keep only the last 5 frames
        if len(self.pose_buffer) > 5:
            self.pose_buffer.pop(0)
            
        if len(self.pose_buffer) == 5:
            times = [b[0] for b in self.pose_buffer]
            # Check time constraint (< 5 seconds)
            if max(times) - min(times) < 5.0:
                # Check spatial constraints (< 0.05m translation, < 0.05rad rotation)
                max_diffs = []
                for i in range(1, 7): # indices 1 to 6 are x, y, z, roll, pitch, yaw
                    vals = [b[i] for b in self.pose_buffer]
                    max_diffs.append(max(vals) - min(vals))
                
                if all(diff < 0.05 for diff in max_diffs):
                    self.stable_pose_achieved = True

    # ===============================
    # HELPER METHODS
    # ===============================
    def trigger_home_pose(self):
        """Trigger the home pose service asynchronously and handle response."""

        self.get_logger().info("Triggering Home Pose...")
        if not self.client_home_pose.wait_for_service(timeout_sec=1.0):
            self.get_logger().error("Home pose service not available!")
            return

        req = Trigger.Request()
        future = self.client_home_pose.call_async(req)
        future.add_done_callback(self._on_home_response)

    def _on_home_response(self, future):
        try:
            response = future.result()
            if response.success:
                self.get_logger().info(f"Home pose succeeded: {response.message}")
            else:
                self.get_logger().warn(f"Home pose failed: {response.message}")
        except Exception as e:
            self.get_logger().error(f"Home service call failed: {e}")

    def reset_flags(self):
        self.nav_is_done = False
        self.motion_status = ""
        self.stable_pose_achieved = False
        self.pose_buffer = []

    def euler_from_quaternion(self, x, y, z, w):
        """Converts quaternion to euler angles manually to avoid TF dependencies."""
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = math.atan2(t0, t1)
        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch_y = math.asin(t2)
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = math.atan2(t3, t4)
        return roll_x, pitch_y, yaw_z

    def publish_string(self, publisher, text):
        msg = String()
        msg.data = text
        publisher.publish(msg)

    # ===============================
    # STATE MACHINE TICK
    # ===============================
    def tick(self):
        if self.state == State.INIT:
            self.get_logger().info("1 & 2. Starting FP and Navigating to open fridge")
            msg_bool = Bool()
            msg_bool.data = True
            self.pub_toggle_fp.publish(msg_bool)
            self.publish_string(self.pub_nav_target, "fridge_open")
            self.reset_flags()
            self.state = State.WAIT_FRIDGE_REACHED

        elif self.state == State.WAIT_FRIDGE_REACHED:
            if self.nav_is_done:
                self.get_logger().info("3. Reached fridge. Triggering home pose.")
                self.trigger_home_pose()
                self.reset_flags()
                self.state = State.WAIT_APRIL_TAG_POSE

        elif self.state == State.WAIT_APRIL_TAG_POSE:
            if self.stable_pose_achieved:
                self.get_logger().info("4 & 5. April tag stable. Opening fridge.")
                self.publish_string(self.pub_target_skill, "open_fridge")
                self.reset_flags()
                self.state = State.WAIT_OPEN_FRIDGE_DONE

        elif self.state == State.WAIT_OPEN_FRIDGE_DONE:
            if self.motion_status == "done":
                if len(self.items_to_process) > 1:
                    self.current_item = self.items_to_process.pop(0)
                    self.get_logger().info(f"6.i. Proceeding with item: {self.current_item}. Going to table.")
                    self.publish_string(self.pub_nav_target, "table")
                    self.trigger_home_pose()
                    self.reset_flags()
                    self.state = State.WAIT_TABLE_REACHED
                else:
                    self.get_logger().info("7. All items processed. Going to intermediate fridge close.")
                    self.publish_string(self.pub_nav_target, "fridge_close_intermidiate")
                    self.trigger_home_pose()
                    self.reset_flags()
                    self.state = State.WAIT_INT_FRIDGE_CLOSE

        elif self.state == State.WAIT_TABLE_REACHED:
            if self.nav_is_done:
                self.get_logger().info("6.ii & 6.iii. Reached table. Waiting for item pose.")
                self.reset_flags()
                self.state = State.WAIT_ITEM_POSE

        elif self.state == State.WAIT_ITEM_POSE:
            if self.stable_pose_achieved:
                self.get_logger().info("6.iv. Item pose stable. Grasping item.")
                if self.current_item is None:
                    self.get_logger().error("No current item set before grasping")
                    return
                self.publish_string(self.pub_target_skill, "grasp_" + self.current_item)
                self.reset_flags()
                self.state = State.WAIT_ITEM_GRASPED

        elif self.state == State.WAIT_ITEM_GRASPED:
            if self.motion_status == "done":
                self.get_logger().info("6.v & 6.vi. Item grasped. Going to fridge.")
                self.publish_string(self.pub_nav_target, "fridge")
                self.trigger_home_pose()
                self.reset_flags()
                self.state = State.WAIT_PLACE_FRIDGE_REACHED

        elif self.state == State.WAIT_PLACE_FRIDGE_REACHED:
            if self.nav_is_done:
                self.get_logger().info("6.vii & 6.viii & 6.ix. Reached fridge. Updating mesh and placing.")
                # Dynamically set target object based on current item
                self.publish_string(self.pub_target_object, f"mesh_update_{self.current_item}")
                self.publish_string(self.pub_target_skill, "place_item")
                self.reset_flags()
                self.state = State.WAIT_ITEM_PLACED

        elif self.state == State.WAIT_ITEM_PLACED:
            # Assuming we need to wait for place completion before looping back
            if self.motion_status == "done":
                self.get_logger().info(f"Item {self.current_item} placed successfully. Looping.")
                self.state = State.WAIT_OPEN_FRIDGE_DONE # Routes back to check if list is empty

        elif self.state == State.WAIT_INT_FRIDGE_CLOSE:
            if self.nav_is_done:
                self.get_logger().info("8. Reached intermediate close pose. Waiting for tag.")
                self.reset_flags()
                self.state = State.WAIT_APRIL_TAG_POSE_2

        elif self.state == State.WAIT_APRIL_TAG_POSE_2:
            if self.stable_pose_achieved:
                self.get_logger().info("9 & 10. Tag stable. Closing fridge.")
                self.publish_string(self.pub_target_skill, "close_fridge")
                self.reset_flags()
                self.state = State.WAIT_CLOSE_FRIDGE

        elif self.state == State.WAIT_CLOSE_FRIDGE:
            if self.motion_status == "done":
                self.get_logger().info("Sequence Complete! Shutting down Orchestrator.")
                self.state = State.DONE
                rclpy.shutdown()

def main(args=None):
    rclpy.init(args=args)
    node = PickPlaceOrchestrator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
    rclpy.shutdown()
