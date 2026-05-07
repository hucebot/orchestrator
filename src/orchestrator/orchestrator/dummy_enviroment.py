import rclpy
from rclpy.node import Node
import time
import threading

# ROS 2 Messages
from std_msgs.msg import String, Bool
from geometry_msgs.msg import PoseStamped
from std_srvs.srv import Trigger

class DummyEnvironment(Node):
    def __init__(self):
        super().__init__('dummy_environment')
        
        self.get_logger().info("Starting Dummy Environment...")

        # --- Publishers (Simulating robot feedback to Orchestrator) ---
        self.pub_nav_done = self.create_publisher(Bool, '/nav/goal/is_done', 10)
        self.pub_motion_status = self.create_publisher(String, '/motion_recorder/status', 10)
        self.pub_object_pose = self.create_publisher(PoseStamped, '/object_pose', 10)

        # --- Subscribers (Listening to Orchestrator commands) ---
        self.sub_toggle_fp = self.create_subscription(Bool, '/orchestrator/pose/toggle_fp', self.toggle_fp_cb, 10)
        self.sub_nav_target = self.create_subscription(String, '/nav/goal/target', self.nav_target_cb, 10)
        self.sub_target_skill = self.create_subscription(String, '/motion_recorder/target_skill', self.target_skill_cb, 10)
        self.sub_target_object = self.create_subscription(String, '/orchestrator/pose/target_object', self.target_object_cb, 10)

        # --- Services (Simulating robot services) ---
        self.srv_home_pose = self.create_service(Trigger, '/trigger_home_pose', self.home_pose_cb)

        # --- Timers ---
        # Publish a perfectly stable pose at 10Hz to satisfy your 5-frame stability check
        self.pose_timer = self.create_timer(0.1, self.publish_stable_pose)

    # ===============================
    # SIMULATED SENSOR OUTPUT
    # ===============================
    def publish_stable_pose(self):
        """Constantly publishes a stable pose so the orchestrator proceeds when waiting."""
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "camera_link"
        
        # Static stable pose
        msg.pose.position.x = 0.5
        msg.pose.position.y = 0.0
        msg.pose.position.z = 0.3
        msg.pose.orientation.x = 0.0
        msg.pose.orientation.y = 0.0
        msg.pose.orientation.z = 0.0
        msg.pose.orientation.w = 1.0
        
        self.pub_object_pose.publish(msg)

    # ===============================
    # ORCHESTRATOR COMMAND LISTENERS
    # ===============================
    def toggle_fp_cb(self, msg: Bool):
        state = "ON" if msg.data else "OFF"
        self.get_logger().info(f"[DUMMY] FoundationPose toggled: {state}")

    def target_object_cb(self, msg: String):
        self.get_logger().info(f"[DUMMY] Target Object updated to: {msg.data}")

    def nav_target_cb(self, msg: String):
        self.get_logger().info(f"[DUMMY] Received Nav Goal: {msg.data}. Simulating driving...")
        # Spin up a thread to simulate a delay without blocking the ROS executor
        threading.Thread(target=self.simulate_nav_completion).start()

    def target_skill_cb(self, msg: String):
        self.get_logger().info(f"[DUMMY] Executing Motion Skill: {msg.data}. Simulating motion...")
        threading.Thread(target=self.simulate_motion_completion).start()

    def home_pose_cb(self, request, response):
        self.get_logger().info("[DUMMY] Home pose service triggered. Moving home...")
        time.sleep(0.5) # Simulate slight delay for arm movement
        response.success = True
        response.message = "Successfully reached home pose"
        return response

    # ===============================
    # DELAYED RESPONSES (Threads)
    # ===============================
    def simulate_nav_completion(self):
        """Simulates navigation taking 2 seconds, then publishes success."""
        time.sleep(2.0)
        self.get_logger().info("[DUMMY] Navigation complete! Publishing is_done=True")
        msg = Bool()
        msg.data = True
        self.pub_nav_done.publish(msg)
        
        # Reset flag quickly so it doesn't instantly trigger the next state early
        time.sleep(0.5)
        msg.data = False
        self.pub_nav_done.publish(msg)

    def simulate_motion_completion(self):
        """Simulates a motion skill taking 2.5 seconds, then publishes 'done'."""
        time.sleep(2.5)
        self.get_logger().info("[DUMMY] Motion complete! Publishing status='done'")
        msg = String()
        msg.data = "done"
        self.pub_motion_status.publish(msg)
        
        # Clear status so it doesn't trigger future states instantly
        time.sleep(0.5)
        msg.data = "idle"
        self.pub_motion_status.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = DummyEnvironment()
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