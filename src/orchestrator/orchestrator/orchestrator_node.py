import rclpy
from rclpy.node import Node


class OrchestratorNode(Node):
	def __init__(self):
		super().__init__('orchestrator')
		self.get_logger().info('Orchestrator node started')
		self._counter = 0
		self._timer = self.create_timer(1.0, self.timer_callback)

	def timer_callback(self):
		self._counter += 1
		self.get_logger().info(f'Heartbeat: {self._counter}')


def main(args=None):
	rclpy.init(args=args)
	node = OrchestratorNode()
	try:
		rclpy.spin(node)
	except KeyboardInterrupt:
		pass
	finally:
		node.destroy_node()
		rclpy.shutdown()


if __name__ == '__main__':
	main()
