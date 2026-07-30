import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class RobotUI(Node):

    def __init__(self):
        super().__init__('robot_ui')

        self.ai_response = ""

        self.ai_response_subscriber = self.create_subscription(
            String,
            "/ai/response",
            self.ai_callback,
            10
        )

        self.ai_request_publisher = self.create_publisher(
            String,
            "/ai/request",
            10
        )

        self.main_chat()

    def ai_callback(self, msg):
        self.ai_response = msg.data
        print("AI:", self.ai_response)
        

    def main_chat(self):
        while rclpy.ok():
            text = input("You: ")

            msg = String()
            msg.data = text

            self.ai_request_publisher.publish(msg)

            # Allow ROS to receive the AI response
            rclpy.spin_once(self)


def main(args=None):
    rclpy.init(args=args)

    node = RobotUI()

    rclpy.shutdown()


if __name__ == "__main__":
    main()