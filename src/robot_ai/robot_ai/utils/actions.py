import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, String

actions = [up_down, right_left]

class RobotActions(Node):
    def __init__(self):
        super().__init__('RobotActions')

        self.servo_publisher = self.create_publisher(Float32MultiArray, 'servo_angles', 10)

        self.action_taker = self.create_subscription(String, 'actions', self.action_taker_callback, 10)

    def action_taker_callback(self, Action):
        if lower(Action) not in actions:
            return

        self.take_action(Action)

    def take_action(self, Action):
        ...


if __name__ == "__main__":
    Node = RobotActions

    while True:
        user_input = input("You: ")

        try:
            response = client.chat_json(
                prompt=user_input,
                system=load_system_prompt()
            )

            print("AI:", response)


        
        except Exception as e:
            print("Error:", e)