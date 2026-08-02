import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, String
from ..llm.hybrid_memory import SkyeMemory

robot_actions = [up_down, right_left, look_forward, look_up, look_down, tilt_left, tilt_right]
memory_actions = [recall_memory]

class RobotActions(Node):
    def __init__(self):
        super().__init__('RobotActions')

        self.servo_publisher = self.create_publisher(Float32MultiArray, 'servo_angles', 10)

        self.action_input = self.create_subscription(String, '/actions_input', action_taker_callback, 10)
        self.action_output = self.create_publisher(String, "/actions_output", 10)

        self.robot_action = self.create_publisher(String, '/robot_action', 10)

    def action_taker_callback(self, Action):
        self.take_action(Action)

    def take_action(self, Action):
        msg = String()
        msg.data 

        if Action in robot_actions:
            self.robot_action.publish(msg)

        elif Action in memory_actions:
            ...

        return
        


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