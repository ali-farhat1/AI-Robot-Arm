import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from tf2_msgs.msg import TFMessage
from rclpy.duration import Duration
from scipy.spatial.transform import Rotation as R


from robot_ai.llm.AI_client import OpenRouterClient

actions = ["up_down", "right_left"]

class Brain(Node):
    def __init__(self):
        super().__init__('Brain')

        self.action_taker_publisher = self.create_publisher(String, 'actions', 10) # Publish Ai Actions

        self.timer = self.create_timer(0.2, self.live_coords)

        self.ai_response_publisher = self.create_publisher(String, "/ai/response", 10)
        self.ai_request_subscriber = self.create_subscription(String, "ai/request", self.ai_request_callback, 10)


        self.client = OpenRouterClient()

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(
            self.tf_buffer,
            self
        )

        self.live_servo_coord = {}


    def take_action(self, Action):
        msg = String()
        msg.data = Action   
        self.action_taker_publisher.publish(msg)

    

    def live_coords(self):
        frames = [
            "servo1_body_link",
            "servo2_body_link",
            "servo3_body_link",
            "servo4_body_link",
            "servo5_body_link"
        ]

        parent_links = [
            "base_link",
            "servo1_horn_link",
            "upper_arm_h_bracket_link",
            "forearm_cylinder_link",
            "wrist_bracket_link"
        ]
        coords_dict = {}

        for num, frame in enumerate(frames):
                try:
                    if not self.tf_buffer.can_transform(
                        parent_links[num],
                        frame,
                        rclpy.time.Time(),
                        timeout=Duration(seconds=1.0)
                    ):
                        #print(f"Can't transform {frame}")
                        continue

                    trans = self.tf_buffer.lookup_transform(
                        parent_links[num],
                        frame,
                        rclpy.time.Time()
                    )

                    # Store the x, y, z values in the dictionary
                    coords_dict[frame] = {
                        "x": trans.transform.translation.x,
                        "y": trans.transform.translation.y,
                        "z": trans.transform.translation.z
                    }

                    # Optional: Keep the print statement to see live updates
                    #print(f"{frame}: {coords_dict[frame]['x']:.4f}, {coords_dict[frame]['y']:.4f}, {coords_dict[frame]['z']:.4f}")

                except Exception as e:
                    #print(frame, e)
                    return
            
        self.live_servo_coord = coords_dict


    def ai_request_callback(self, msg):
        prompt = msg.data

        response = self.ai_chat(prompt)

        actions = response.get("action_sequence", [])

        for action in actions:
            self.take_action(action)

    
    def ai_chat(self, prompt):
        try:
            response = self.client.chat_main(
                prompt=prompt,
                addtional_ai_context={"Current Robot Tf Reading": self.live_servo_coord} if self.live_servo_coord else ""
            )

            ros_msg = String()
            #ros_msg.data = str(response["text"])
            ros_msg.data = str(response)

            self.ai_response_publisher.publish(ros_msg)

            return response

        except Exception as e:
            msg = String()
            msg.data = "Error: " + str(e)

            self.ai_response_publisher.publish(msg)

            return None

        

        

    
    def test(self):
        user_input = input("You: ")

        try:
            addtional_ai_context =  {"Current Robot Tf Reading: " : self.live_servo_coord}
            response = self.client.chat_main(
                prompt=user_input,
                addtional_ai_context= addtional_ai_context
            )

            print("AI:", response)

            if response["action_sequence"]:
                for action in response["action_sequence"]:
                    self.take_action(action)
        
        except Exception as e:
            print("Error:", e)
        


def main():
    rclpy.init()
    node = Brain()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()