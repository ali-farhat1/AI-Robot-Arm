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

robot_actions = ["up_down", "right_left", "look_forward", "look_up", "look_down", "tilt_left", "tilt_right"]

class Brain(Node):
    def __init__(self):
        super().__init__('Brain')

        self.action_taker_publisher = self.create_publisher(String, '/actions_input', 10) # Publish Ai Actions

        self.robot_action = self.create_publisher(String, '/robot_action', 10) # AI controls Robot 

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

        if Action in robot_actions:    
            self.robot_action.publish(msg)

    

    def live_coords(self):
        # TODO: make it be a dicitonary which stores TF values from 20 seconds/ticks before to now

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


                except Exception as e:
                    #print(frame, e)
                    return
            
        self.live_servo_coord = coords_dict


    def ai_request_callback(self, msg):
        response = self.ai_chat(msg.data)
        if not response:
            return

        response = self.resolve_memory(response)
        if not response:
            return

        for action in response.get("action_sequence", []):
            self.take_action(action)

        # publish/handle response here same as before
        msg = String()
        msg.data = str(response)

        self.ai_response_publisher.publish(msg)



    def resolve_memory(self, response, previous_memory="", depth=0):
        query = response.get("memory_query")
        if not query:
            return response

        print("Searching Memory")
        memory_result = self.client.recall_memory(query)
        print(f"Memory Result: {memory_result}")
        full_memory = str(previous_memory) + str(memory_result)

        at_cap = depth >= 2  

        context = {"recalled_memory": full_memory}
        prompt = "Here is what you recalled."

        if at_cap:
            context["memory_search_limit_reached"] = True
            prompt = (
                "Here is what you recalled. This is the last memory you can search for right now — "
                "you must answer with what you have, even if it's incomplete or uncertain."
            )

        followup = self.ai_chat(prompt=prompt, addtional_ai_context=context)

        if not followup:
            return None

        if at_cap or not followup.get("memory_query"):
            return followup

        return self.resolve_memory(followup, previous_memory=full_memory, depth=depth + 1)

            
            
    def ai_chat(self, prompt, addtional_ai_context = ""):
        try:
            context = {}
            if self.live_servo_coord:
                context["Robot Tf Reading from 20 seconds ago to now"] = self.live_servo_coord
            if addtional_ai_context:
                context.update(addtional_ai_context)

            response = self.client.chat_main(prompt=prompt, addtional_ai_context=context)


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