import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
from sensor_msgs.msg import PointCloud2, PointField
import struct
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

class UltraSonic_PointCloud_Publisher(Node):
    def __init__(self):
        super().__init__('UltraSonic_PointCloud_Publisher')

        self.ultrasonic = 0
        # Gets ultrasonic data from Esp32 publisher
        self.ultrasonic_sub = self.create_subscription(Float32, "/ultrasonic_data", self.ultrasonic_callback, 10)

        # Publisher to pointcloud2
        self.pc_pub = self.create_publisher(PointCloud2, "/sonar/cloud", 20)

        # Create a timer to look up the coordinates every 0.2 seconds
        self.timer = self.create_timer(0.2, self.get_link_coordinates)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(
            self.tf_buffer,
            self
        )

        self.target_frame = 'ultrasonic_sensor_frame'
        self.source_frame = 'base_link'
        
    
    def ultrasonic_callback(self, distance):
        self.ultrasonic = float(distance.data)
        #print(f"Distance: {self.ultrasonic} cm")


        # Calculate the Z coordinate in meters
        # Is negative due to urdf
        z_coords = - (self.ultrasonic / 100.0)

        # Create a standard ROS2 PointCloud2 Message
        msg = PointCloud2()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'ultrasonic_sensor_frame' 

        # Define the structure of the 3D point cloud array (X, Y, Z coordinates as floats)
        msg.fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1)
        ]
        
        msg.is_bigendian = False
        msg.point_step = 12  # 3 fields x 4 bytes per float = 12 bytes
        msg.row_step = 12
        msg.height = 1        # Unorganized point cloud structure
        msg.width = 1         # Sending 1 single point at a time
        msg.is_dense = True

        #                           x   y       z
        # Pack the 3D coordinates (0.0, 0.0, z_coords) into binary bytes
        
        msg.data = struct.pack('<fff', 0.0, 0.0, z_coords)

        # 3. Publish the point cloud to OctoMap
        self.pc_pub.publish(msg)


    def get_link_coordinates(self):
        try:
            # Look up the latest available transform
            now = rclpy.time.Time()
            trans = self.tf_buffer.lookup_transform(
                self.source_frame,
                self.target_frame,
                now)

            # Extract the exact X, Y, Z coordinates
            x = trans.transform.translation.x
            y = trans.transform.translation.y
            z = trans.transform.translation.z

            # Extract orientation 
            rx = trans.transform.rotation.x
            ry = trans.transform.rotation.y
            rz = trans.transform.rotation.z
            rw = trans.transform.rotation.w

            # Log the extracted coordinates to the console
            self.get_logger().info(f'Link Coordinates -> X: {x:.4f}, Y: {y:.4f}, Z: {z:.4f}')

        except TransformException as ex:
            self.get_logger().warning(f'Could not transform {self.target_frame} to {self.source_frame}: {ex}')


def main():
    rclpy.init()
    node = UltraSonic_PointCloud_Publisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
