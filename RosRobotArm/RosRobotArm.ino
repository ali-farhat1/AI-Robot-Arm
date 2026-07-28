#include <ESP32Servo.h>
#include <ServoEasing.hpp>

// ====== Micro Ros ===========
#include <micro_ros_arduino.h>
#include <stdio.h>
#include <rcl/rcl.h>
#include <rcl/error_handling.h>
#include <rclc/rclc.h>
#include <rclc/executor.h>
#include <std_msgs/msg/int32.h>
#include <std_msgs/msg/float32.h>
#include <std_msgs/msg/float32_multi_array.h>
#include <std_msgs/msg/char.h>

// Declare ROS objects
rcl_node_t node;

rcl_publisher_t UltraSonic_Publisher;
std_msgs__msg__Float32 UltraSonic_msg;

rcl_subscription_t Servo_Movements_Subscriber;
std_msgs__msg__Float32MultiArray Servo_Movements_msg;

rcl_publisher_t Servo_Movements_Publisher;
std_msgs__msg__Float32MultiArray Servo_Movements_Publisher_msg;

rclc_executor_t executor;
rclc_support_t support;
rcl_allocator_t allocator;

//----------------------------------

// ===== Ultrasonic =====
const int trig = 13;
const int echo = 12;


// ===== GLOBAL SPEED =====
uint16_t servoSpeed = 45;  

// ===== Servos =====
ServoEasing Base, Shoulder, Elbow, Wrist, ClawMover, Claw;

// ===== Targets =====
float Base_Target = 90, Shoulder_Target = 90, Elbow_Target = 45;
float Wrist_Target = 90, ClawMove_Target = 90, Claw_Target = 0;

float lastBase = 90, lastShoulder = 90, lastElbow = 45;
float lastWrist = 90, lastClawMover = 90, lastClaw = 0;

// forward declaration
void applySpeed();

// ===== Ultrasonic =====
float getDistance() {
  digitalWrite(trig, LOW);
  delayMicroseconds(2);

  digitalWrite(trig, HIGH);
  delayMicroseconds(10);
  digitalWrite(trig, LOW);

  long duration = pulseIn(echo, HIGH, 30000);
  return duration * 0.0343 / 2;
}

void updateCurrentAngles() {
  lastBase      = Base.getCurrentAngle();
  lastShoulder  = Shoulder.getCurrentAngle();
  lastElbow     = Elbow.getCurrentAngle();
  lastWrist     = Wrist.getCurrentAngle();
  lastClawMover = ClawMover.getCurrentAngle();
  lastClaw      = Claw.getCurrentAngle();
}



void sermoMovement_callback(const void * msg_in) {
  // 1. Cast the generic message into a type the compiler understands
  const std_msgs__msg__Float32MultiArray * msg = (const std_msgs__msg__Float32MultiArray *) msg_in;

  // 2. Access the data array
  // msg->data.data is the actual pointer to your list of angles
  if (msg->data.size >= 6) { // Make sure we got at least 6 angles
    Base.startEaseTo(msg->data.data[0]);
    Shoulder.startEaseTo(msg->data.data[1]);
    Elbow.startEaseTo(msg->data.data[2]);
    Wrist.startEaseTo(msg->data.data[3]);
    ClawMover.startEaseTo(msg->data.data[4]);
    Claw.startEaseTo(msg->data.data[5]);
  }
}


void setup() {
  Serial.begin(115200);

  // ======= ROS ===========
  set_microros_transports();

  std_msgs__msg__Float32__init (&UltraSonic_msg);
  std_msgs__msg__Float32MultiArray__init (&Servo_Movements_msg);
  std_msgs__msg__Float32MultiArray__init (&Servo_Movements_Publisher_msg);

  // 1. Initialize micro-ROS
  allocator = rcl_get_default_allocator();
  rclc_support_init(&support, 0, NULL, &allocator);
  rclc_node_init_default(&node, "esp32_publisher", "", &support);
  
  // 2. Create Publisher on topic
  rclc_publisher_init_default(
    &UltraSonic_Publisher,
    &node,
    ROSIDL_GET_MSG_TYPE_SUPPORT(std_msgs, msg, Float32),
    "ultrasonic_data");


  rclc_publisher_init_default(
    &Servo_Movements_Publisher,
    &node,
    ROSIDL_GET_MSG_TYPE_SUPPORT(std_msgs, msg, Float32MultiArray),
    "live_servo_movements");

  
  // 3. Create Subscriber on topic "servo_movement_data"
  rclc_subscription_init_default(
    &Servo_Movements_Subscriber,
    &node,
    ROSIDL_GET_MSG_TYPE_SUPPORT(std_msgs, msg, Float32MultiArray),
    "servo_angles"
  );



  const size_t number_of_handles = 2; 
  rclc_executor_init(&executor, &support.context, number_of_handles, &allocator);

  rclc_executor_add_subscription( &executor, &Servo_Movements_Subscriber, &Servo_Movements_msg, &sermoMovement_callback, ON_NEW_DATA);

  // Allocate memory for the incoming array (e.g., 6 elements) servo_movements subscriber
  static float servo_data_buffer[6];
  Servo_Movements_msg.data.capacity = 6;
  Servo_Movements_msg.data.size = 0;
  Servo_Movements_msg.data.data = servo_data_buffer;


  // For Servo Movement Publisher
  // 1. Define the buffer for the publisher
  static float servo_pub_data_buffer[6]; 

  // 2. Setup the publisher message structure
  Servo_Movements_Publisher_msg.data.capacity = 6;
  Servo_Movements_Publisher_msg.data.size = 6; // Set to 6 since it is publishing 6 angles
  Servo_Movements_Publisher_msg.data.data = servo_pub_data_buffer;



  //----------------------------------
  pinMode(trig, OUTPUT);
  pinMode(echo, INPUT);

  Base.attach(26);
  Shoulder.attach(25);
  Elbow.attach(33);
  Wrist.attach(32);
  ClawMover.attach(18);
  Claw.attach(19);

  applySpeed();

  // ===== smooth startup movement =====
  Base.startEaseTo(90);
  Shoulder.startEaseTo(90);
  Elbow.startEaseTo(45);
  Wrist.startEaseTo(90);
  ClawMover.startEaseTo(90);
  Claw.startEaseTo(0);

  delay(3000); // give time to settle

  updateCurrentAngles();
}

void applySpeed() {
  Base.setSpeed(servoSpeed);
  Shoulder.setSpeed(servoSpeed);
  Elbow.setSpeed(servoSpeed);
  Wrist.setSpeed(servoSpeed);
  ClawMover.setSpeed(servoSpeed);
  Claw.setSpeed(servoSpeed);
}

void loop() {

  rclc_executor_spin_some(&executor, RCL_MS_TO_NS(10));

  updateCurrentAngles();
  UltraSonic_msg.data = getDistance();
  rcl_publish(&UltraSonic_Publisher, &UltraSonic_msg, NULL);

  Servo_Movements_Publisher_msg.data.data[0] = Base.getCurrentAngle();
  Servo_Movements_Publisher_msg.data.data[1] = Shoulder.getCurrentAngle();
  Servo_Movements_Publisher_msg.data.data[2] = Elbow.getCurrentAngle();
  Servo_Movements_Publisher_msg.data.data[3] = Wrist.getCurrentAngle();
  Servo_Movements_Publisher_msg.data.data[4] = ClawMover.getCurrentAngle();
  Servo_Movements_Publisher_msg.data.data[5] = Claw.getCurrentAngle();

  rcl_publish(&Servo_Movements_Publisher, &Servo_Movements_Publisher_msg, NULL);

  delay(100);
}