import ast
import json
import subprocess
import tkinter as tk

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class RobotUI(Node):

    def __init__(self):
        super().__init__("robot_ui")

        self.waiting = False

        # ROS
        self.ai_sub = self.create_subscription(
            String,
            "/ai/response",
            self.ai_callback,
            10,
        )

        self.ai_pub = self.create_publisher(
            String,
            "/ai/request",
            10,
        )

        # Window
        self.root = tk.Tk()
        self.root.title("Robot AI")
        self.root.geometry("700x500")
        self.root.configure(bg="#1e1e1e")

        tk.Label(
            self.root,
            text="🤖 Robot AI",
            bg="#1e1e1e",
            fg="white",
            font=("Segoe UI", 24, "bold"),
        ).pack(pady=10)

        # Chat
        self.chat = tk.Text(
            self.root,
            bg="#252526",
            fg="white",
            insertbackground="white",
            wrap="word",
            relief="flat",
            font=("Segoe UI", 16),
            padx=15,
            pady=15,
        )
        self.chat.pack(fill="both", expand=True, padx=15)
        self.chat.config(state="disabled")

        self.chat.tag_config("you", foreground="#4FC3F7", font=("Segoe UI", 16, "bold"))
        self.chat.tag_config("robot", foreground="#7CFC98", font=("Segoe UI", 16, "bold"))

        # Bottom
        bottom = tk.Frame(self.root, bg="#1e1e1e")
        bottom.pack(fill="x", padx=15, pady=15)

        self.entry = tk.Entry(
            bottom,
            bg="#333333",
            fg="white",
            insertbackground="white",
            relief="flat",
            font=("Segoe UI", 16),
        )
        self.entry.pack(side="left", fill="x", expand=True, ipady=12)
        self.entry.bind("<Return>", self.send_message)

        tk.Button(
            bottom,
            text="Send",
            command=self.send_message,
            bg="#0e639c",
            fg="white",
            relief="flat",
            padx=15,
        ).pack(side="left", padx=(8, 5))

        tk.Button(
            bottom,
            text="Clear",
            command=self.clear_chat,
            bg="#555555",
            fg="white",
            relief="flat",
            padx=15,
        ).pack(side="left")

        self.status = tk.Label(
            self.root,
            text="Ready",
            bg="#1e1e1e",
            fg="lightgreen",
            anchor="w",
            font=("Segoe UI", 14),
        )
        self.status.pack(fill="x", padx=15, pady=(0, 10))

        self.root.after(50, self.spin_ros)
        self.root.mainloop()

    def add_message(self, sender, text):
        self.chat.config(state="normal")

        if sender == "You":
            self.chat.insert(tk.END, "You:\n", "you")
        else:
            self.chat.insert(tk.END, "Skye:\n", "robot")

        self.chat.insert(tk.END, text + "\n\n")

        self.chat.config(state="disabled")
        self.chat.see(tk.END)

    def send_message(self, event=None):
        text = self.entry.get().strip()

        if not text or self.waiting:
            return

        self.add_message("You", text)

        msg = String()
        msg.data = text
        self.ai_pub.publish(msg)

        self.entry.delete(0, tk.END)

        self.waiting = True
        self.status.config(text="Waiting for AI...", fg="orange")

    def ai_callback(self, msg):
        self.waiting = False
        self.status.config(text="Ready", fg="lightgreen")

        try:
            data = json.loads(msg.data)
        except json.JSONDecodeError:
            try:
                data = ast.literal_eval(msg.data)
            except (ValueError, SyntaxError):
                self.get_logger().warn("Received unparseable data on /ai/response")
                self.add_message("Robot", msg.data)
                return

        text = data.get("text", "")
        self.add_message("Robot", text)

        if text:
            subprocess.Popen(["espeak-ng", text])

    def clear_chat(self):
        self.chat.config(state="normal")
        self.chat.delete("1.0", tk.END)
        self.chat.config(state="disabled")

    def spin_ros(self):
        rclpy.spin_once(self, timeout_sec=0)
        self.root.after(50, self.spin_ros)


def main(args=None):
    rclpy.init(args=args)

    RobotUI()

    rclpy.shutdown()


if __name__ == "__main__":
    main()