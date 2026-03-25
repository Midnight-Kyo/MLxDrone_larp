#!/usr/bin/env python3
"""
ROS2 gesture-to-drone bridge node (WSL2 side).

Receives gesture commands over TCP from the Windows gesture_bridge.py and
publishes them as ROS2 messages to control the Tello drone in Gazebo.

Usage (in WSL2 terminal, with ROS2 + tello workspace sourced):
    python3 gesture_ros2_node.py

The Tello Gazebo plugin has 4 flight states:
    landed → taking_off → flying → landing → landed

Key rules from the plugin source:
    - takeoff only accepted in 'landed' state
    - land only accepted in 'flying' state
    - cmd_vel only processed in 'flying' state
    - taking_off lasts ~2s (until altitude > 1.0m)
    - landing lasts ~3s (until altitude < 0.1m)
    - any command during a transition returns rc=3 (BUSY)
"""

import json
import socket
import threading
import time

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from tello_msgs.srv import TelloAction

from search_behavior import (
    EPS_X,
    H_HOLD,
    KP_LOCK,
    MAX_ANG_LOCK,
    M_ACQUIRE,
    M_LOSS,
    OMEGA_SEARCH,
)

TCP_HOST = "0.0.0.0"
TCP_PORT = 9090

# Twist linear.* in m/s (plugin scale). Keep low for first Gazebo / indoor tests.
VELOCITY_FORWARD = 0.12
VELOCITY_UP = 0.10
PUBLISH_RATE = 10.0

RC_OK = 1
RC_BUSY = 3


class GestureDroneNode(Node):
    def __init__(self):
        super().__init__("gesture_drone_bridge")

        self.cmd_pub = self.create_publisher(Twist, "/drone1/cmd_vel", 10)
        self.tello_client = self.create_client(TelloAction, "/drone1/tello_action")

        self.current_command = "IDLE"
        self.last_recv_time = time.time()
        self.lock = threading.Lock()

        # Autonomous SEARCH / FACE_LOCK (v1); TCP carries fist_edge + face_ok + face_x_norm
        self.beh_state = "MANUAL"  # MANUAL | SEARCH | FACE_LOCK
        self.acq_streak = 0
        self.loss_streak = 0
        self.hold_streak = 0
        self.autonomy_angular_z = 0.0
        self.last_face_ok = False
        self.last_face_x_norm = 0.0

        # Mirrors the Tello plugin's flight state machine
        # landed | taking_off | flying | landing
        self.flight_state = "landed"
        self.action_pending = False

        self.timer = self.create_timer(1.0 / PUBLISH_RATE, self.publish_cmd_vel)

        self.tcp_thread = threading.Thread(target=self.tcp_server, daemon=True)
        self.tcp_thread.start()

        self.get_logger().info(f"Gesture bridge node started, TCP server on {TCP_HOST}:{TCP_PORT}")
        self.get_logger().info(
            f"cmd_vel (flying): forward={VELOCITY_FORWARD} m/s, up={VELOCITY_UP} m/s @ {PUBLISH_RATE} Hz"
        )
        self.get_logger().info("Waiting for gesture_bridge.py connection from Windows...")

    def tcp_server(self):
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((TCP_HOST, TCP_PORT))
        server.listen(1)
        server.settimeout(None)

        while rclpy.ok():
            self.get_logger().info("TCP: waiting for connection...")
            try:
                conn, addr = server.accept()
                self.get_logger().info(f"TCP: connected from {addr}")
                self.handle_client(conn)
                self.get_logger().warn(f"TCP: client {addr} disconnected")
            except Exception as e:
                self.get_logger().error(f"TCP error: {e}")
                time.sleep(1)

    def handle_client(self, conn):
        buffer = ""
        conn.settimeout(2.0)

        while rclpy.ok():
            try:
                data = conn.recv(4096)
                if not data:
                    break
                buffer += data.decode("utf-8")

                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        msg = json.loads(line)
                        snap = self.process_command(msg)
                        if snap:
                            try:
                                conn.sendall((json.dumps(snap) + "\n").encode("utf-8"))
                            except (BrokenPipeError, ConnectionResetError, OSError):
                                break
                    except json.JSONDecodeError:
                        self.get_logger().warn(f"Invalid JSON: {line[:80]}")

            except socket.timeout:
                continue
            except (ConnectionResetError, BrokenPipeError):
                break

        conn.close()

    def _reset_beh_manual(self):
        self.beh_state = "MANUAL"
        self.acq_streak = 0
        self.loss_streak = 0
        self.hold_streak = 0
        self.autonomy_angular_z = 0.0

    def _update_search_behavior_locked(self, fist_edge, face_ok, face_x_norm):
        flying = self.flight_state == "flying"
        if not flying:
            if self.beh_state != "MANUAL":
                self._reset_beh_manual()
            return

        if fist_edge:
            if self.beh_state == "MANUAL":
                self.beh_state = "SEARCH"
                self.acq_streak = 0
                self.loss_streak = 0
                self.hold_streak = 0
                self.autonomy_angular_z = OMEGA_SEARCH
            elif self.beh_state in ("SEARCH", "FACE_LOCK"):
                self._reset_beh_manual()

        if self.beh_state == "SEARCH":
            if face_ok:
                self.acq_streak += 1
                self.loss_streak = 0
                if self.acq_streak >= M_ACQUIRE:
                    self.beh_state = "FACE_LOCK"
                    self.acq_streak = 0
                    self.loss_streak = 0
                    self.hold_streak = 0
            else:
                self.acq_streak = 0
            self.autonomy_angular_z = OMEGA_SEARCH
        elif self.beh_state == "FACE_LOCK":
            if face_ok:
                self.loss_streak = 0
                ex = face_x_norm
                if abs(ex) < EPS_X:
                    self.hold_streak += 1
                    if self.hold_streak >= H_HOLD:
                        self.autonomy_angular_z = 0.0
                    else:
                        wz = -KP_LOCK * ex
                        self.autonomy_angular_z = max(
                            -MAX_ANG_LOCK, min(MAX_ANG_LOCK, wz)
                        )
                else:
                    self.hold_streak = 0
                    wz = -KP_LOCK * ex
                    self.autonomy_angular_z = max(
                        -MAX_ANG_LOCK, min(MAX_ANG_LOCK, wz)
                    )
            else:
                self.loss_streak += 1
                self.hold_streak = 0
                self.autonomy_angular_z = 0.0
                if self.loss_streak >= M_LOSS:
                    self.beh_state = "SEARCH"
                    self.acq_streak = 0
                    self.loss_streak = 0
                    self.hold_streak = 0
                    self.autonomy_angular_z = OMEGA_SEARCH

    def _beh_debug_snapshot(self):
        return {
            "type": "beh_debug",
            "beh_state": self.beh_state,
            "acq_streak": self.acq_streak,
            "M_acquire": M_ACQUIRE,
            "loss_streak": self.loss_streak,
            "M_loss": M_LOSS,
            "face_ok": self.last_face_ok,
            "face_x_norm": round(self.last_face_x_norm, 4),
            "autonomy_yaw": round(self.autonomy_angular_z, 4),
            "flying": self.flight_state == "flying",
        }

    def process_command(self, msg):
        command = msg.get("command", "IDLE")
        gesture = msg.get("gesture", "")
        confidence = float(msg.get("confidence", 0.0))
        fist_edge = bool(msg.get("fist_edge", False))
        face_ok = bool(msg.get("face_ok", False))
        face_x_norm = float(msg.get("face_x_norm", 0.0))

        need_takeoff = False
        need_land = False

        with self.lock:
            old = self.current_command
            self.current_command = command
            self.last_recv_time = time.time()
            self.last_face_ok = face_ok
            self.last_face_x_norm = face_x_norm
            self._update_search_behavior_locked(fist_edge, face_ok, face_x_norm)

            if not self.action_pending:
                if command in ("MOVE_FORWARD", "MOVE_UP") and self.flight_state == "landed":
                    need_takeoff = True
                elif command == "LAND" and self.flight_state == "flying":
                    need_land = True
                    self._reset_beh_manual()

            snap = self._beh_debug_snapshot()

        if command != old:
            self.get_logger().info(
                f"Command: {command} (gesture={gesture}, conf={confidence:.0%}) | "
                f"beh={snap['beh_state']} [drone: {self.flight_state}]"
            )

        if need_takeoff:
            self.call_tello_action("takeoff")
        elif need_land:
            self.call_tello_action("land")

        return snap

    def call_tello_action(self, action_cmd):
        if not self.tello_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().error("tello_action service not available!")
            return

        self.action_pending = True
        request = TelloAction.Request()
        request.cmd = action_cmd
        future = self.tello_client.call_async(request)

        self.get_logger().info(f"Tello action: {action_cmd} (state: {self.flight_state})")

        future.add_done_callback(lambda f: self._on_action_done(f, action_cmd))

    def _on_action_done(self, future, action_cmd):
        self.action_pending = False
        result = future.result()
        if result is None:
            self.get_logger().error(f"Tello {action_cmd} call failed (no result)")
            return

        rc = result.rc
        self.get_logger().info(f"Tello response: {action_cmd} rc={rc}")

        if rc == RC_OK:
            if action_cmd == "takeoff":
                self.flight_state = "taking_off"
                self.get_logger().info("State → taking_off (will be 'flying' in ~2s)")
                threading.Timer(2.5, self._finish_takeoff).start()
            elif action_cmd == "land":
                self.flight_state = "landing"
                self.get_logger().info("State → landing (will be 'landed' in ~3s)")
                threading.Timer(3.5, self._finish_landing).start()
        elif rc == RC_BUSY:
            self.get_logger().warn(
                f"Tello busy (rc=3), can't {action_cmd} during '{self.flight_state}'"
            )

    def _finish_takeoff(self):
        if self.flight_state == "taking_off":
            self.flight_state = "flying"
            self.get_logger().info("State → flying (cmd_vel now active)")

    def _finish_landing(self):
        if self.flight_state == "landing":
            self.flight_state = "landed"
            self.get_logger().info("State → landed")

    def publish_cmd_vel(self):
        with self.lock:
            command = self.current_command
            staleness = time.time() - self.last_recv_time
            beh = self.beh_state
            az_auto = self.autonomy_angular_z
            if staleness > 3.0:
                self._reset_beh_manual()
                command = "STOP"
                beh = "MANUAL"
                az_auto = 0.0

        twist = Twist()

        if self.flight_state != "flying":
            self.cmd_pub.publish(twist)
            return

        if beh in ("SEARCH", "FACE_LOCK"):
            twist.linear.x = 0.0
            twist.linear.z = 0.0
            twist.angular.z = float(az_auto)
            self.cmd_pub.publish(twist)
            return

        if command == "MOVE_FORWARD":
            twist.linear.x = VELOCITY_FORWARD
        elif command == "MOVE_UP":
            twist.linear.z = VELOCITY_UP
        elif command in ("FOLLOW_ARM", "IDLE", "STOP"):
            pass  # zero twist (hover)
        elif command == "LAND":
            if not self.action_pending:
                self.call_tello_action("land")

        self.cmd_pub.publish(twist)


def main():
    rclpy.init()
    node = GestureDroneNode()

    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    except Exception:
        pass
    finally:
        try:
            node.get_logger().info("Shutting down gesture bridge node")
        except Exception:
            pass
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
