import rclpy
from rclpy.node import Node
from mavros_msgs.srv import SetMode
from geometry_msgs.msg import TwistStamped, PoseStamped
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import time
import numpy as np
import threading
import math

class TestNode(Node):
    def __init__(self):
        super().__init__('test_node')

        self.target_altitude = 1.0  # Целевая высота зависания над фигурой
        self.figure_area_threshold = 4000  # Площадь фигуры, при которой дрон останавливается

        # Создаем QoS профиль с режимом BEST_EFFORT
        qos_profile = QoSProfile(reliability=QoSReliabilityPolicy.BEST_EFFORT, depth=10)

        # Инициализация сервисов и подписчиков
        self.set_mode_client = self.create_client(SetMode, '/uav1/mavros/set_mode')
        self.bridge = CvBridge()
        self.camera_sub = self.create_subscription(Image, '/uav1/camera_down', self.camera_callback, 10)
        self.pose_sub = self.create_subscription(PoseStamped, '/uav1/mavros/local_position/pose', self.pose_callback, qos_profile=qos_profile)
        self.cmd_vel_pub = self.create_publisher(TwistStamped, '/uav1/mavros/setpoint_velocity/cmd_vel', 10)
        self.local_pos_pub = self.create_publisher(PoseStamped, '/uav1/mavros/setpoint_position/local', 10)

        # Создание потока для обработки кадров
        self.frame_queue = None
        self.processing_thread = threading.Thread(target=self.process_frame_thread)
        self.processing_thread.daemon = True
        self.processing_thread.start()

        self.drone_position = None  # Для хранения текущей позиции дрона
        self.cube_position = None   # Координаты куба
        self.cube_distance = None   # Дистанция до куба
        self.searched_cube = False
        self.start_position = None  # Начальная позиция дрона

    def take_off(self):
        # Функция для взлета дрона
        self.get_logger().info('Попытка взлета...')
        pose = PoseStamped()
        pose.pose.position.z = 4.0  # Набор высоты
        pose.pose.position.x = 0.0
        pose.pose.position.y = 0.0
        pose.header.stamp = self.get_clock().now().to_msg()

        # Публикуем команду взлета
        for _ in range(100):
            self.local_pos_pub.publish(pose)
            self.get_clock().sleep_for(rclpy.time.Duration(seconds=0.05))
        self.get_logger().info('Команда на взлет отправлена...')

        cmd = TwistStamped()
        cmd.header.stamp = self.get_clock().now().to_msg()
        if self.searched_cube == False:
            cmd.twist.linear.y = 0.5
            self.get_logger().info("Режим блуждание")
        else:
            self.get_logger().info("НЕ режим блуждания...")
            cmd.twist.linear.y = 0.0
        self.cmd_vel_pub.publish(cmd)

    def camera_callback(self, image_msg):
        # Обработка входящих изображений от камеры
        frame = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding='bgr8')
        self.frame_queue = frame  # Отправляем кадр в очередь для обработки

    def pose_callback(self, pose_msg):
        # Обновление позиции дрона
        self.drone_position = (pose_msg.pose.position.x, pose_msg.pose.position.y, pose_msg.pose.position.z)
        if self.start_position is None:
            self.start_position = self.drone_position

    def process_frame_thread(self):
        while True:
            if self.frame_queue is None:
                continue

            frame = self.frame_queue
            self.frame_queue = None  # Очищаем очередь после обработки

            # Обнаружение цветной фигуры на изображении
            figure_detected, figure_center, figure_area, figure_color = self.detect_figure(frame)

            if figure_detected:
                self.searched_cube = True
                self.get_logger().info('Обнаружена цветная фигура')
                self.get_logger().info(f'Площадь фигуры: {figure_area}')
                self.get_logger().info(f'Цвет фигуры: {figure_color}')

                # Если фигура достаточно велика (дрон достаточно близко), он останавливается
                #if figure_area >= self.figure_area_threshold:
                    #self.get_logger().info('Дрон останавливается над фигурой')
                    #self.hover_over_figure()
                    #continue

                # Если фигура еще не достигла нужного размера, корректируем траекторию
                self.control_drone(figure_center, frame.shape[1] // 2, frame.shape[0] // 2)

            if self.drone_position is not None:
                x, y, z = self.drone_position
                coords_text = f"Drone Position: X={x:.2f}, Y={y:.2f}, Z={z:.2f}"
                cv2.putText(frame, coords_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            else:
                self.get_logger().warn("Drone position data not available yet.")

            # Показываем обновленное изображение
            cv2.imshow("Drone Camera Down", frame)
            cv2.waitKey(1)

    def detect_figure(self, frame):
        # Обнаружение фигуры определенного цвета на изображении
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Настройки для поиска красного цвета
        lower_red1 = np.array([0, 74, 50], dtype=np.uint8)
        upper_red1 = np.array([3, 255, 255], dtype=np.uint8)
        lower_red2 = np.array([170, 74, 50], dtype=np.uint8)
        upper_red2 = np.array([180, 255, 255], dtype=np.uint8)
        mask_red1 = cv2.inRange(hsv_frame, lower_red1, upper_red1)
        mask_red2 = cv2.inRange(hsv_frame, lower_red2, upper_red2)
        mask_red = cv2.bitwise_or(mask_red1, mask_red2)

        # Настройки для поиска желтого цвета
        lower_yellow = np.array([20, 100, 100], dtype=np.uint8)
        upper_yellow = np.array([30, 255, 255], dtype=np.uint8)
        mask_yellow = cv2.inRange(hsv_frame, lower_yellow, upper_yellow)

        # Настройки для поиска синего цвета
        lower_blue = np.array([100, 100, 100], dtype=np.uint8)
        upper_blue = np.array([130, 255, 255], dtype=np.uint8)
        mask_blue = cv2.inRange(hsv_frame, lower_blue, upper_blue)

        # Объединение масок
        mask = cv2.bitwise_or(mask_red, mask_yellow)
        mask = cv2.bitwise_or(mask, mask_blue)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            if cv2.contourArea(contour) > 100:  # Игнорируем мелкие контуры
                x, y, w, h = cv2.boundingRect(contour)
                center_x = x + w // 2
                center_y = y + h // 2
                area = w * h

                # Определение цвета фигуры
                figure_color = self.get_color(hsv_frame, contour)

                # Рисуем контуры для визуализации
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.circle(frame, (center_x, center_y), 5, (255, 0, 0), -1)

                return True, (center_x, center_y), area, figure_color

        return False, (0, 0), 0, None

    def get_color(self, hsv_frame, contour):
        # Определение цвета фигуры
        mask = np.zeros(hsv_frame.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, -1)
        mean_color = cv2.mean(hsv_frame, mask=mask)

        if mean_color[0] < 10 or mean_color[0] > 170:
            return "Red"
        elif mean_color[0] < 30:
            return "Yellow"
        elif mean_color[0] < 130:
            return "Blue"
        else:
            return "Unknown"

    def control_drone(self, figure_center, frame_center_x, frame_center_y):
        # Функция для управления дроном в направлении центра фигуры
        self.get_logger().info('Корректировка положения дрона...')
        delta_x = figure_center[0] - frame_center_x
        delta_y = figure_center[1] - frame_center_y
        tolerance = 50  # Допустимое отклонение для центрирования

        cmd = TwistStamped()
        cmd.header.stamp = self.get_clock().now().to_msg()

        x, y, z = self.drone_position

        if z > 4.0:
            cmd.twist.linear.z = -0.1
            self.get_logger().info(f"Корректировка дрона вниз... {z}")
        elif z < 4.0:
            cmd.twist.linear.z = 0.1
            self.get_logger().info(f"Корректировка дрона вверх... {z}")

        # Регулируем движение в зависимости от смещения фигуры
        if abs(delta_x) > tolerance:
            cmd.twist.linear.y = -0.001 * delta_x  # Поправка по оси Y
            self.get_logger().info(f'Смещение по Y: {delta_x}')
        if abs(delta_y) > tolerance:
            cmd.twist.linear.x = 0.001 * delta_y  # Поправка по оси X
            self.get_logger().info(f'Смещение по X: {delta_y}')

        # Если дрон находится над центром фигуры
        if abs(delta_x) <= tolerance and abs(delta_y) <= tolerance and self.cube_position == None:
            self.cube_position = self.drone_position
            self.cube_distance = math.sqrt(pow(x, 2) + pow(y, 2))
            self.get_logger().error(f'Дрон над фигурой, позиция дрона {self.cube_position}, расстояние до куба: {self.cube_distance}')

        self.cmd_vel_pub.publish(cmd)

    def hover_over_figure(self):
        # Остановка дрона над центром фигуры
        stop_cmd = TwistStamped()
        stop_cmd.header.stamp = self.get_clock().now().to_msg()
        stop_cmd.twist.linear.x = 0.0
        stop_cmd.twist.linear.y = 0.0
        stop_cmd.twist.linear.z = 0.0
        self.cmd_vel_pub.publish(stop_cmd)

    def land(self):
        # Функция для посадки дрона
        self.get_logger().info('Посадка...')
        if self.set_mode_client.wait_for_service(timeout_sec=5.0):
            set_mode_req = SetMode.Request()
            set_mode_req.custom_mode = 'AUTO.LAND'
            future = self.set_mode_client.call_async(set_mode_req)
            rclpy.spin_until_future_complete(self, future)
            if future.result() and future.result().mode_sent:
                self.get_logger().info('Режим посадки установлен')
            else:
                self.get_logger().error('Не удалось установить режим посадки')
        else:
            self.get_logger().error('Сервис установки режима недоступен')

def main(args=None):
    rclpy.init(args=args)
    node = TestNode()
    node.take_off()
    time.sleep(5) # чтобы немного подождал после взлета
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

