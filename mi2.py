import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TwistStamped, PoseStamped, Twist
from sensor_msgs.msg import Imu, Image
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
from cv_bridge import CvBridge
import cv2
import numpy as np
import time

class LineFollowingNode(Node):
    def __init__(self):
        super().__init__('line_following_node')
        self.bridge = CvBridge()

        # Создаем QoS профиль с режимом BEST_EFFORT
        qos_profile = QoSProfile(reliability=QoSReliabilityPolicy.BEST_EFFORT, depth=10)

        # Подписчик на изображение с нижней камеры
        self.camera_sub = self.create_subscription(Image, '/uav1/camera_down', self.camera_callback, 10)
        self.pose_sub = self.create_subscription(PoseStamped, '/uav1/mavros/local_position/pose', self.pose_callback, qos_profile=qos_profile)
        # Подписка на данные IMU
        self.imu_sub = self.create_subscription(Imu, '/uav1/imu/data', self.imu_callback, qos_profile=qos_profile)

        # Подписчик на изображение с передней камеры
        self.front_camera_sub = self.create_subscription(Image, '/uav1/camera', self.front_camera_callback, 10)

        # Публикация команд движения
        self.cmd_vel_pub = self.create_publisher(TwistStamped, '/uav1/mavros/setpoint_velocity/cmd_vel', 10)
        self.local_pos_pub = self.create_publisher(PoseStamped, '/uav1/mavros/setpoint_position/local', 10)

        # Диапазон для черного цвета в пространстве HSV (можно настраивать)
        self.lower_black = np.array([0, 0, 0])
        self.upper_black = np.array([180, 255, 50])  # Поправлен верхний диапазон для черного цвета

        # Диапазон для зеленого цвета в пространстве HSV (можно настраивать)
        self.lower_green = np.array([0, 87, 120])
        self.upper_green = np.array([73, 255, 255])

        # Для хранения данных IMU
        self.imu_data = None

        # Минимальная площадь контура для распознавания линии
        self.min_contour_area = 500  # Настройте это значение по вашему усмотрению

        # Переменные для хранения данных о зеленой рамке и черной линии
        self.green_frame_center_x = None
        self.green_frame_center_y = None
        self.line_center_x = None
        self.angle_to_line = None

        # Переменные для хранения текущей позиции дрона
        self.current_pose = None

        self.get_logger().info('Node initialized, waiting for camera images...')

    def imu_callback(self, imu_msg):
        # Сохраняем данные ориентации IMU
        self.imu_data = imu_msg.orientation
        self.get_logger().info(f"Получены данные IMU: {self.imu_data}")

    def camera_callback(self, image_msg):
        # Преобразование изображения из ROS-сообщения в OpenCV формат
        frame = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding='bgr8')

        # Обрезаем 30% слева и 30% справа
        h, w = frame.shape[:2]
        left_offset = int(w * 0.3)
        right_offset = int(w * 0.7)
        frame = frame[:, left_offset:right_offset]

        # Обнаружение черной линии на обрезанном изображении
        found_line, self.line_center_x, self.angle_to_line = self.detect_black_line(frame)

        if found_line:
            # Корректируем положение дрона, чтобы двигаться вдоль линии
            self.follow_line(self.line_center_x, self.angle_to_line, frame.shape[1] // 2)

        # Визуализация для отладки
        cv2.imshow("Downward Camera - Line Following", frame)
        cv2.waitKey(1)

    def front_camera_callback(self, image_msg):
        # Преобразование изображения из ROS-сообщения в OpenCV формат
        frame = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding='bgr8')

        # Обнаружение зеленой рамки на полном изображении
        found_green_frame, self.green_frame_center_x, self.green_frame_center_y = self.detect_green_frame(frame)

        if found_green_frame:
            self.get_logger().info(f'Зеленая рамка обнаружена: Центр X={self.green_frame_center_x}, Центр Y={self.green_frame_center_y}')
            self.follow_green_frame(frame)

        # Визуализация для отладки
        cv2.imshow("Front Camera - Green Frame Detection", frame)
        cv2.waitKey(1)

    def detect_black_line(self, frame):
        # Преобразуем изображение в пространство HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Применяем фильтр для выделения черного цвета
        mask = cv2.inRange(hsv, self.lower_black, self.upper_black)

        # Поиск контуров на маске
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        line_center_x = 0
        angle_to_line = 0
        if contours:
            # Находим самый большой контур, вероятно это черная линия
            largest_contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest_contour) > self.min_contour_area:
                x, y, w, h = cv2.boundingRect(largest_contour)
                line_center_x = x + w // 2

                # Рассчитываем угол между дроном и линией
                rect = cv2.minAreaRect(largest_contour)
                angle_to_line = rect[2]

                # Рисуем контур для визуализации
                cv2.drawContours(frame, [largest_contour], -1, (0, 255, 0), 2)

                return True, line_center_x, angle_to_line

        return False, 0, 0

    def detect_green_frame(self, frame):
        # Преобразуем изображение в пространство HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Применяем фильтр для выделения зеленого цвета
        mask = cv2.inRange(hsv, self.lower_green, self.upper_green)

        # Поиск контуров на маске
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        green_frame_center_x = 0
        green_frame_center_y = 0
        if contours:
            # Находим самый большой контур, вероятно это зеленая рамка
            largest_contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest_contour) > self.min_contour_area:
                x, y, w, h = cv2.boundingRect(largest_contour)
                green_frame_center_x = x + w // 2
                green_frame_center_y = y + h // 2

                # Рисуем контур для визуализации
                cv2.drawContours(frame, [largest_contour], -1, (0, 255, 0), 2)

                return True, green_frame_center_x, green_frame_center_y

        return False, 0, 0

    def follow_line(self, line_center_x, angle_to_line, frame_center_x):
        # Вычисляем отклонение линии от центра кадра
        deviation = line_center_x - frame_center_x
        tolerance = 20  # Допустимое отклонение от центра

        cmd = TwistStamped()
        cmd.header.stamp = self.get_clock().now().to_msg()

        # Корректируем движение вдоль линии
        if abs(deviation) > tolerance:
            cmd.twist.linear.y = -0.01 * deviation  # Поправка по оси Y для центрирования
            self.get_logger().info(f'Корректировка по Y: {cmd.twist.linear.y}')

        # Корректируем угол поворота
        cmd.twist.angular.z = -0.008 * angle_to_line  # Поправка угла для следования за линией
        self.get_logger().info(f'Корректировка угла: {cmd.twist.angular.z}')

        # Продвигаемся вперед вдоль линии
        cmd.twist.linear.x = 0.4  # Уменьшена скорость для более плавного прохождения

        # Устанавливаем компонент Z в 0 (или в нужное значение)
        cmd.twist.linear.z = 0.0  # Убедитесь, что z-значение определено

        # Публикуем команду движения
        self.cmd_vel_pub.publish(cmd)

        # Встроим вызов setVelLocal:
        self.setVelLocal(cmd.twist.linear.x, cmd.twist.linear.y, cmd.twist.linear.z)

    def setVelLocal(self, x, y, z):
        if self.imu_data is None:
            self.get_logger().warn("Данные IMU еще не получены.")
            return

        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        temptwist = Twist()
        vector_local = np.array([x, y, z])  # Пример вектора в локальных координатах

        # Получаем ориентацию с IMU (предположим, что она доступна)
        quaternion = np.array([self.imu_data.x, self.imu_data.y, self.imu_data.z, self.imu_data.w])

        # Преобразуем локальный вектор в глобальные координаты
        ans = self.transform_vector_local_to_global(vector_local, quaternion)

        # Заполняем данные о линейной скорости
        temptwist.linear.x = ans[0]
        temptwist.linear.y = ans[1]
        temptwist.linear.z = ans[2]

        # Публикуем сообщение
        msg.twist = temptwist
        msg.header.frame_id = 'FRAME_BODY_NED'
        self.cmd_vel_pub.publish(msg)

    def transform_vector_local_to_global(self, vector, quaternion):
        """Transforms a vector from local to global coordinates using a quaternion."""
        vector = np.array(list(vector)[::-1])

        def quaternion_multiply(q1, q2):
            w1, x1, y1, z1 = q1
            w2, x2, y2, z2 = q2
            return np.array([
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
            ])

        def quaternion_conjugate(q):
            return np.array([q[0], -q[1], -q[2], -q[3]])

        # Convert vector to a quaternion with a 0 scalar part
        vector_q = np.array([0] + list(vector))

        # Rotate vector by applying quaternion and its conjugate
        rotated_q = quaternion_multiply(quaternion_multiply(quaternion, vector_q), quaternion_conjugate(quaternion))

        # Return only the vector part of the quaternion
        return (rotated_q[1:])[::-1]

    def pose_callback(self, pose_msg):
        # Сохраняем текущую позицию дрона
        self.current_pose = pose_msg.pose.position

    def take_off(self):
        pose = PoseStamped()
        pose.header.stamp = self.get_clock().now().to_msg()

        pose.pose.position.x = 0.0
        pose.pose.position.y = 0.0
        pose.pose.position.z = 5.0

        for _ in range(0, 100):
            self.local_pos_pub.publish(pose)
            self.get_clock().sleep_for(rclpy.time.Duration(seconds=0.05))
            self.get_logger().info('Команда на взлет отправлена...')

        # Поворот на 180 градусов
        self.rotate(180)

    def rotate(self, angle):
        cmd = TwistStamped()
        cmd.header.stamp = self.get_clock().now().to_msg()

        # Устанавливаем угловую скорость для поворота
        cmd.twist.angular.z = 0.5  # Угловая скорость для поворота

        # Публикуем команду поворота
        start_time = self.get_clock().now()
        while (self.get_clock().now() - start_time).nanoseconds / 1e9 < angle / 30:  # Поворот на 180 градусов за 6 секунд
            self.cmd_vel_pub.publish(cmd)
            self.get_clock().sleep_for(rclpy.time.Duration(seconds=0.1))

        # Останавливаем поворот
        cmd.twist.angular.z = 0.0
        self.cmd_vel_pub.publish(cmd)

        # Движение вперед после поворота
        self.move_forward(2.0)  # Движение вперед на 2 секунды

    def move_forward(self, duration):
        cmd = TwistStamped()
        cmd.header.stamp = self.get_clock().now().to_msg()

        # Устанавливаем линейную скорость для движения вперед
        cmd.twist.linear.x = 0.4  # Скорость движения вперед

        # Публикуем команду движения вперед
        start_time = self.get_clock().now()
        while (self.get_clock().now() - start_time).nanoseconds / 1e9 < duration:
            self.cmd_vel_pub.publish(cmd)
            self.get_clock().sleep_for(rclpy.time.Duration(seconds=0.1))

        # Останавливаем движение вперед
        cmd.twist.linear.x = 0.0
        self.cmd_vel_pub.publish(cmd)

    def follow_green_frame(self, frame):
        if self.green_frame_center_x is not None and self.green_frame_center_y is not None:
            frame_center_x = frame.shape[1] // 2
            frame_center_y = frame.shape[0] // 2

            deviation_x = self.green_frame_center_x - frame_center_x
            deviation_y = self.green_frame_center_y - frame_center_y
            tolerance = 20  # Допустимое отклонение от центра

            cmd = TwistStamped()
            cmd.header.stamp = self.get_clock().now().to_msg()

            # Корректируем движение вдоль линии
            if abs(deviation_x) > tolerance:
                cmd.twist.linear.y = -0.01 * deviation_x  # Поправка по оси Y для центрирования
                self.get_logger().info(f'Корректировка по Y: {cmd.twist.linear.y}')

            if abs(deviation_y) > tolerance:
                cmd.twist.linear.z = -0.01 * deviation_y  # Поправка по оси Z для центрирования
                self.get_logger().info(f'Корректировка по Z: {cmd.twist.linear.z}')

            # Продвигаемся вперед вдоль линии
            cmd.twist.linear.x = 0.4  # Уменьшена скорость для более плавного прохождения

            # Публикуем команду движения
            self.cmd_vel_pub.publish(cmd)

            # Встроим вызов setVelLocal:
            self.setVelLocal(cmd.twist.linear.x, cmd.twist.linear.y, cmd.twist.linear.z)

    def correct_position(self):
        if self.current_pose is not None and self.line_center_x is not None:
            # Вычисляем отклонение дрона от черной линии
            deviation = self.line_center_x - self.current_pose.y
            max_deviation = 0.5  # Максимальное допустимое отклонение в метрах

            if abs(deviation) > max_deviation:
                cmd = TwistStamped()
                cmd.header.stamp = self.get_clock().now().to_msg()

                # Корректируем движение вдоль линии
                cmd.twist.linear.y = -0.01 * deviation  # Поправка по оси Y для центрирования
                self.get_logger().info(f'Корректировка по Y: {cmd.twist.linear.y}')

                # Публикуем команду движения
                self.cmd_vel_pub.publish(cmd)

                # Встроим вызов setVelLocal:
                self.setVelLocal(cmd.twist.linear.x, cmd.twist.linear.y, cmd.twist.linear.z)

def main(args=None):
    rclpy.init(args=args)
    node = LineFollowingNode()
    node.take_off()
    time.sleep(10)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

