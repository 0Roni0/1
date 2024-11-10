import rclpy
from rclpy.node import Node
from mavros_msgs.srv import SetMode, CommandBool
from mavros_msgs.msg import State
from geometry_msgs.msg import PoseStamped

class TestNode(Node):
    def __init__(self):
        super().__init__('offboard_node')

        # Создаем клиента для установки режима и армирования
        self.set_mode_client = self.create_client(SetMode, '/uav1/mavros/set_mode')
        self.arming_client = self.create_client(CommandBool, '/uav1/mavros/cmd/arming')

        # Подписываемся на топик состояния дрона
        self.state_sub = self.create_subscription(State, '/uav1/mavros/state', self.state_cb, 10)

        # Издатель для локальной позиции
        self.local_pos_pub = self.create_publisher(PoseStamped, '/uav1/mavros/setpoint_position/local', 10)

        # Текущее состояние дрона
        self.current_state = None
        self.pose = PoseStamped()  # Целевая позиция
        self.pose.pose.position.x = 0.0
        self.pose.pose.position.y = 0.0 

        self.test_services()

    def state_cb(self, msg):
        self.current_state = msg

    def test_services(self):
        # Ждем подключения дрона
        while not self.current_state or not self.current_state.connected:
            self.get_logger().info('\x1b[33m Ожидание подключения дрона... \x1b[0m')
            rclpy.spin_once(self)

        # Начинаем публикацию сетпоинтов
        self.publish_setpoints()

        # Проверка доступности сервиса армирования
        if self.arming_client.wait_for_service(timeout_sec=5.0):
            arm_req = CommandBool.Request()
            arm_req.value = True
            future = self.arming_client.call_async(arm_req)
            rclpy.spin_until_future_complete(self, future)
            if future.result() and future.result().success:
                self.get_logger().info('\x1b[32m Дрон армирован \x1b[0m')
            else:
                self.get_logger().error('\x1b[31m Ошибка армирования \x1b[0m')
        else:
            self.get_logger().error('\x1b[31m Сервис армирования не доступен \x1b[0m')

        # Установка режима OFFBOARD
        if self.set_mode_client.wait_for_service(timeout_sec=5.0):
            set_mode_req = SetMode.Request()
            set_mode_req.custom_mode = 'OFFBOARD'
            future = self.set_mode_client.call_async(set_mode_req)
            rclpy.spin_until_future_complete(self, future)
            if future.result() and future.result().mode_sent:
                self.get_logger().info('\x1b[32m OFFBOARD режим установлен')
            else:
                self.get_logger().error('\x1b[31m Ошибка установки OFFBOARD режима \x1b[0m')
        else:
            self.get_logger().error('\x1b[31m Сервис установки режима не доступен \x1b[0m')

    def publish_setpoints(self):
        # Публикуем сетпоинты чтобы MavROS принял OFFBOARD
        self.get_logger().info('\x1b[33m  Публикация сетпоинта для режима OFFBOARD \x1b[0m ')
        self.local_pos_pub.publish(self.pose)

def main(args=None):
    rclpy.init(args=args)
    node = TestNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
