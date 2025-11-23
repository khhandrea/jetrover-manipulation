from servo_controller_msgs.msg import ServosPosition

from manipulation.utils.grasping_base import GraspingNodeBase
from manipulation.utils.action_group_controller import ActionGroupController


ACTION_GROUP_SAVE_DIR = '/home/ubuntu/software/arm_pc/ActionGroups'


class ManipulationNode(GraspingNodeBase):
    def __init__(self, name):
        super().__init__(name)
        joints_pub = self.create_publisher(ServosPosition, 'servo_controller', 1)
        self.controller = ActionGroupController(joints_pub, ACTION_GROUP_SAVE_DIR)

    def execute_action_group(self, name):
        self.controller.run_action(name)
