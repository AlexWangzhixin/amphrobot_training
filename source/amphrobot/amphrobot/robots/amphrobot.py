# Configuration for Amphibious Robot

import os

import isaaclab.sim as sim_utils
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils import configclass

from amphrobot.robots import unitree_actuators


@configclass
class AmphrobotArticulationCfg(ArticulationCfg):
    """Configuration for Amphrobot articulations."""

    joint_sdk_names: list[str] = None

    soft_joint_pos_limit_factor = 0.9


@configclass
class AmphrobotUsdFileCfg(sim_utils.UsdFileCfg):
    activate_contact_sensors: bool = True
    rigid_props = sim_utils.RigidBodyPropertiesCfg(
        disable_gravity=False,
        retain_accelerations=False,
        linear_damping=0.0,
        angular_damping=0.0,
        max_linear_velocity=1000.0,
        max_angular_velocity=1000.0,
        max_depenetration_velocity=1.0,
    )
    articulation_props = sim_utils.ArticulationRootPropertiesCfg(
        enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=4
    )

""" Configuration for the Amphrobot robots."""

AMPHROBOT_CFG = AmphrobotArticulationCfg(
    spawn=AmphrobotUsdFileCfg(
        usd_path="C:\\Users\\zitu1\\Documents\\amphibious\\urdf\\model\\model.usd",
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.10),
        joint_pos={
            "Front_Left_Side_joint": 0.2,
            "Front_Right_Side_joint": -0.2,
            "Hind_Left_Side_joint": -0.1,
            "Hind_Right_Side_joint": 0.1, 
            
            "Front_(Left|Right)_Thigh_joint": 0.8,
            "Hind_(Left|Right)_Thigh_joint": 1.0,
            ".*_Calf_joint": -1.5,
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        "GO2HV": unitree_actuators.UnitreeActuatorCfg_Go2HV(
            joint_names_expr=[".*"],
            stiffness=25.0,
            damping=0.5,
            friction=0.01,
        ),
    },
    # fmt: off
    joint_sdk_names=[
        "Front_Left_Side_joint", "Front_Left_Thigh_joint", "Front_Left_Calf_joint",
        "Front_Right_Side_joint", "Front_Right_Thigh_joint", "Front_Right_Calf_joint",
        "Hind_Left_Side_joint", "Hind_Left_Thigh_joint", "Hind_Left_Calf_joint",
        "Hind_Right_Side_joint", "Hind_Right_Thigh_joint", "Hind_Right_Calf_joint"
    ],
    # fmt: on
)

