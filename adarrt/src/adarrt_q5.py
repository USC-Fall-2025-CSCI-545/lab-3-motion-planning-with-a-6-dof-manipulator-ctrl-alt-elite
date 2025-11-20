#!/usr/bin/env python

import argparse
import time
import random

import adapy
import numpy as np
import rospy


class AdaRRT():
    """
    Rapidly-Exploring Random Trees (RRT) for the ADA controller.
    """
    joint_lower_limits = np.array([-3.14, 1.57, 0.33, -3.14, 0, 0])
    joint_upper_limits = np.array([3.14, 5.00, 5.00, 3.14, 3.14, 3.14])

    class Node():
        """
        A node for a doubly-linked tree structure.
        """
        def __init__(self, state, parent):
            self.state = np.asarray(state)
            self.parent = parent
            self.children = []

        def __iter__(self):
            nodelist = [self]
            while nodelist:
                node = nodelist.pop(0)
                nodelist.extend(node.children)
                yield node

        def __repr__(self):
            return 'Node({})'.format(', '.join(map(str, self.state)))

        def add_child(self, state):
            child = AdaRRT.Node(state=state, parent=self)
            self.children.append(child)
            return child

    def __init__(self,
                 start_state,
                 goal_state,
                 ada,
                 joint_lower_limits=None,
                 joint_upper_limits=None,
                 ada_collision_constraint=None,
                 step_size=0.25,
                 goal_precision=1.0,
                 max_iter=10000):
        self.start = AdaRRT.Node(start_state, None)
        self.goal = AdaRRT.Node(goal_state, None)
        self.ada = ada
        self.joint_lower_limits = joint_lower_limits or AdaRRT.joint_lower_limits
        self.joint_upper_limits = joint_upper_limits or AdaRRT.joint_upper_limits
        self.ada_collision_constraint = ada_collision_constraint
        self.step_size = step_size
        self.goal_precision = goal_precision
        self.max_iter = max_iter

    # ------------------- RRT Methods -------------------
    def build(self):
        """
        Build an RRT.
        """
        nodes = [self.start]

        for k in range(self.max_iter):
            if random.random() < 0.2:
	        sample = self._get_random_sample_near_goal()
	    else:
	        sample = self._get_random_sample()
            neighbor = self._get_nearest_neighbor(sample, nodes)
            new_node = self._extend_sample(sample, neighbor)

            if new_node:
                nodes.append(new_node)
                if self._check_for_completion(new_node):
                    self.goal.parent = new_node
                    new_node.children.append(self.goal)
                    path = self._trace_path_from_start(self.goal)
                    return path

        print("Failed to find path from {0} to {1} after {2} iterations!".format(
            self.start.state, self.goal.state, self.max_iter))
        return None

    def _get_random_sample(self):
        """
        Uniformly sample within joint limits.
        """
        return np.array([random.uniform(low, high) for low, high in zip(
            self.joint_lower_limits, self.joint_upper_limits)])

    def _get_nearest_neighbor(self, sample, nodes):
        """
        Finds the node in nodes closest to the sample.
        """
        min_dist = float('inf')
        nearest = None
        for node in nodes:
            dist = np.linalg.norm(node.state - sample)
            if dist < min_dist:
                min_dist = dist
                nearest = node
        return nearest

    def _extend_sample(self, sample, neighbor):
        """
        Extend neighbor toward sample by step_size.
        """
        direction = sample - neighbor.state
        norm = np.linalg.norm(direction)
        if norm == 0:
            return None

        step = self.step_size * direction / norm
        new_state = neighbor.state + step

        # Clip to joint limits
        new_state = np.minimum(np.maximum(new_state, self.joint_lower_limits),
                               self.joint_upper_limits)

        if not self._check_for_collision(new_state):
            return neighbor.add_child(new_state)
        else:
            return None

    def _check_for_completion(self, node):
        """
        Check if node is close enough to goal.
        """
        return np.linalg.norm(node.state - self.goal.state) <= self.goal_precision

    def _trace_path_from_start(self, node=None):
        """
        Trace path from start to node.
        """
        if node is None:
            node = self.goal

        path = []
        while node is not None:
            path.append(node.state)
            node = node.parent

        path.reverse()
        return path

    def _check_for_collision(self, sample):
        """
        Checks if a sample point is in collision with any collision object.
        """
        if self.ada_collision_constraint is None:
            return False
        return self.ada_collision_constraint.is_satisfied(
            self.ada.get_arm_state_space(),
            self.ada.get_arm_skeleton(), sample)


    def _get_random_sample_near_goal(self, radius=0.05):

        low = self.goal.state - radius
        high = self.goal.state + radius

        # Respect joint limits
        low = np.maximum(low, self.joint_lower_limits)
        high = np.minimum(high, self.joint_upper_limits)

        return np.array([random.uniform(l, h) for l, h in zip(low, high)])


# ------------------- Main -------------------
def main(is_sim):
    
    if not is_sim:
        from moveit_ros_planning_interface._moveit_roscpp_initializer import roscpp_init
        roscpp_init('adarrt', [])

    # instantiate an ada
    ada = adapy.Ada(is_sim)

    armHome = [-1.5, 3.22, 1.23, -2.19, 1.8, 1.2]
    goalConfig = [-1.72, 4.44, 2.02, -2.04, 2.66, 1.39]
    delta = 0.25
    eps = 0.2

    if is_sim:
        ada.set_positions(goalConfig)
    else:
        raw_input("Please move arm to home position with the joystick. " +
            "Press ENTER to continue...")

    # launch viewer
    viewer = ada.start_viewer("dart_markers/simple_trajectories", "map")

    # add objects to world
    canURDFUri = "package://pr_assets/data/objects/can.urdf"
    sodaCanPose = [0.25, -0.35, 0.0, 0, 0, 0, 1]
    tableURDFUri = "package://pr_assets/data/furniture/uw_demo_table.urdf"
    tablePose = [0.3, 0.0, -0.7, 0.707107, 0, 0, 0.707107]
    world = ada.get_world()
    can = world.add_body_from_urdf(canURDFUri, sodaCanPose)
    table = world.add_body_from_urdf(tableURDFUri, tablePose)

    # add collision constraints
    collision_free_constraint = ada.set_up_collision_detection(
            ada.get_arm_state_space(),
            ada.get_arm_skeleton(),
            [can, table])
    full_collision_constraint = ada.get_full_collision_constraint(
            ada.get_arm_state_space(),
            ada.get_arm_skeleton(),
            collision_free_constraint)

    # easy goal
    adaRRT = AdaRRT(
        start_state=np.array(armHome),
        goal_state=np.array(goalConfig),
        ada=ada,
        ada_collision_constraint=full_collision_constraint,
        step_size=delta,
        goal_precision=eps)

    rospy.sleep(1.0)

    if not is_sim:
        ada.start_trajectory_executor()

    path = adaRRT.build()
    if path is not None:
        print("Path waypoints:")
        print(np.asarray(path))
        waypoints = []
        for i, waypoint in enumerate(path):
            waypoints.append((0.0 + i, waypoint))

        t0 = time.clock()
        traj = ada.compute_smooth_joint_space_path(
            ada.get_arm_state_space(), waypoints)
        t = time.clock() - t0
        print(str(t) + "seconds elapsed")
        raw_input('Press ENTER to execute trajectory and exit')
        ada.execute_trajectory(traj)
        rospy.sleep(1.0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sim', dest='is_sim', action='store_true')
    parser.add_argument('--real', dest='is_sim', action='store_false')
    parser.set_defaults(is_sim=True)
    args = parser.parse_args()
    main(args.is_sim)
