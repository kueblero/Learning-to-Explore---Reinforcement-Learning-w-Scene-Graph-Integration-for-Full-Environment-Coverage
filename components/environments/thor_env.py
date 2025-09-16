import math
import os
import platform
import random
import warnings
from collections import defaultdict
from copy import deepcopy

import numpy as np
from PIL import Image
from ai2thor.controller import Controller
from ai2thor.platform import CloudRendering, Linux64

from components.environments.exploration_map import ExplorationMap
from components.graph.global_graph import GlobalSceneGraph
from components.graph.gt_graph import GTGraph
from components.graph.local_graph_builder import LocalSceneGraphBuilder
from components.scripts.generate_gt_graphs import generate_gt_scene_graphs
from components.utils.observation import Observation

warnings.filterwarnings("ignore", message="could not connect to X Display*", category=UserWarning)


class ThorEnv:
    def __init__(self, rho=0.02, scene_number=None, render=False, grid_size=0.25, max_actions=40, additional_images=False):
        super().__init__()
        self.rho = rho
        self.grid_size = grid_size
        self.visibilityDistance = 50  # high value so objects in the frame are always visible, visibility deals with far objects
        self.max_actions = max_actions
        self.additional_images = additional_images
        # On Linux, use the specified 'render' flag; on other platforms, always set render=True (no headless mode support)
        self.render = render if platform.system() == "Linux" else True
        if self.render:
            self.controller = Controller(
                moveMagnitude=self.grid_size,
                grid_size=self.grid_size,
                visibilityDistance=self.visibilityDistance,
                renderDepthImage=additional_images,
                renderSemanticSegmentation=additional_images,
                renderInstanceSegmentation=additional_images,
            )
        else:
            self.controller = Controller(
                moveMagnitude=self.grid_size,
                grid_size=self.grid_size,
                visibilityDistance=self.visibilityDistance,
                platform=CloudRendering,
                renderDepthImage=additional_images,
                renderSemanticSegmentation=additional_images,
                renderInstanceSegmentation=additional_images,
            )
        self.builder = LocalSceneGraphBuilder()
        self.global_sg = GlobalSceneGraph()
        self.state = None
        self.gt_graph = None
        self.viewpoints = defaultdict(set)
        self.last_score = 0.0
        self.step_count = 0
        self.scene_number = 1 if scene_number is None else scene_number
        self.map_origin = None
        self.exploration_map = None
        self.occupancy_map = None
        self.num_orientations = 4
        self.stop_index = self.get_actions().index(["Pass", "Pass"])
        self.agent_state = None

        self.td_center_x = None
        self.td_center_z = None
        self.td_ortho_size = None

    def get_action_dim(self):
        return len(self.get_actions())

    def get_actions(self):
        agent_rotations = ["RotateRight", "RotateLeft", "Pass"]
        movements = ["MoveAhead", "MoveRight", "MoveLeft", "MoveBack", "Pass"]
        return [[move, rot1] for move in movements for rot1 in agent_rotations]

    def get_state_dim(self):
        warnings.warn(
            "The state dimension cannot be reliably determined from the environment. " "This method should not be used.", UserWarning
        )
        return [3, 128, 256]

    def reset(self, scene_number=None, random_start=False, start_position=None, start_rotation=None):
        if scene_number is not None:
            self.scene_number = scene_number
        self.controller.reset(
            scene=f"FloorPlan{self.scene_number}",
            moveMagnitude=self.grid_size,
            grid_size=self.grid_size,
            visibilityDistance=self.visibilityDistance,
        )
        self.builder = LocalSceneGraphBuilder()
        self.global_sg = GlobalSceneGraph()
        self.step_count = 0
        self.last_score = 0.0
        self.viewpoints.clear()

        if random_start:
            reachable = self.safe_step(action="GetReachablePositions").metadata["actionReturn"]
            pos = random.choice(reachable)
            rot = {"x": 0, "y": random.choice([0, 90, 180, 270]), "z": 0}
        elif start_position and start_rotation:
            pos = start_position
            rot = start_rotation
        else:
            pos = None

        if pos is not None:
            self.safe_step(action="Teleport", position=pos, rotation=rot)

        event = self.safe_step(action="Pass")
        rgb = event.frame
        local_sg = self.builder.build_from_metadata(event.metadata)

        agent_view = event.metadata["agent"]
        agent_pos = agent_view["position"]
        agent_rot = agent_view["rotation"]["y"]

        viewpoint = (
            round(agent_pos["x"] / self.grid_size) * self.grid_size,
            round(agent_pos["z"] / self.grid_size) * self.grid_size,
            round(agent_rot / 90) * 90,
        )
        for node in local_sg.nodes.values():
            self.viewpoints[node.object_id].add(viewpoint)
        self.global_sg.add_local_sg(local_sg)

        # Get scene size information (used for estimating a generous map size)
        bounds = event.metadata["sceneBounds"]
        size = bounds["size"]

        # Get agent's starting position
        start_x = agent_pos["x"]
        start_z = agent_pos["z"]

        # Compute extended map dimensions: 2x scene width and height (in grid units)
        map_width = math.ceil((size["x"] * 2) / self.grid_size)
        map_height = math.ceil((size["z"] * 2) / self.grid_size)

        # Force odd size for perfect centering
        if map_width % 2 == 0:
            map_width += 1
        if map_height % 2 == 0:
            map_height += 1

        # Set the map origin so that the agent starts roughly at the center of the occupancy map
        self.map_origin = (start_x - (map_width // 2) * self.grid_size, start_z - (map_height // 2) * self.grid_size)

        # Initialize the occupancy map with zeros (unvisited); shape: [H, W, rotations]
        self.exploration_map = ExplorationMap(grid_size=self.grid_size, map_width=map_width, map_height=map_height, origin=self.map_origin)
        self.exploration_map.update_from_event(event)

        self.occupancy_map = np.zeros((map_height, map_width, self.num_orientations), dtype=np.float32)
        agent_x, agent_z = self._update_occupancy(event)

        self.state = [rgb, local_sg, self.global_sg, self.exploration_map]

        obs = Observation(state=self.state, info={"event": event})
        self._compute_reward(obs)

        self.gt_graph = self.get_ground_truth_graph(f"FloorPlan{self.scene_number}")

        # --- Add Top-Down Camera after reset ---
        self.td_center_x, self.td_center_z, self.td_ortho_size = self.add_topdown_camera_covering_scene()

        return obs

    def add_topdown_camera_covering_scene(self, pad=0.10, desired_hw=None):
        """
        Adds an orthographic top-down third-party camera that covers the reachable scene
        with a small padding. Returns (center_x, center_z, ortho_size).
        - pad: padding (meters) added on both sides in world units
        - desired_hw: optional (H, W) you plan to use; only used to compute aspect if you know it a priori.
        """
        # 1) Get scene bounds (center & size in world units)
        ev = self.safe_step(action="Pass")
        bounds = ev.metadata["sceneBounds"]
        center = bounds["center"]  # dict x,y,z
        size = bounds["size"]  # dict x,y,z

        # 2) Decide aspect ratio for the camera coverage
        #    If you don't know yet, we’ll assume square first, but we will fix it right after by reading the actual frame size.
        if desired_hw is not None:
            H, W = desired_hw
            aspect = W / H
        else:
            aspect = 1.0  # temporary; we’ll re-add the camera if aspect differs after grabbing a frame

        # 3) Compute orthographic size to cover X and Z with padding, respecting aspect
        z_span = size["z"] + 2 * pad
        x_span = size["x"] + 2 * pad
        required_ortho_for_z = 0.5 * z_span
        required_ortho_for_x = 0.5 * x_span / max(1e-6, aspect)
        ortho_size = max(required_ortho_for_z, required_ortho_for_x)

        # 4) Place camera. Height does not affect coverage in orthographic mode.
        top_camera_height = size["y"] - 0.5
        self.safe_step(
            action="AddThirdPartyCamera",
            rotation=dict(x=90, y=0, z=0),  # look straight down
            position=dict(x=center["x"], y=top_camera_height, z=center["z"]),
            orthographic=True,
            orthographicSize=ortho_size,
            # fieldOfView is ignored in orthographic mode
            # keep defaults for near/far; increase far if you ever hide tall content
        )

        # 5) Read actual frame to get the true aspect; if it differs, recompute once.
        ev2 = self.safe_step(action="Pass")
        frame = ev2.third_party_camera_frames[0]
        H, W, _ = frame.shape
        true_aspect = W / H
        if abs(true_aspect - aspect) > 1e-6:
            # Recompute & re-add the camera with correct aspect-derived ortho_size.
            required_ortho_for_x = 0.5 * x_span / max(1e-6, true_aspect)
            ortho_size = max(required_ortho_for_z, required_ortho_for_x)
            self.safe_step(
                action="AddThirdPartyCamera",
                rotation=dict(x=90, y=0, z=0),
                position=dict(x=center["x"], y=top_camera_height, z=center["z"]),
                orthographic=True,
                orthographicSize=ortho_size,
            )
        return center["x"], center["z"], ortho_size

    def get_ground_truth_graph(self, floorplan_name: str):
        """
        Loads or generates and returns the full ground-truth scene graph for a given floorplan.
        If no saved graph exists, it will be generated and saved automatically.
        """
        save_path = os.path.join(os.path.dirname(__file__), "..", "data", "gt_graphs", f"{floorplan_name}.json")

        # Generate if not exists
        if not os.path.exists(save_path):
            print(f"⚠️ GT Graph for {floorplan_name} not found. Generating...")
            generate_gt_scene_graphs(floorplans=[floorplan_name])

        return GTGraph().load_from_file(save_path)

    def safe_step(self, *args, **kwargs):
        try:
            self.agent_state = self.get_agent_state()
            return self.controller.step(*args, **kwargs)
        except TimeoutError as e:
            print(f"[TIMEOUT] Action '{kwargs.get('action', 'unknown')}' timed out. Restarting environment.")
            self.reset_hard()
            return self.controller.step(*args, **kwargs)

    def reset_hard(self):
        try:
            self.controller.stop()
        except Exception as e:
            print(f"[WARN] Failed to stop controller cleanly: {e}")

        if self.render:
            self.controller = Controller(
                moveMagnitude=self.grid_size,
                grid_size=self.grid_size,
                visibilityDistance=self.visibilityDistance,
                renderDepthImage=self.additional_images,
                renderSemanticSegmentation=self.additional_images,
                renderInstanceSegmentation=self.additional_images,
            )
        else:
            self.controller = Controller(
                moveMagnitude=self.grid_size,
                grid_size=self.grid_size,
                visibilityDistance=self.visibilityDistance,
                platform=CloudRendering,
                renderDepthImage=self.additional_images,
                renderSemanticSegmentation=self.additional_images,
                renderInstanceSegmentation=self.additional_images,
            )
        self.reset(scene_number=self.scene_number)
        self.restore_agent_state(self.agent_state)

    def step(self, action):
        actions = self.get_actions()[action]
        error_msgs = {}
        all_success = True
        for primitive_action in actions:
            if "Move" in primitive_action:
                event = self.safe_step(action=primitive_action, moveMagnitude=self.grid_size)
            else:
                event = self.safe_step(action=primitive_action)
            success = event.metadata["lastActionSuccess"]
            if not success:
                error_msgs[primitive_action] = event.metadata["errorMessage"]
                all_success = False
        event = self.safe_step(action="Pass")

        self.exploration_map.update_from_event(event)
        self.exploration_map.mark_discoveries(event, self.global_sg)
        agent_x, agent_z = self._update_occupancy(event)

        rgb = event.frame
        local_sg = self.builder.build_from_metadata(event.metadata)
        self.global_sg.add_local_sg(local_sg)

        self.state = [rgb, local_sg, self.global_sg, self.exploration_map]

        agent_view = event.metadata["agent"]
        agent_pos = agent_view["position"]
        agent_rot = agent_view["rotation"]["y"]

        viewpoint = (
            round(agent_pos["x"] / self.grid_size) * self.grid_size,
            round(agent_pos["z"] / self.grid_size) * self.grid_size,
            round(agent_rot / 90) * 90,
        )

        # Update viewpoint tracking
        for node in local_sg.nodes.values():
            self.viewpoints[node.object_id].add(viewpoint)

        self.step_count += 1

        truncated = action == self.stop_index or self.step_count >= self.max_actions
        terminated = (
            len([k for k, n in self.global_sg.nodes.items() if n.visibility >= 0.8]) == len(self.gt_graph.nodes)
            and action == self.stop_index
        )

        if terminated:
            truncated = False
        obs = Observation(state=self.state, truncated=truncated, terminated=terminated)

        score, recall_node, recall_edge = self.compute_score(obs)

        obs.info = {
            "event": event,
            "score": score,
            "recall_node": recall_node,
            "recall_edge": recall_edge,
            "action": action,
            "agent_pos": (agent_x, agent_z),
            "allActionsSuccess": all_success,
            "errorMessages": error_msgs,
            "max_steps_reached": self.step_count >= self.max_actions,
        }

        obs.reward = self._compute_reward(obs)
        return obs

    def compute_score(self, obs):
        """
        Computes score based on discovered objects and termination status.
        Also returns recall for nodes and edges.
        """
        num_gt_objects = len(self.gt_graph.nodes)
        discovered_nodes = [n for n in self.global_sg.nodes.values() if n.visibility >= 0.8]
        num_discovered = len(discovered_nodes)
        # Recall for nodes
        recall_node = num_discovered / num_gt_objects if num_gt_objects > 0 else 0.0

        # Compute edge recall
        num_gt_edges = len(self.gt_graph.edges)
        num_discovered_edges = len(self.global_sg.edges) if hasattr(self.global_sg, "edges") else 0
        recall_edge = num_discovered_edges / num_gt_edges if num_gt_edges > 0 else 0.0

        termination_bonus = 0.0 if obs.terminated else 0.0
        score = recall_node + termination_bonus

        return score, recall_node, recall_edge

    def get_occupancy_indices(self, event):
        # Extract agent's current position and orientation
        pos = event.metadata["agent"]["position"]
        rot_y = event.metadata["agent"]["rotation"]["y"]

        x, z = pos["x"], pos["z"]

        # Compute offset from map origin
        dx = x - self.map_origin[0]
        dz = z - self.map_origin[1]

        i = self.occupancy_map.shape[0] - 1 - int(dz / self.grid_size)
        j = int(dx / self.grid_size)

        # Quantize rotation
        rot_idx = int(round(rot_y / 90.0)) % self.num_orientations

        return i, j, rot_idx

    def _update_occupancy(self, event):
        i, j, rot_idx = self.get_occupancy_indices(event)

        assert 0 <= rot_idx < self.num_orientations, f"Invalid rotation index: {rot_idx}"
        assert 0 <= i < self.occupancy_map.shape[0], f"Invalid i index: {i}"
        assert 0 <= j < self.occupancy_map.shape[1], f"Invalid j index: {j}"

        self.occupancy_map[i, j, rot_idx] = 1.0

        return i, j

    def _compute_reward(self, obs):
        # Parameters as set in the original paper
        lambda_node = 0.1
        lambda_p = 0.5
        lambda_d = 0.001
        rho = self.rho

        # Recall for nodes and edges, extracted from the current global scene graph
        Rnode = obs.info.get("recall_node", 0.0)  # Recall for nodes
        Redge = obs.info.get("recall_edge", 0.0)  # Recall for edges

        if hasattr(self.global_sg, "nodes") and self.global_sg.nodes:
            Pnode = np.mean([n.visibility for n in self.global_sg.nodes.values()])
        else:
            Pnode = 0.0

        Pedge = 1.0

        # Diversity: sum of unique viewpoints for all objects
        diversity = sum(len(v) for v in self.viewpoints.values())

        # Compute similarity score between generated and ground truth scene graph
        sim = lambda_node * (Rnode + lambda_p * Pnode) + Redge + lambda_p * Pedge

        # Overall score at the current step, as in the paper
        score = sim + lambda_d * diversity - rho * self.step_count

        # Reward is the change in score since the last step (dense reward)
        reward = score - self.last_score
        self.last_score = score

        return reward

    def get_agent_state(self):
        agent = self.controller.last_event.metadata["agent"]
        return {"position": agent["position"], "rotation": agent["rotation"]}

    def restore_agent_state(self, agent_state):
        self.controller.step(action="Teleport", position=agent_state["position"], rotation=agent_state["rotation"], horizon=0)
        self.controller.step(action="Pass")

    def get_env_state(self):
        return {
            "state": deepcopy(self.state),
            "global_sg": deepcopy(self.global_sg),
            "exploration_map": deepcopy(self.exploration_map),
            "viewpoints": deepcopy(self.viewpoints),
            "last_score": self.last_score,
            "step_count": self.step_count,
        }

    def restore_env_state(self, env_state):
        self.state = deepcopy(env_state["state"])
        self.global_sg = deepcopy(env_state["global_sg"])
        self.exploration_map = deepcopy(env_state["exploration_map"])
        self.viewpoints = deepcopy(env_state["viewpoints"])
        self.last_score = env_state["last_score"]
        self.step_count = env_state["step_count"]

    def try_action(self, action, agent_pos=None, agent_rot=None):
        env_state = self.get_env_state()
        agent_state = self.get_agent_state()
        if agent_pos is not None and agent_rot is not None:
            event = self.safe_step(action="Teleport", position=agent_pos, rotation=dict(x=0, y=agent_rot, z=0))
        event = self.safe_step(action=action)
        self.restore_env_state(env_state)
        self.restore_agent_state(agent_state)
        return event.metadata["lastActionSuccess"]

    def get_top_down_view(self):
        """
        Returns the current top-down view image from the third-party camera as a numpy array (H,W,3).
        """
        event = self.safe_step(action="Pass")
        if hasattr(event, "third_party_camera_frames") and event.third_party_camera_frames:
            return event.third_party_camera_frames[0]
        else:
            raise RuntimeError("No third-party camera frames found.")

    def visualize_shortest_path(self, start, target):
        """
        Visualizes the shortest path from start to goal on the current top-down view.
        - start, goal: dicts with 'x', 'y', 'z'
        """
        if len(target) == 2:
            target = {"x": target["x"], "y": start["y"], "z": target["z"]}
        event = self.safe_step(action="GetShortestPathToPoint", position=start, target=target)
        path = event.metadata["actionReturn"]["corners"]  # list of dicts

        event = self.safe_step(action="VisualizePath", positions=path, grid=False, endText="Target")
        if hasattr(event, "third_party_camera_frames") and event.third_party_camera_frames:
            arr = event.third_party_camera_frames[0]
            return Image.fromarray(arr)
        return None

    def close(self):
        self.controller.stop()
