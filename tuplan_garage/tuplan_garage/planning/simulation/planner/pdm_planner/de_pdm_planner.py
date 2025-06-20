from typing import List, Optional, cast
import gc
import os
import time
import io
import copy

import numpy as np
from scipy.interpolate import interp1d
from omegaconf import OmegaConf
import torch
from PIL import Image
import cv2
from matplotlib import colormaps
import matplotlib.pyplot as plt
from shapely import Point
import numpy.typing as npt

from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import TimePoint
from nuplan.common.maps.abstract_map_objects import LaneGraphEdgeMapObject
from nuplan.common.maps.nuplan_map.lane import NuPlanLane
from nuplan.common.maps.nuplan_map.lane_connector import NuPlanLaneConnector
from nuplan.planning.simulation.planner.abstract_planner import PlannerInput
from nuplan.planning.simulation.trajectory.interpolated_trajectory import (
    InterpolatedTrajectory,
)
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from nuplan.planning.simulation.planner.ml_planner.model_loader import ModelLoader
from nuplan.planning.script.builders.model_builder import build_torch_module_wrapper
from nuplan.planning.training.modeling.lightning_module_wrapper import LightningModuleWrapper
from nuplan.planning.training.preprocessing.features.trajectory import Trajectory
from nuplan.planning.simulation.planner.ml_planner.transform_utils import transform_predictions_to_states
from nuplan.planning.training.callbacks.utils.visualization_utils import visualize
from nuplan.common.maps.maps_datatypes import SemanticMapLayer
from nuplan.common.actor_state.tracked_objects_types import (
    TrackedObjectType,
)


from tuplan_garage.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import (
        PDMScorer,
    )
from tuplan_garage.planning.simulation.planner.pdm_planner.abstract_pdm_planner import (
    AbstractPDMPlanner,
)
from tuplan_garage.planning.simulation.planner.pdm_planner.observation.pdm_observation import (
    PDMObservation,
)
from tuplan_garage.planning.simulation.planner.pdm_planner.proposal.batch_idm_policy import (
    BatchIDMPolicy,
)
from tuplan_garage.planning.simulation.planner.pdm_planner.proposal.pdm_generator import (
    PDMGenerator,
)
from tuplan_garage.planning.simulation.planner.pdm_planner.proposal.pdm_proposal import (
    PDMProposalManager,
)
from tuplan_garage.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import (
    PDMSimulator,
)
from tuplan_garage.planning.simulation.planner.pdm_planner.utils.pdm_emergency_brake import (
    PDMEmergencyBrake,
)
from tuplan_garage.planning.simulation.planner.pdm_planner.utils.pdm_geometry_utils import (
    parallel_discrete_path,
)
from tuplan_garage.planning.simulation.planner.pdm_planner.utils.pdm_path import PDMPath
from tuplan_garage.planning.simulation.planner.pdm_planner.observation.pdm_observation_utils import (
    get_drivable_area_map,
)
from tuplan_garage.planning.simulation.planner.pdm_planner.utils.pdm_array_representation import (
    states_se2_to_array,
    state_array_to_ego_states,
    ego_states_to_state_array
)
from tuplan_garage.planning.simulation.planner.pdm_planner.utils.pdm_enums import (
    MultiMetricIndex,
    WeightedMetricIndex,
    BBCoordsIndex
)
from tuplan_garage.planning.simulation.planner.pdm_planner.utils.lane_graph_parser import parse_lane_graph
from tuplan_garage.planning.simulation.planner.pdm_planner.utils.diffusion_utils import (
    interpolate_trajectory,
    convert_to_local,
    convert_to_global,
    fig_to_numpy
)
import time


def seed_everything(seed=123):
    torch.manual_seed(seed)
    np.random.seed(seed)


class DEPDMPlanner(AbstractPDMPlanner):
    """
    Interface for planners incorporating PDM-Closed. Used for PDM-Closed and PDM-Hybrid.
    """

    requires_scenario = True

    def __init__(
        self,
        trajectory_sampling: TrajectorySampling,
        proposal_sampling: TrajectorySampling,
        idm_policies: BatchIDMPolicy,
        lateral_offsets: Optional[List[float]],
        map_radius: float,
        checkpoint_path: str,
        dump_gifs_path: str,
        scenario,
        scorer_config,
        comfort_config,
        STDP_config,
        ablation_dual_timescale,
        ablation_planner,
        frequency_config,

        follow_centerline=False,
        use_idm_speed=False,
        use_pdm_proposals=False,
        use_pdm_closed_only=False,
        time_idxs_to_save=[],
    ):
        """
        Constructor for AbstractPDMClosedPlanner
        :param trajectory_sampling: Sampling parameters for final trajectory
        :param proposal_sampling: Sampling parameters for proposals
        :param idm_policies: BatchIDMPolicy class
        :param lateral_offsets: centerline offsets for proposals (optional)
        :param map_radius: radius around ego to consider
        """

        super(DEPDMPlanner, self).__init__(map_radius)

        assert (
            trajectory_sampling.interval_length == proposal_sampling.interval_length
        ), "AbstractPDMClosedPlanner: Proposals and Trajectory must have equal interval length!"

        # config parameters
        self._trajectory_sampling: int = trajectory_sampling
        self._proposal_sampling: int = proposal_sampling
        self._idm_policies: BatchIDMPolicy = idm_policies
        self._lateral_offsets: Optional[List[float]] = lateral_offsets
        
        self._checkpoint_path = checkpoint_path
        self._dump_gifs_path = dump_gifs_path

        # observation/forecasting class
        self._observation = PDMObservation(
            trajectory_sampling, trajectory_sampling if not os.environ.get('CLOSED', 0) else proposal_sampling, map_radius
        )

        # proposal/trajectory related classes
        self._generator = PDMGenerator(trajectory_sampling, trajectory_sampling if not os.environ.get('CLOSED', 0) else proposal_sampling)
        self._simulator = PDMSimulator(proposal_sampling)
        self._scorer = PDMScorer(proposal_sampling, scorer_config, comfort_config)
        self._emergency_brake = PDMEmergencyBrake(trajectory_sampling)

        # lazy loaded
        self._proposal_manager: Optional[PDMProposalManager] = None
        self._speed_limit = None

        # frequency parameters
        self._replan_freq = frequency_config['replan_freq'] 
        self._diffusion_freq_low = frequency_config['diffusion_freq_low'] 
        self._diffusion_freq = frequency_config['diffusion_freq'] 
        self._diffusion_freq_high = frequency_config['diffusion_freq_high'] 

        # initial neuron weights
        self._neuron_wr = 0.5
        self._neuron_wc = 0.5

        self._planner = "PDM"
        self._trajectory = None

        self._pdm_99_count = 0
        self._trapped_count = 0
        self._trapped = False

        # warm start
        self._use_warm_start = False
        self._prev_predictions = None

        # viz
        self._frames = []

        self._log_name = scenario.log_name
        self._token = scenario.token
        self._scenario_type = scenario.scenario_type

        self._yield_token = None
        self._lane_goal = None

        self._follow_centerline = follow_centerline
        self._use_idm_speed = use_idm_speed
        self._use_pdm_proposals = use_pdm_proposals
        self._use_pdm_closed_only = use_pdm_closed_only
        self._time_idxs_to_save = time_idxs_to_save

        self._STDP_config = STDP_config
        self._ablation_dual_timescale = ablation_dual_timescale
        self._ablation_planner = ablation_planner

    def name(self) -> str:
        """Inherited, see superclass."""
        return self.__class__.__name__

    def observation_type(self):
        """Inherited, see superclass."""
        return DetectionsTracks  # type: ignore
    
    def compute_planner_trajectory(
        self, current_input: PlannerInput
    ):
        """Inherited, see superclass."""

        gc.disable()
        self.ego_state, _ = current_input.history.current_state

        # Apply route correction on first iteration (ego_state required)
        if self._iteration == 0:
            self._route_roadblock_correction(self.ego_state)

        # Update/Create drivable area polygon map
        self._drivable_area_map = get_drivable_area_map(
            self._map_api, self.ego_state, self._map_radius,
            self._route_lane_dict
        )

        start_time = time.time()  
        trajectory = self._get_closed_loop_trajectory(current_input)
        end_time = time.time() 
        execution_time = end_time - start_time 
        self._execution_times.append([self._iteration, execution_time])  # Save each run time to the list
        self._planner_types.append([self._iteration, self._planner=="Diffusion"])  # Save each run time to the list
        self._iteration += 1
        return trajectory

    def initialize(self, initialization) -> None:
        """Inherited, see superclass."""
        
        self._initialization = initialization
        self._iteration = 0
        self._map_api = initialization.map_api
        self._load_route_dicts(initialization.route_roadblock_ids)

        gc.collect()

        # model loading
        config_path = '/'.join(self._checkpoint_path.split('/')[:-2])
        config_path = os.path.join(config_path, 'code/hydra/config.yaml')
        model_config = OmegaConf.load(config_path).model


        torch_module_wrapper = build_torch_module_wrapper(model_config)
        self._model = LightningModuleWrapper.load_from_checkpoint(
            self._checkpoint_path, model=torch_module_wrapper
        ).model

        self._model_loader = ModelLoader(self._model)
        self._model_loader.initialize()

        # TODO: magic
        self._model.predictions_per_sample = 256
        # Check the memory usage
        print(f"GPU memeory: {torch.cuda.memory_allocated() / 1024**2:.2f} MiB")

        self._trajectory_diffusion = None
        self._trajectory_pdm = None

        self._execution_times = []
        self._planner_types = []

        seed_everything(int(os.environ.get('SEED',0))) # random seed
    def _update_proposal_manager(self, ego_state: EgoState):
        """
        Updates or initializes PDMProposalManager class
        :param ego_state: state of ego-vehicle
        """

        self._current_lane = self._get_starting_lane(ego_state)
        create_new_proposals = self._iteration == 0

        if create_new_proposals:
            proposal_paths: List[PDMPath] = self._get_proposal_paths(self._current_lane)

            self._proposal_manager = PDMProposalManager(
                lateral_proposals=proposal_paths,
                longitudinal_policies=self._idm_policies,
            )

        # update proposals
        self._proposal_manager.update(self._current_lane.speed_limit_mps)

        # set speed limit
        if isinstance(self._current_lane, NuPlanLane):
            self._speed_limit = self._current_lane.speed_limit_mps
        else:
            # if its a lane connector, get the speed limit from adjacent lanes
            edges = self._current_lane.incoming_edges + self._current_lane.outgoing_edges + [self._current_lane]
            speed_limits = [edge.speed_limit_mps for edge in edges if edge.speed_limit_mps is not None]
            self._speed_limit = max(speed_limits) if len(speed_limits) > 0 else 100




    def _get_proposal_paths(
        self, current_lane: LaneGraphEdgeMapObject
    ) -> List[PDMPath]:
        """
        Returns a list of path's to follow for the proposals. Inits a centerline.
        :param current_lane: current or starting lane of path-planning
        :return: lists of paths (0-index is centerline)
        """
        centerline_discrete_path = self._get_discrete_centerline(current_lane)
        self._centerline = PDMPath(centerline_discrete_path)

        # 1. save centerline path (necessary for progress metric)
        output_paths: List[PDMPath] = [self._centerline]

        # 2. add additional paths with lateral offset of centerline
        if self._lateral_offsets is not None:
            for lateral_offset in self._lateral_offsets:
                offset_discrete_path = parallel_discrete_path(
                    discrete_path=centerline_discrete_path, offset=lateral_offset
                )
                output_paths.append(PDMPath(offset_discrete_path))

        return output_paths

    def _get_closed_loop_trajectory(
        self,
        current_input: PlannerInput,
    ) -> InterpolatedTrajectory:
        """
        Creates the closed-loop trajectory for PDM-Closed planner.
        :param current_input: planner input
        :return: trajectory
        """
        
        print("---------------------------------",self._iteration,"-------------------------------------")
        if self._iteration % self._replan_freq == 0:
            ego_state, observation = current_input.history.current_state
            ##################################### PDM #####################################
            if self._planner == "PDM" or self._iteration % self._diffusion_freq == 0:
                # 1. Environment forecast and observation update
                self._observation.update(
                    ego_state,
                    observation,
                    current_input.traffic_light_data,
                    self._route_lane_dict,
                )

                # 2. Centerline extraction and proposal update
                self._update_proposal_manager(ego_state)

                # 3. Generate/Unroll proposals
                # Generate PDM-Closed proposals
                pdm_proposals_global = self._generator.generate_proposals(
                    ego_state, self._observation, self._proposal_manager
                )
                
                
                # 4. Simulate proposals
                simulated_proposals_array = self._simulator.simulate_proposals(
                    pdm_proposals_global, ego_state
                )

                # lead_agent
                # Set IDM-determined speed limit
                if self._use_idm_speed:
                    self._speed_limit = self._generator.get_max_speed()
                self._lead_agent = self._generator.get_lead_agent()
                self._red_light = self._generator._red_light

                # 5. Score proposals
                state_history = ego_states_to_state_array(current_input.history.ego_states)

                proposal_scores = self._scorer.score_proposals(
                    simulated_proposals_array, # Scorer.states identified here
                    self.ego_state,
                    self._observation,
                    self._centerline,
                    self._route_lane_dict,
                    self._drivable_area_map,
                    self._map_api,
                    self._speed_limit,
                    self._lead_agent,
                    state_history,
                )

                # 6.a Apply brake if emergency is expected
                trajectory = self._emergency_brake.brake_if_emergency(
                    ego_state, proposal_scores, self._scorer
                )

                current_time_point = copy.deepcopy(self.ego_state.time_point)
                time_points = [current_time_point]
                for _ in range(1, self._proposal_sampling.num_poses + 1, 1):
                    current_time_point += TimePoint(int(0.1 * 1e6))
                    time_points.append(copy.deepcopy(current_time_point))
                
                best_idx = proposal_scores.argmax()
                
                states = state_array_to_ego_states(
                    self._scorer._states[best_idx],
                    time_points,
                    self.ego_state.car_footprint.vehicle_parameters,
                )
                self._trajectory_pdm = InterpolatedTrajectory(states)
                commands = self._simulator.get_commands()
                self._trajectory_pdm.store_commands(commands[best_idx])

            
            ##################################### Diffusion #####################################
            if self._iteration % self._diffusion_freq == 0:
                features = self._model_loader.build_features(current_input, self._initialization)
                features['constraints'] = self._generate_constraints(current_input)
                features['counter'] = len(self._frames)
                predictions = self._model_loader._model.run_diffusion(features)


                current_time_point = copy.deepcopy(self.ego_state.time_point)
                time_points = [current_time_point]
                for _ in range(1, self._proposal_sampling.num_poses + 1, 1):
                    current_time_point += TimePoint(int(0.1 * 1e6))
                    time_points.append(copy.deepcopy(current_time_point))

  

                diffusion_proposals_local = predictions["multimodal_trajectories"].detach().cpu().numpy()
                diffusion_proposals_local = diffusion_proposals_local.reshape(-1,16,3)


                # Np.concatenate: Add a fully zero time step before the trajectory to facilitate interpolation and subsequent calculations.
                diffusion_proposals_local = np.concatenate([np.zeros_like(diffusion_proposals_local[:,:1]), diffusion_proposals_local], axis=1)
                diffusion_proposals_local = interpolate_trajectory(diffusion_proposals_local, 81)
                diffusion_proposals_global = convert_to_global(self.ego_state, diffusion_proposals_local)
                diffusion_proposals_global = np.concatenate([
                    diffusion_proposals_global, np.zeros((diffusion_proposals_global.shape[0], diffusion_proposals_global.shape[1], 8))
                ], axis=2)

                # Simulate
                all_proposals = np.concatenate([pdm_proposals_global,diffusion_proposals_global],axis=0)

                all_states_array = self._simulator.simulate_proposals(all_proposals, self.ego_state) # Simulation of proposals by emulator
                
                all_states_scores = self._scorer.score_proposals(
                    all_states_array, 
                    self.ego_state,
                    self._observation,
                    self._centerline,
                    self._route_lane_dict,
                    self._drivable_area_map,
                    self._map_api,
                    self._speed_limit,
                    self._lead_agent,
                    state_history,
                )
                all_states_scores = all_states_scores * 100

                pdm_scores = all_states_scores[0:pdm_proposals_global.shape[0]]
                diffusion_scores = all_states_scores[pdm_proposals_global.shape[0]:]

                pdm_idx = pdm_scores.argmax()
                pdm_state_array = all_states_array[pdm_idx]
                diffusion_idx = diffusion_scores.argmax()
                diffusion_state_array = all_states_array[diffusion_idx]

                best_pdm_score = pdm_scores[pdm_idx]
                print("best_pdm_score",best_pdm_score)
                
                best_diffusion_score = diffusion_scores[diffusion_idx]
                print("best_diffusion_score",best_diffusion_score)

                # diffusion proposal number regulator
                if best_diffusion_score > 98:
                    self._model.predictions_per_sample = int(self._model.predictions_per_sample / 2)
                if best_diffusion_score < 98:
                    self._model.predictions_per_sample = int(self._model.predictions_per_sample * 2)
                self._model.predictions_per_sample = max(min(self._model.predictions_per_sample, 256), 32)

                best_idx = all_states_scores.argmax()
                best_state_array = all_states_array[best_idx]

                states = state_array_to_ego_states(
                all_states_array[pdm_proposals_global.shape[0]+diffusion_idx],
                time_points,
                self.ego_state.car_footprint.vehicle_parameters,
                )
                trajectory = InterpolatedTrajectory(states)
                commands = self._simulator.get_commands()
                trajectory.store_commands(commands[best_idx])
                self._trajectory_diffusion = trajectory

                if best_diffusion_score < 80 and best_pdm_score < 80:
                    self._trapped_count = self._trapped_count+1
                else:
                    self._trapped_count = 0
                    self._trapped = False

                if self._trapped_count >= 5:
                    self._trapped = True
                    print("traaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaped")

                ####################################### fuse #####################################
                S_a = best_diffusion_score  
                S_c = best_pdm_score  

 
                alpha = 5  # You can adjust this value to control the weighting effect

                # Calculate the weight after exponential weighting
                w_a = 1 / (1 + np.exp(alpha * (S_c-S_a)))  
                w_c = 1 / (1 + np.exp(alpha * (S_a-S_c)))  

                # proposal fusion
                fused_proposal = pdm_proposals_global[pdm_idx] * w_c + diffusion_proposals_global[diffusion_idx] * w_a
                fused_proposal = np.expand_dims(fused_proposal,axis=0)
                fused_state_array = self._simulator.simulate_proposals(fused_proposal, self.ego_state) 
                fused_state_array = np.squeeze(fused_state_array)
                states_fused = state_array_to_ego_states(
                fused_state_array,
                time_points,
                self.ego_state.car_footprint.vehicle_parameters,
                )
                trajectory = InterpolatedTrajectory(states_fused)
                commands = self._simulator.get_commands()
                trajectory.store_commands(commands[0])
                self._trajectory_diffusion = trajectory
                ##################################### STDP compute #############################
                s1 = self._STDP_config['good_score']  
                # STDP parameters
                tau_pos = self._STDP_config['tau_pos']  
                tau_neg = self._STDP_config['tau_neg']  
                A_plus = self._STDP_config['A_plus']  # LTP gain
                A_minus = self._STDP_config['A_minus']  # Ltd attenuation

                delta_t1 = 1/(best_pdm_score - s1)  
                delta_t2 = 1/(best_diffusion_score - s1)  
                
                # Calculate the impact of stdp rules on two systems
                if delta_t1 > 0:
                    # System 1:ltp
                    self._neuron_wc = self._neuron_wc + A_plus * np.exp(-abs(delta_t1)/tau_pos)
                else:
                    # System 1：LTD
                    self._neuron_wc = self._neuron_wc - A_minus * np.exp(-abs(delta_t1)/tau_neg)

                if delta_t2 > 0:
                    # System 2：LTP
                    self._neuron_wr = self._neuron_wr + A_plus * np.exp(-abs(delta_t2)/tau_pos)
                else:
                    # System 2：LTD
                    self._neuron_wr = self._neuron_wr - A_minus * np.exp(-abs(delta_t2)/tau_neg)

                # Limit weights between 0 and 1
                self._neuron_wc = max(0, min(1, self._neuron_wc))
                self._neuron_wr = max(0, min(1, self._neuron_wr))
                ################################### Select Planner #############################
                if self._ablation_dual_timescale == 0:
                    ####################### dual time-scale decision neuron #########################
                    # turn to diffusion
                    if (((self._neuron_wr > self._neuron_wc or (self._neuron_wr < 0.01 and self._neuron_wc < 0.01)) and best_pdm_score > 73 and best_diffusion_score > 73 ) \
                        or best_pdm_score < 73 ) and (self._generator._num_agent_ahead <= 3) or self._trapped == True:
                        self._planner = "Diffusion"
                        self._diffusion_freq = self._diffusion_freq_high
                        if best_diffusion_score > 95:
                            self._diffusion_freq = self._diffusion_freq_low
                        self._pdm_99_count = 0

                    # turn to PDM
                    if ((self._neuron_wc > self._neuron_wr or (self._neuron_wr > 0.99 and self._neuron_wc > 0.99)) and best_pdm_score > 73 and best_diffusion_score > 73 \
                        or (best_diffusion_score < 73 and best_pdm_score > 73)) or self._pdm_99_count >= 6 or self._red_light == True:
                        self._planner = "PDM"
                        self._diffusion_freq = self._diffusion_freq_low
                        if best_pdm_score < 73:
                            self._planner = "Diffusion"
                            self._diffusion_freq = self._diffusion_freq_high
                            self._pdm_99_count = 0
                elif self._ablation_dual_timescale == 1:
                    ####################### score rule #########################
                    if best_pdm_score < 73:
                        self._planner = "Diffusion"
                        self._diffusion_freq = self._diffusion_freq_high

                    # rule1 rule2
                    if best_diffusion_score < 73 and best_pdm_score > 73:
                        self._planner = "PDM"
                        self._diffusion_freq = self._diffusion_freq_low
                elif self._ablation_dual_timescale == 2:
                    ####################### decision neuron #########################
                    if self._neuron_wr > self._neuron_wc:
                        self._planner = "Diffusion"
                        self._diffusion_freq = self._diffusion_freq_high
                    # rule1 rule2
                    if self._neuron_wc > self._neuron_wr:
                        self._planner = "PDM"
                        self._diffusion_freq = self._diffusion_freq_low
                elif self._ablation_dual_timescale == 3:
                    ####################### scenario rule ########################
                    if  self._trapped == True:
                        self._planner = "Diffusion"
                        self._diffusion_freq = self._diffusion_freq_high
                        self._pdm_99_count = 0
                    # rule1 rule2
                    if self._pdm_99_count >= 6 or self._red_light == True:
                        self._planner = "PDM"
                        self._diffusion_freq = self._diffusion_freq_low
                else:
                    raise RuntimeError("unknown ablation_dual_timescale")
                
                ####################### ablation planner ########################
                if self._ablation_planner == 1:
                    self._planner = "Diffusion"
                    self._diffusion_freq = self._diffusion_freq_high
                    self._pdm_99_count = 0
                elif self._ablation_planner == 2:
                    self._planner = "PDM"
                    self._diffusion_freq = self._diffusion_freq_low

                if best_pdm_score > 93:
                    self._pdm_99_count = self._pdm_99_count + 1
                else:
                    self._pdm_99_count = 0
            #########################################select##############################

            if self._planner == "PDM" :
                trajectory = self._trajectory_pdm
                print("rule based planner")
            else:
                trajectory = self._trajectory_diffusion
                print("learning based planner")
            self._trajectory = trajectory        
        return self._trajectory
    

    def _generate_constraints(self, current_input):
        """
        Each constraint is a function that maps an ego-trajectory to some scalar cost to be minimized.
        """
        return [
            self._make_pdm_scorer_constraint(current_input, weight=100.0)
        ]
    
    def _make_pdm_scorer_constraint(self, current_input, weight=1.0):
        def constraint_fn(trajectory): 
            device = trajectory.device

            # Convert to scorer format (numpy, 10 Hz, global frame)

            trajectory = trajectory.detach().cpu().numpy()
            trajectory = trajectory.reshape(-1,16,3)

            trajectory = np.concatenate([np.zeros_like(trajectory[:,:1]), trajectory], axis=1)
            trajectory_interp = interpolate_trajectory(trajectory, 81)
            trajectory_global = convert_to_global(self.ego_state, trajectory_interp)
            trajectory_padded = np.concatenate([
                trajectory_global, np.zeros((trajectory_global.shape[0], trajectory_global.shape[1], 8))
            ], axis=2)

            # Simulate
            trajectory_sim = self._simulator.simulate_proposals(trajectory_padded, self.ego_state) # 
            trajectory_sim_local = convert_to_local(self.ego_state, trajectory_sim[...,:3])  # for viz

            # Scoring
            state_history = ego_states_to_state_array(current_input.history.ego_states)
            scores = self._scorer.score_proposals(
                trajectory_sim, 
                self.ego_state,
                self._observation,
                self._centerline,
                self._route_lane_dict,
                self._drivable_area_map,
                self._map_api,
                self._speed_limit,
                self._lead_agent,
                state_history,
            )

            scores = -torch.as_tensor(scores, device=device)
            scores = scores * weight
            return scores, {'traj_sim': trajectory_sim_local}

        return constraint_fn

    def _save_gif(self, fname='', frames=None, duration=200):
        if frames is None:
            frames = self._frames
        if fname == '':
            fname_dir = f'{self._dump_gifs_path}'
            if not os.path.exists(fname_dir):
                os.makedirs(fname_dir)
            fname = f'{fname_dir}/{self._scenario_type}__{self._log_name}__{self._token}__{time.strftime("%m%d-%H%M")}.gif'
        frames = [Image.fromarray(frame) for frame in frames]
        frames[0].save(fname, save_all=True, append_images=frames[1:], duration=duration, loop=0)
        print(f'SAVING GIF TO {fname}')

    def _visualize(self, ego_state, features, predictions):
        # Viz
        if 'traj_sim' in predictions:
            all_trajectories_sim = predictions['traj_sim']
        all_trajectories = predictions['multimodal_trajectories'].detach().cpu().numpy()
        all_trajectories = all_trajectories.reshape(-1,self.H,3)
        scores = predictions['scores'].detach().cpu().numpy()

        centerline = self._centerline.get_nearby_path(self.ego_state, speed_limit=self._speed_limit)
        centerline = np.array(centerline.coords)
        centerline = convert_to_local(ego_state, centerline)

        frame = visualize(
            features['vector_set_map'].to_device('cpu'),
            features['agent_history'].to_device('cpu'),

            centerline=centerline,
            centerline_color='orange',

            alltrajectoriessim=all_trajectories_sim,
            alltrajectoriessim_c=scores,
            alltrajectoriessim_cmap='winter_r',
            alltrajectoriessim_cmin=-110,
            alltrajectoriessim_cmax=0,

            alltrajectories=all_trajectories,
            alltrajectories_c=scores,
            alltrajectories_cmap='autumn',
            alltrajectories_cmin=-110,
            alltrajectories_cmax=0,

            pixel_size=0.1,
            radius=60,
        )
        return frame