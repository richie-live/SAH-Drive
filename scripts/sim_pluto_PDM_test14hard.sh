SPLIT=test14-hard
CHALLENGE=closed_loop_reactive_agents # open_loop_boxes, closed_loop_nonreactive_agents, closed_loop_reactive_agents
EXPERIMENT=sim_pluto_PDM_test14hard_R

python $NUPLAN_DEVKIT_ROOT/nuplan/planning/script/run_simulation.py \
+simulation=$CHALLENGE \
planner=pluto_pdm_planner \
scenario_builder=nuplan_test \
scenario_filter=$SPLIT \
experiment_name=$EXPERIMENT \
hydra.searchpath="[pkg://tuplan_garage.planning.script.config.common, pkg://tuplan_garage.planning.script.config.simulation, pkg://nuplan.planning.script.config.common, pkg://nuplan.planning.script.experiments]" \
ego_controller=one_stage_controller \
worker=single_machine_thread_pool \
worker.use_process_pool=True \