
EXPERIMENT=interplan_planTF_PDM

python $INTERPLAN_PLUGIN_ROOT/interplan/planning/script/run_simulation.py \
+simulation=default_interplan_benchmark \
planner=plantf_pdm_planner \
scenario_filter=interplan10 \
experiment_name=$EXPERIMENT \
ego_controller=one_stage_controller \
worker=single_machine_thread_pool \
worker.use_process_pool=True \
hydra.searchpath="[\
pkg://interplan.planning.script.config.common,\
pkg://interplan.planning.script.config.simulation,\
pkg://interplan.planning.script.experiments,\
pkg://tuplan_garage.planning.script.config.common,\
pkg://tuplan_garage.planning.script.config.simulation,\
pkg://nuplan.planning.script.config.common,\
pkg://nuplan.planning.script.config.simulation,\
pkg://nuplan.planning.script.experiments\
]"
