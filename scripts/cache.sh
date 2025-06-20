python $NUPLAN_DEVKIT_ROOT/nuplan/planning/script/run_training.py \
    py_func=cache \
    +training=training_diffusion_proposal_model \
    cache.force_feature_computation=True \
    scenario_builder=nuplan_train \
    worker=single_machine_thread_pool \
    worker.use_process_pool=True \
    worker.max_workers=null \
    hydra.searchpath="[pkg://tuplan_garage.planning.script.config.common, pkg://tuplan_garage.planning.script.config.simulation, pkg://nuplan.planning.script.config.common, pkg://nuplan.planning.script.experiments]" \

