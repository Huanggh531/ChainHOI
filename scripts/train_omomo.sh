#train on omomo on gpu[4]
python src/train.py trainer.devices=\'4,\'  data=omomo data.joint_format=behave_joint_5 logger=tensorboard \
    callbacks.model_checkpoint.save_top_k=5 callbacks.model_checkpoint.save_last=False data.batch_size=8 trainer.max_epochs=400 \
    callbacks.model_checkpoint.monitor=Metrics/contact_distance_mean callbacks.model_checkpoint.mode=min model=chainhoi_omomo \
    test=true seed=2024 trainer.deterministic=False model.gcn_denoiser.joint_dim=12 model.gcn_denoiser.num_layers=6 \
    model.gcn_denoiser.base_dim=64 trainer.strategy=ddp_find_unused_parameters_true trainer.check_val_every_n_epoch=1 \
    data.val_batch_size=32 model.noise_scheduler.prediction_type=sample model.gcn_denoiser.pos_emb=cos \
    model.gcn_denoiser.layout=behave_graph_6 model.gcn_denoiser.input_proj_type=5 model.guidance_scale=2 \
    model.gcn_denoiser.block_arch=0 model.gcn_denoiser.reduce_joint=conv4 model.gcn_denoiser.temp_arch=OCA \
    model.gcn_denoiser.arch=7 model.gcn_denoiser.trans_layers=4 model.obj_flag=True model.gcn_denoiser.has_obj=true \
    model.gcn_denoiser.obj_clound_dim=512