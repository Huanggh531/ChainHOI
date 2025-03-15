CUDA_VISIBLE_DEVICES=2 python src/test_time.py trainer.devices=\'0,\'  data=behave_gcn data.joint_format=behave_joint_5 data.batch_size=32 \
    model=chainhoi_behave     seed=2024 model.gcn_denoiser.joint_dim=12 model.gcn_denoiser.num_layers=6     \
    model.gcn_denoiser.base_dim=64 data.val_batch_size=32 model.noise_scheduler.prediction_type=sample     \
    model.gcn_denoiser.pos_emb=cos model.gcn_denoiser.layout=behave_graph_6     model.gcn_denoiser.input_proj_type=5 \
    model.guidance_scale=2 model.gcn_denoiser.block_arch=0 model.gcn_denoiser.reduce_joint=conv4     \
    model.gcn_denoiser.temp_arch=OCA model.gcn_denoiser.arch=7 model.gcn_denoiser.trans_layers=4     \
    model.obj_flag=True model.gcn_denoiser.has_obj=true model.gcn_denoiser.obj_clound_dim=512 trainer.precision=32     \
    ckpt_path="logs/behave/e063.ckpt"