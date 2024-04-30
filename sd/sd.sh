# Cooperate UQ into ddim sampler, experiment results will be saved in ./ddim_exp/skipUQ/
CUDA_VISIBLE_DEVICES=0 python ddim_skipUQ.py --prompt "A photo of little girl with a red umbrella" \
--ckpt C:\Users\cilev\Documents\DL2\external_utils\SD\v1-5-pruned-emaonly.ckpt --local_image_path C:\Users\cilev\Documents\DL2\external_utils\SD\Images --laion_art_path C:\Users\cilev\Documents\DL2\external_utils\SD\laion-art.parquet \
--H 512 --W 512 --scale 3 --train_la_data_size 1000 --train_la_batch_size 10 \
--sample_batch_size 2 --total_n_samples 48 --timesteps 50

# Cooperate UQ into dpm-solver-2 sampler, experiment results will be saved in ./dpm_solver_2_exp/skipUQ/
CUDA_VISIBLE_DEVICES=1 python dpmsolver_skipUQ.py --prompt "A photo of a little girl with a red umbrella" \
--ckpt C:\Users\cilev\Documents\DL2\external_utils\SD\v1-5-pruned-emaonly.ckpt --local_image_path C:\Users\cilev\Documents\DL2\external_utils\SD\Images --laion_art_path C:\Users\cilev\Documents\DL2\external_utils\laion-art.parquet \
--H 512 --W 512 --scale 3 --train_la_data_size 1000 --train_la_batch_size 10 \
--sample_batch_size 2 --total_n_samples 48 --timesteps 50
