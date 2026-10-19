nohup python train_esn.py \
	--data /mnt/sda/lixf/SQG/byAI/25/sqg64_300s.zarr\
	--nsub 16\
	--gamma 0.1\
	--reservoir 3000\
	--workers 16\
	--localport 8788\
	--output models/nsub16-gamma01-nr3000\
	>./logfile/16sub_01_3000_16workers.log

