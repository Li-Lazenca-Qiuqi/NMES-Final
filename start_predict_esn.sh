python -im predict_esn\
       --model models/nsub16-gamma01-nr6000-GPUtest/model.zarr\
       --metadata models/nsub16-gamma01-nr6000-GPUtest/run.json\
       --input /mnt/sda/lixf/SQG/byAI/25/sqg64_300s.zarr\
       --start-index 2076480\
       --output forecasts/nsub01-gamma01-nr6000-7210day.zarr\
       #>./logfile/predict_esn/nsub01-gamma01-nr3000.log

