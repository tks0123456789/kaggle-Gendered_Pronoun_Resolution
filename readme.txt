Put the test file into this folder.
Build docker image using docker/Dockerfile then start the docker.

The following commands will create two submission files subm001.csv, subm002.csv.

./train.sh offsets 30
./train.sh nli 30
python offsets_predict.py --test_file "test_file_name"
python nli_predict.py --test_file "test_file_name"
python make_submission.py



The model directory only contains checksum of model files because the total file size of model is over 100G.

Directory tree
.
├── docker
│   └── Dockerfile
├── log              Training stats: loss, time..
│   ├── nli_000.csv
│   ├── nli_001.csv
│   ├── nli_002.csv
│   ├── nli_003.csv
│   ├── nli_004.csv
│   ├── nli_005.csv
│   ├── nli_006.csv
│   ├── nli_007.csv
│   ├── nli_008.csv
│   ├── nli_009.csv
│   ├── nli_010.csv
│   ├── nli_011.csv
│   ├── nli_012.csv
│   ├── nli_013.csv
│   ├── nli_014.csv
│   ├── nli_015.csv
│   ├── nli_016.csv
│   ├── nli_017.csv
│   ├── nli_018.csv
│   ├── nli_019.csv
│   ├── nli_020.csv
│   ├── nli_021.csv
│   ├── nli_022.csv
│   ├── nli_023.csv
│   ├── nli_024.csv
│   ├── nli_025.csv
│   ├── nli_026.csv
│   ├── nli_027.csv
│   ├── nli_028.csv
│   ├── nli_029.csv
│   ├── offsets_000.csv
│   ├── offsets_001.csv
│   ├── offsets_002.csv
│   ├── offsets_003.csv
│   ├── offsets_004.csv
│   ├── offsets_005.csv
│   ├── offsets_006.csv
│   ├── offsets_007.csv
│   ├── offsets_008.csv
│   ├── offsets_009.csv
│   ├── offsets_010.csv
│   ├── offsets_011.csv
│   ├── offsets_012.csv
│   ├── offsets_013.csv
│   ├── offsets_014.csv
│   ├── offsets_015.csv
│   ├── offsets_016.csv
│   ├── offsets_017.csv
│   ├── offsets_018.csv
│   ├── offsets_019.csv
│   ├── offsets_020.csv
│   ├── offsets_021.csv
│   ├── offsets_022.csv
│   ├── offsets_023.csv
│   ├── offsets_024.csv
│   ├── offsets_025.csv
│   ├── offsets_026.csv
│   ├── offsets_027.csv
│   ├── offsets_028.csv
│   └── offsets_029.csv
├── make_submission.py
├── model
│   └── MD5SUMS          Checksum of 90 model files.
├── nli_cfg.json
├── nli_predict.py
├── nli_train.py
├── nli_v1.csv
├── offsets_cfg.json
├── offsets_predict.py
├── offsets_train.py
├── offsets_v1.csv
├── readme.txt            This file.
├── subm001.csv           Same as offsets_v1.csv.
├── subm002.csv           0.7 * offsets_v1.csv + 0.3 * nli_v1.csv
├── test_stage_1.tsv
├── train.sh
└── utility.py
