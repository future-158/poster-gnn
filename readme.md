

# update
2022-05-17: stemgnn test rmse as low as 0.6~0.7

2022-05-20: new error.
ValueError: Values were told. Values cannot be specified when state is TrialState.PRUNED or TrialState.FAIL.


# 개요
한반도 정점 수온 자료를 가지고 GNN 모델로 12, 24시간 후 수온 예측 

# 사용 모델 
pytorch geometric temporal 제공 모델

# 입력자료 (API 다운로드)
- 조위관측소
- 해양관측부이
- 기상청 파고부이
- 기상청 해양부이
- 관측시간
2018~2021년 12시간 간격 자료

# 환경 설치
steps:
    - make install

# 전처리 
설명: data/clean/sst.csv를 읽어서 knn imputation, lof outlier removal, kalman smoothing을 수행함
steps:
    - make install
    - conda activate poster-prepare
    - python prepare/load_impute.py

# 학습
설명: 

입력 자료 data/model_in/sst.csv
config 파일 conf/config.yml, conf/model/pyg.yml
을 읽어서 하이퍼파라미터 튜닝함
conf/model/pyg.yml에서 search_space를 수정하면 됨

steps:
    - conda activate poster-prepare
    - mlflow server --host 0.0.0.0 --port 23005 &
    - CUDA_VISIBLE_DEVICES='GPUID' python src/pyg/main.py


mlflow 서버 background에서 실행. 서버이름:23002로 접속하면 확인 가능

# TODO
- microsoft/stemgnn vs atg
- workflow 추가. pytest.
- set optuna-storege-name with dependency hash. reuse automatically when dependency(code, config file, input data) not changed

python task/stemgnn/main.py --multirun 'args.dropout_rate=interval(0.1,0.5)' 'args.lr=interval(0.00001,0.01)' 'args.multi_layer=int(interval(1,5))' 'args.window_size=int(interval(1,31))'









