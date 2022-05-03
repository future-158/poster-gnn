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

# 전처리 
- make install-prepare
prepare anaconda 환경을 envs/prepare에 설치

- make activate-prepare
venv -> envs/prepare로 심볼릭 링크 생성
code에서 작업할 때 python interpreter로 ./venv/bin/python을 설정하면 됨

- conda activate venv/

- python prepare/load_impute.py
data/clean/sst.csv를 읽어서 knn imputation, lof outlier removal, kalman smoothing을 수행함

# 학습
- make install-pyg
prepare anaconda 환경을 envs/pyg에 설치

- make activate-pyg
venv -> envs/pyg로 심볼릭 링크 생성
code에서 작업할 때 python interpreter로 ./venv/bin/python을 설정하면 됨

- conda activate && conda activate venv/

- mlflow --host 0.0.0.0 --port 23002 &
mlflow 서버 background에서 실행. 서버이름:23002로 접속하면 확인 가능

- CUDA_VISIBLE_DEVICES='GPUID' python src/pyg/main.py
입력 자료 data/model_in/sst.csv
config 파일 conf/config.yml, conf/model/pyg.yml
을 읽어서 하이퍼파라미터 튜닝함
conf/model/pyg.yml에서 search_space를 수정하면 됨

# TODO
- microsoft/stemgnn 모델과 비교
- workflow 추가 **현재는 파일만 만들어놓음**









