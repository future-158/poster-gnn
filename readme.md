# update
2022-05-22: plot 그리는 코드, 환경 추가
문서 아래 부분  # plot 부분 참조

2022-05-20: ostia 연안 픽셀 추가. 넣을 때 성능이 더 안나옴. ostia=false 옵션으로 없앨 수 있음.
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
- (추가)ostia 연안 기준 pixel
2018~2021년 12시간 간격 자료

# 환경 설치
steps:
    - make install

# 전처리 
설명: data/clean/sst.csv를 읽어서 knn imputation, lof outlier removal, kalman smoothing을 수행함
steps:
    - make prepare

# 학습

입력 자료 data/model_in/sst.csv 혹은 ostia=true이면 data/model_in/merge.csv
을 읽어서 하이퍼파라미터 튜닝함


steps:
    - make mlflow # run mlflow server
    - make hpo # hpo
    - make visualize # 실행 후 http://serverip:20006으로 들어가면 정점별 obs,pred 볼 수 있음



# TODO
- compare with atg
- compare with univariate temporal model (ex. lstm, 1dcnn)
- workflow 추가. pytest.
- automatically catch dependency change and increase api_version.

python task/stemgnn/main.py --multirun 'args.dropout_rate=interval(0.1,0.5)' 'args.lr=interval(0.00001,0.01)' 'args.multi_layer=int(interval(1,5))' 'args.window_size=int(interval(1,31))'


# plot
steps
- conda env create --file plot-environment.yml
stemgnn-plot 이름으로 새 환경 설치
geopandas, plotly  등 라이브러리가 기존 stemgnn-poster와 충돌이 나서 새로 만들었음

- conda activate stemgnn-plot
환경 activate

- python step4_plot.py
시계열, box plot을 data/report에 생성함.








