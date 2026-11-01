from pathlib import Path # 파일, 폴더 경로 다룬다.
from ultralytics import YOLO # Yolo 불러옴

EPOCHS = 100
IMG_SIZE = 640 # 학습할때 이미지 크기
PROJECT = Path(__file__).resolve().parent / "runs" # __file__: 코드 위치, resolve(): 경로를 절대경로로 변환, .parent: 그것의 상위경로
NAME = "coco_yolo26n_Fine-tuning" # 학습 결과 파일 이름
RESUME = True # False: 학습 새로 시작 / True: 기존 last.pt에서 중단된 학습 재개

# 첫 학습, 도중 학습 구분
if RESUME: # RESUME이 True면 도중 학습
    checkpoint = Path(PROJECT) / NAME / "weights" / "last.pt" # 저장하는 파일 경로. NAME, weights는 폴더 이름
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint를 찾을 수 없습니다: {checkpoint}") # 파일이 없으면 예외 발생
    model = YOLO(str(checkpoint)) # 앞에서 checkpoint가 Path 객체라서 YOLO() 한테 현재 모델의 경로를 str로 알려줘야 한다.
    result = model.train(resume=True)
else: # 첫 학습
    model = YOLO("yolo26n.pt")
    result = model.train(
        data="coco.yaml", # 공식 데이터 셋 설정 파일
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        optimizer="auto",
        batch=-1, # GPU 사용량에 따른 자동 설정
        device=0, # 단일 GPU
        val=True, # validation 쓰겠다.
        plots=True, # 학습 결과 그래프 생성
        save=True, # checkpoint 저장
        project=PROJECT,
        name=NAME,
        exist_ok=False, # 동일 폴더 존재하면 새 이름으로 저장
        workers=1,  # 데이터 로딩 프로세스 2개 사용
        cache=False  # 이미지 RAM 캐싱 비활성화
    )

print("학습 종료")
print(f"결과 저장 위치: {model.trainer.save_dir}") # 모델 결과가 저장된 위치 알려줌. 이건 실제 위치

