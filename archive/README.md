# 원본 보존 자료

`notebooks/`는 2026-09-07 정비 직전 노트북 3개를 바이트 단위로 복사한 원본입니다.
`README-original.md`는 정비 이전 소개문입니다.

원본 02/03의 일부 모델 평가 셀에는 이전 모델의 예측값을 재사용하는 오류가 있으므로, 출력된 동일 성능을 모델별 독립 평가 결과로 인용하면 안 됩니다.
원본 소개문의 StandardScaler, BatchNormalization, 변수 중요도 예시 등도 코드와 일치하지 않는 부분이 있어 역사 기록으로만 보존합니다.

수정 실행 코드는 `../reanalysis/`, 측정 결과는 `../reanalysis/results/`를 확인하세요.
루트의 원본 `base_model.keras`, `base_model (1).keras`, `scaler.pkl`과 CSV는 변경하지 않았습니다.
