"""Generate evidence charts and executed reading notebooks from measured outputs."""
from pathlib import Path
import contextlib
import io
import json
import shutil
import os
import nbformat as nb
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reanalysis' / 'results'
ASSETS = ROOT / 'reanalysis' / 'assets'


def create_notebook(name, sections):
    path = ROOT / name
    archive = ROOT / 'archive' / 'notebooks' / name
    archive.parent.mkdir(parents=True, exist_ok=True)
    if not archive.exists():
        shutil.copy2(path, archive)
    doc = nb.v4.new_notebook()
    doc.metadata.kernelspec = {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'}
    doc.cells = [nb.v4.new_markdown_cell(s) if kind == 'md' else nb.v4.new_code_cell(s) for kind, s in sections]
    env = {}
    count = 0
    for cell in doc.cells:
        if cell.cell_type == 'code':
            count += 1
            stream = io.StringIO()
            with contextlib.redirect_stdout(stream):
                exec(compile(cell.source, name, 'exec'), env)
            cell.execution_count = count
            cell.outputs = [nb.v4.new_output('stream', name='stdout', text=stream.getvalue())] if stream.getvalue() else []
    nb.validate(doc)
    nb.write(doc, path)


def main():
    os.chdir(ROOT)
    ASSETS.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False})
    r = json.loads((OUT / 'results.json').read_text(encoding='utf-8'))
    legacy = json.loads((OUT / 'legacy_shared_results.json').read_text(encoding='utf-8'))
    fig, ax = plt.subplots(figsize=(11, 6), layout='constrained')
    df = pd.read_csv(OUT / 'survey_metrics.csv').query("partition == 'test'").set_index('model')
    df.index = ['Random\nForest', 'Logistic\nDense', 'DNN\nhidden', 'DNN\ndropout', 'DNN\nearly stop', 'Rebuilt shared\narchitecture*']
    df[['accuracy', 'macro_f1', 'recall_0']].plot.bar(ax=ax, color=['#2859b6', '#2d9a8a', '#dc9750'], rot=0)
    ax.set_ylim(0, 1.12)
    ax.set_title('Survey holdout | September 2026 revalidation | n=3,856')
    ax.set_ylabel('Score (0-1)')
    ax.set_xlabel('Selected on validation: RandomForest | *Auxiliary reconstruction, NOT the original team checkpoint')
    ax.legend(['Accuracy', 'Macro F1', 'Recall: neutral or dissatisfied'], fontsize=9, loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=3)
    fig.savefig(ASSETS / 'model-comparison.png', dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 5), layout='constrained')
    imp = pd.read_csv(OUT / 'feature_importance.csv').head(7).iloc[::-1]
    ax.barh(imp.feature, imp.macro_f1_decrease, xerr=imp['std'], color='#2859b6', capsize=3)
    ax.set_title('Random Forest | validation permutation importance')
    ax.set_xlabel('Decrease in macro F1 | 5 repeats | not causal effects')
    fig.savefig(ASSETS / 'feature-importance.png', dpi=170)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 5), layout='constrained')
    cm = r['survey_models'][r['selected_survey_model']]['test']['confusion_matrix']
    ax.imshow(cm, cmap='Blues')
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i][j]), ha='center', va='center', fontsize=22, color='white' if cm[i][j] > 1700 else '#12213b')
    ax.set_xticks([0, 1], ['Neutral / dissatisfied', 'Satisfied'])
    ax.set_yticks([0, 1], ['Neutral / dissatisfied', 'Satisfied'])
    ax.set_xlabel('Predicted label')
    ax.set_ylabel('Actual label')
    ax.set_title('Random Forest | survey test n=3,856')
    fig.savefig(ASSETS / 'confusion-matrix.png', dpi=160)
    plt.close(fig)
    intro = ('md', '> **2026년 9월 코드 정비·재검증본**\n> 4월 당시 원본과 출력은 `archive/notebooks/`에 보존했습니다. 아래 수치는 당시 성과를 소급 수정한 것이 아니라 새 평가 절차로 측정한 결과입니다.\n> 저장된 결과를 확인하는 노트북입니다. 전체 재학습: `python -m reanalysis.pipeline` 및 `python -m reanalysis.legacy_shared`.')
    setup = ('code', "from pathlib import Path\nimport pandas as pd\nimport json\nROOT = Path.cwd()\nRESULTS = ROOT / 'reanalysis' / 'results'\nassert RESULTS.exists(), '저장소 루트에서 실행하세요.'")
    create_notebook('01 - 탐색적 데이터 분석.ipynb', [intro, ('md', '# 01. 데이터 이해와 해석\n개인 EDA 단계에서 확인했던 분포와 변수 중요도를 근거 중심으로 정리합니다. 0은 **중립 또는 불만족**, 1은 만족입니다.'), setup,
        ('code', "data = pd.read_csv('survey.csv')\nprint('shape:', data.shape)\nprint(data['Satisfaction'].value_counts())\nprint(data.isna().sum()[lambda x: x > 0])"),
        ('md', '## 개인 분석에서 팀 논의로\n원본 EDA에는 Random Forest와 연령대별 분석이 있습니다. 아래는 새 전처리와 validation permutation importance로 다시 계산한 결과이며, 원본 impurity importance와 수치가 다릅니다. 중요도는 인과효과가 아닙니다.'),
        ('code', "print(pd.read_csv(RESULTS / 'feature_importance.csv').head(7).to_string(index=False))"),
        ('md', '![검증 데이터 변수 중요도](reanalysis/assets/feature-importance.png)\n\n**서비스 개선 제안:** 여행 목적별 고객 경험을 구분하고, Wi-Fi·온라인 탑승 절차의 불편 지점을 추가 조사합니다. 서비스 도입 효과는 측정하지 않았습니다.')])
    create_notebook('02 - 기본 모델링.ipynb', [intro, ('md', '# 02. 개인 기본 모델링의 평가 오류 수정\n원본 모델별 예측값 갱신 누락을 수정했습니다. 학습/검증/평가를 먼저 60/20/20 계층 분할하고 학습 데이터만으로 전처리를 학습합니다. 모델 선택은 validation macro F1, 동률이면 0 클래스 recall입니다. 이는 **재검증 때 도입한 기준**입니다.'), setup,
        ('code', "r = json.loads((RESULTS / 'results.json').read_text(encoding='utf-8'))\nprint('selected on validation:', r['selected_survey_model'])\nprint(pd.read_csv(RESULTS / 'survey_metrics.csv').to_string(index=False))"),
        ('md', '## 실제 평가 로직\n공통 모듈은 모델마다 predict를 새로 호출합니다. 아래 코드는 실제 실행 함수의 소스를 확인합니다.'),
        ('code', "import inspect\nfrom reanalysis.pipeline import predict_probability, evaluate, preprocessor\nprint(inspect.getsource(evaluate))\nprint(inspect.getsource(predict_probability))\nprint(inspect.getsource(preprocessor))"),
        ('md', '![모델 비교](reanalysis/assets/model-comparison.png)\n\n테스트 점수만 보면 다른 모델의 macro F1이 더 높을 수 있어도 선택을 사후 변경하지 않았습니다. 원본 공유 구조를 83개 입력으로 새로 학습한 보조 실험은 다수 클래스로 붕괴했습니다. 이 보조 모델은 **원본 팀 저장 모델이 아니며**, 팀 성능으로 해석하지 않습니다.')])
    create_notebook('03 - 모델 추가 학습.ipynb', [intro, ('md', '# 03. 팀 공통 저장 모델의 추가 학습 재검증\n개인 최고 모델을 직접 제공했다고 주장하지 않습니다. 팀원 결과를 비교해 공통 모델을 정하고 추가 분석을 진행한 협업 단계입니다. 여기서는 실제 제공된 `base_model.keras`를 사용합니다.\n\n원본 인코더·결측치 처리 객체는 보존되지 않아 survey와 원본 코드로 복원했습니다. 저장 scaler와 재구성 scaler의 수치 일치를 확인했지만, 모델의 전체 학습 이력까지 증명하지는 못합니다.'), setup,
        ('code', "r = json.loads((RESULTS / 'legacy_shared_results.json').read_text(encoding='utf-8'))\nprint('scaler match:', r['scaler_matches_original_split'])\nprint('selected on validation:', r['selected_by_validation'])\nprint(pd.read_csv(RESULTS / 'legacy_shared_metrics.csv').to_string(index=False))"),
        ('md', '## 수정한 핵심\n- 변환할 때마다 중앙값/인코딩을 다시 학습하지 않습니다.\n- 78개 입력 이름과 순서를 저장된 scaler에 맞춥니다.\n- 추가 학습 실험마다 원본 모델을 독립적으로 다시 불러옵니다.\n- 모델별 예측을 다시 계산합니다.\n- 추가 학습 850건을 680/170건으로 나누고 검증 기준을 먼저 정합니다.\n- 350건 평가는 이미 과거에 살펴본 데이터이므로 새 블라인드 벤치마크라고 표현하지 않습니다.'),
        ('code', "import inspect\nfrom reanalysis.legacy_shared import LegacyPreprocessor\nprint(inspect.getsource(LegacyPreprocessor))"),
        ('md', '## 해석 한계\n단일 seed·작은 표본의 재검증입니다. 추가 학습이 항상 개선을 보장하지 않으며, 모델 성능 변화가 서비스 개선 효과를 의미하지 않습니다. 원본 파일은 덮어쓰지 않았습니다.')])
    print('Generated 3 figures; executed and validated 3 reading notebooks; archived exact originals.')


if __name__ == '__main__':
    main()
