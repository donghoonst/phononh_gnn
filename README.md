# IFC2 Pair GNN — 연구용 기준 구현

## 처음 읽는 분을 위한 쉬운 설명서

**[공과 용수철로 이해하는 Phonon GNN](docs/BEGINNER_GUIDE_KO.md)** · **[PDF 보기 — 25쪽](docs/BEGINNER_GUIDE_KO.pdf)** · **[PDF 다운로드](https://github.com/donghoonst/phononh_gnn/raw/refs/heads/main/docs/BEGINNER_GUIDE_KO.pdf)**

공과 용수철의 움직임에서 시작해 원자 이웃 관계, GNN 학습, 힘 상수, 포논 그래프까지 차근차근 설명합니다. 12개 장과 수식·용어·검증 근거 부록을 담았으며, 노트북 실행 순서와 현재 확인된 성과·남은 문제도 함께 정리했습니다.

## 한국어 통합 주피터 노트북

**[IFC2_All_in_One_Korean.ipynb 열기](notebooks/IFC2_All_in_One_Korean.ipynb)** · **[노트북 파일 다운로드](https://github.com/donghoonst/phononh_gnn/raw/refs/heads/main/notebooks/IFC2_All_in_One_Korean.ipynb)**

노트북 하나에 데이터 처리·GNN·loss·학습·추론·Phonopy 비교·테스트·ablation 코드를 직접 담았습니다. 별도 `ifc2` 패키지나 저장소 clone 없이 실행할 수 있으며, 한국어 설명과 상세 주석으로 **수식 → tensor shape → 연산 → 결과**를 연결합니다. 실행한 출력과 그래프도 포함합니다.

1. 위 `.ipynb` 파일을 내려받아 JupyterLab 또는 Google Colab에서 엽니다. GitHub에서는 읽기만 가능합니다.
2. Python 3.12 이상 kernel을 선택하고 처음부터 순서대로 실행합니다. 첫 셀은 없는 외부 라이브러리를 설치하며 기존 PyTorch는 유지합니다.
3. 기본 실습은 합성 데이터, CPU, 17개 테스트, 작은 학습과 A–F/대조군 비교입니다. 실제 DFT 결과로 해석하지 마세요.
4. 14절에서 모델/epoch를, 18절에서 실제 데이터 경로를 설정합니다. 결과는 실행한 폴더의 `ifc2_notebook_runs/실행ID/`에 저장됩니다.

로컬 JupyterLab 실행 예:

```bash
python -m pip install jupyterlab
python -m jupyterlab IFC2_All_in_One_Korean.ipynb
```

개발자가 노트북을 재생성/검증하려면 `pip install -e '.[test,notebook]'`, `python scripts/build_notebook.py`, `python scripts/validate_notebook.py`를 사용합니다. 생성기는 원본 구현에 한국어 설명/주석을 붙이고, 검증기는 원본과 함수 AST가 같은지 확인한 뒤 **프로젝트 import를 차단한 임시 폴더**에서 전체 노트북을 실행합니다. 결과는 `docs/notebook_validation.json`에 기록됩니다.

## 모듈형 프로젝트

결정 구조에서 3×3 IFC2를 직접 예측하는 실행 가능한 PyTorch/e3nn 프로젝트다. 코드·학습·추론·Phonopy 비교 경로를 CPU에서 검증했다. **실제 DFT 성능과 연구 신규성은 아직 검증하지 않았다.** 사용자 제공 Zenodo 데이터는 현재 환경 프록시 403 때문에 내려받지 못했다. 결과는 명시적으로 표시한 합성 harmonic 데이터의 smoke 실험이다.

- [기존 코드 감사](docs/AUDIT.md): 확인한 근거, 라이선스, 오류·미검증 사항
- [수식·tensor shape·계산 흐름](docs/DESIGN.md): 10개 기법 비교, 물리 제약과 연구 가설
- [실행 결과](docs/RESULTS.md): 실제 실행한 테스트와 ablation, 한계
- [전체 프로젝트 ZIP](downloads/phononh_gnn-research-prototype.zip): 코드, 합성 데이터, checkpoint, 실행 결과 포함
- [합성 데이터 포논 비교 그래프](docs/figures/synthetic_phonon_comparison.png)

## 설치와 빠른 실행

클라우드 CPU 환경(Python 3.12)에서:

```bash
cd /workspace/phononh_gnn
bash scripts/setup.sh
source /workspace/ifc2-env/bin/activate
export OMP_NUM_THREADS=1
export MPLCONFIGDIR=/workspace/.cache/matplotlib
export XDG_CACHE_HOME=/workspace/.cache
python -m pytest -q
ifc2 synthetic --out data/synthetic --count 6
ifc2 audit --data data/synthetic --out artifacts/data_audit.json
ifc2 train --data data/synthetic --config configs/small.yaml --out artifacts/smoke --epochs 8 --reciprocal-epochs 2
ifc2 predict --checkpoint artifacts/smoke/best.pt --material data/synthetic/synthetic_000 --out artifacts/prediction
ifc2 compare --checkpoint artifacts/smoke/best.pt --material data/synthetic/synthetic_000 --out artifacts/phonons
ifc2 ablate --data data/synthetic --out artifacts/ablation --epochs 8 --width 3
```

일반 머신에서는 적합한 PyTorch를 설치한 가상환경에서 `pip install -e '.[test]'`를 사용한다. `requirements-lock.txt`는 이번 CPU 환경의 전체 설치 버전 기록이다. GPU 학습은 적절한 CUDA PyTorch 설치 후 `train --device cuda`로 요청할 수 있지만 이 환경에는 GPU가 없어 검증하지 않았다. `setup.sh`는 클라우드 CPU 설치 전용이다.

## 실제 데이터

```
data/real/
  material_unique_id/
    phonopy.yaml
    FORCE_CONSTANTS
    BORN                 # optional
```

FC는 eV/Å², 길이는 Å, 질량은 amu라는 계약이다. 코드가 원래 계산기의 단위를 자동 판별·변환하지 않는다. compact와 full FC를 읽어 primitive/supercell mapping을 검사한다. 자료의 DFT provenance, 단위, magnetic state, supercell 수렴과 라이선스를 먼저 확인해야 한다. 동일 물질의 변형들은 같은 split에 넣는다.

```bash
python scripts/fetch_zenodo.py --record 20196565
# 파일 목록/용량/라이선스 확인 후 정확한 이름으로 선택 다운로드
python scripts/fetch_zenodo.py --record 20196565 --file ACTUAL_FILENAME --out data/downloads
ifc2 train --data data/real --config configs/small.yaml --out artifacts/real --epochs 100 --split split.json
ifc2 compare --checkpoint artifacts/real/best.pt --material data/real/MATERIAL --out artifacts/real_phonons --qpoints qpath.txt
```

download helper는 checksum을 검증하고 자동 압축해제하지 않는다. 현재 네트워크 차단으로 실제 다운로드 실행은 검증하지 못했다. split.json 형식은 `{"train":["id1"],"validation":["id2"],"test":["id3"]}`이며 IDs는 data root 상대 폴더 경로다. 같은 데이터셋 전체를 겹침 없이 포함해야 한다. 모델 선택에는 validation만 사용한다.

실제 학습 전에 `ifc2 audit --data data/real --out artifacts/real_audit.json`을 실행해 raw ASR/reciprocity/Hermiticity, Phonopy 비교와 cutoff 오차를 확인한다. 동일 cutoff를 train config에도 사용한다. 서로 다른 cutoff를 비교하면 truncation budget을 추정할 수 있다.

## 정답 없이 추론

`predict --material DIRECTORY`는 phonopy.yaml만 있으면 된다. FORCE_CONSTANTS는 필요하지 않다. ASE가 읽는 CIF/POSCAR 같은 구조 파일만 있는 경우:

```bash
ifc2 predict --checkpoint artifacts/smoke/best.pt --material structure.cif --structure --supercell 3 3 3 --out artifacts/structure_prediction
```

실제 조성으로 학습되지 않은 합성 checkpoint의 출력을 물리 예측으로 신뢰하면 안 된다. 임의 조성과 원자 수를 입력할 수 있다는 기능과 universal accuracy는 다른 주장이다. supercell은 compact 출력 규격이며 alias를 줄일 만큼 충분히 크게 선택해야 한다.

`FORCE_CONSTANTS`, `phonopy.yaml`, `pairs.npz`, `diagnostics.json`이 출력된다. npz의 `image_fc[M,3,3]`는 `image_owner[M]`별로 합하면 `compact_pair_fc[P,3,3]`가 된다. `translation[M,3]`와 pair 원자 index로 periodic image를 식별한다. compact label에 여러 image가 대응하면 그 분배는 유일하게 식별되지 않는다. 자세한 convention은 설계 문서를 따른다.

`compare`는 reference/prediction에 동일한 Phonopy 구조·질량·q path를 사용한다. `--nac`는 공급된 BORN/dielectric를 양쪽에 동일 적용하며 이 값들을 예측하지 않는다. `--qpoints`가 없으면 예시 fractional 경로를 사용하므로 일반적인 물질별 high-symmetry band path라고 부르지 않는다. 출력 그래프·npz·JSON의 reference가 DFT인지 합성인지는 입력 데이터 provenance에 달려 있다.

## 코드 구성

| 모듈 | 역할 |
|---|---|
| `data.py` | compact/full FC, 모든 shortest vectors, geometry graph, split |
| `model.py` | irrep embedding, message/correlation/attention, persistent pair, tensor head |
| `physics.py` | decomposition, reciprocity+ASR projection, raw complex D(q) |
| `loss.py` | train-only RMS, real-space/reciprocal/eigenvalue loss |
| `training.py` | 학습, validation checkpoint, 테스트 metrics, A–F/E_shared CSV |
| `inference.py` | label-free 추론, compact export, Phonopy 분산 그래프 |
| `synthetic.py` | DFT와 구분된 harmonic fixture 생성 |
| `tests/` | 등변성·mapping·물리·gradient·checkpoint·NAC 검증 |

각 experiment directory에 split/config/history/metrics/best.pt가 저장된다. config에서 width/layers/lmax/cutoffs/correlation(1,2,3)/attention/head count/aggregation/persistent pair/enhanced fusion/independent heads/physics residual을 바꾼다. `independent_heads`가 backbone 선택 옵션이다. MACE pretrained backbone이나 HamGNN 원본 backend는 제공하지 않는다.

## 현재 한계

실제 DFT 데이터 학습, parameter-matched 다중 seed 비교, 원본 notebook baseline의 정확한 재현, full MACE symmetric basis와 비교, rotational sum rule 및 global energy integrability, 장거리/2D electrostatics, magnetic inputs, 대규모 GPU 성능과 신규성 판정은 남아 있다. attention segment loop와 dense graph pseudoinverse는 작은 구조를 위한 기준 구현이다. 임의 안정성을 강제하지 않아 imaginary mode가 생길 수 있으며 이를 metrics에 보고한다. 완전한 universal 연구 모델이 완성되었다고 해석하지 않는다.
