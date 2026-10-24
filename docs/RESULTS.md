# Phase 4–6 — 실행 결과와 연구 해석

이 문서의 모든 학습 수치는 **합성 harmonic fixtures**에서 나온 실제 실행 결과다. Zenodo/DFT 성능 결과가 아니다. Python 3.12, PyTorch CPU, float64, torch threads=1. 패키지 버전은 requirements-lock.txt를 따른다. GPU 없음.

## 물리 및 실행 검증

`python -m pytest -q --junitxml=artifacts/tests.xml`: **17 passed, 0 failed, 0 skipped**. conventional/primitive 테스트의 auto primitive 및 supercell point-group 관련 Phonopy 경고 2개가 남는다. 경고를 숨기거나 테스트를 skip하지 않았다.

검증 범위: orthonormal FC 분해/복원, proper/improper rotation, 주기적 원자 이동·순열, cutoff 밖 attention 제외, image reciprocity·compact 합산, nonzero 1e 유지, ASR·onsite symmetry, 보정 전 complex Hermiticity, Phonopy D(q) 비교, finite-difference gradcheck, forward/backward 및 optimizer step, compact/full 입출력, conventional→primitive 매핑, checkpoint reload, label-free prediction, plot 생성, BORN/NAC, cutoff audit.

합성 데이터 6개에 대한 raw D와 Phonopy의 최대 차이: **5.552e-17 eV/Å²/amu**. 테스트 허용오차는 2e-12. 단일 원자의 실수 D만 비교하지 않고 imaginary D가 있는 다원자 구조를 사용했다.

## 동일 조건의 짧은 ablation

각각 train 4 / validation 1 / test 1, 같은 material IDs와 seed 17, 8 epochs, width 3, layers 2, lr 0.003. test는 checkpoint 선택에 사용하지 않았다. 같은 구조 계열의 합성 변형이므로 universal generalization 실험이 아니다. A는 첨부 모델의 축소 독립 재구현이며 원본 모델의 정확한 재현 결과가 아니다.

| 모델 | Parameters | FC MAE (eV/Å²) | 1e MAE | Frequency MAE (THz) | CPU train (s) | Imaginary modes* |
|---|---:|---:|---:|---:|---:|---:|
| A | 27234 | 0.064717 | 0.0009464 | 5.1432 | 4.64 | 7 |
| B | 32094 | 0.062320 | 0.0009424 | 5.3365 | 5.68 | 11 |
| C | 28166 | 0.063348 | 0.0009393 | 5.2674 | 9.88 | 11 |
| D | 33026 | 0.061700 | 0.0009592 | 4.8906 | 8.90 | 8 |
| E | 38066 | 0.059924 | 0.0010135 | 4.6117 | 15.01 | 2 |
| F | 38172 | 0.057250 | 0.0009323 | 3.8852 | 8.72 | 0 |
| E_shared | 19033 | 0.059946 | 0.0009734 | 4.6188 | 3.83 | 2 |
| E_attention_only | 38066 | 0.059863 | 0.0009767 | 4.5962 | 10.54 | 2 |

*3 q-points × 9 branches에서 −0.1 THz 아래 mode 수. near-zero Gamma 수치 오차는 이 threshold로 제외. GPU peak memory는 모든 행에서 null(측정 불가)이다.

A–F의 파라미터 수는 27,234–38,172로 약 40% 차이가 난다. parameter-matched 비교가 아니므로 순위에 architecture 효과와 capacity 효과가 섞인다. CPU 시간은 모델 생성 시간을 제외하고 epoch 실행을 측정했으며 일부 실험은 다른 검증 작업과 겹쳤다. 정밀한 속도 benchmark로 쓰지 않는다. 모든 결과 컬럼은 `docs/ablation_results.csv`와 `artifacts/ablation/results.json`에 있다.

해석: B는 A보다 FC 오차가 작지만 이 짧은 실행에서 주파수 오차는 더 크다. E의 pair product는 D보다 전체 FC 오차가 작지만 1e 오차는 커졌다. F의 residual이 이 데이터에서 가장 작은 FC 오차를 보였지만 synthetic label 자체가 bond stiffness 형태를 포함하므로 유리한 prior라는 편향이 있다. E_shared는 절반의 parameter로 E와 비슷한 전체 FC 오차를 보였다. attention_only는 E와 비슷하며 sum 경로의 우위를 입증하지 못했다. 단일 seed, 작은 데이터, 미수렴 실행으로 인과적 성능 결론을 내리지 않는다.

## 더 긴 fitting 확인 (별도 실행)

동일한 split, E_shared, real-space 120 epochs + reciprocal 20 epochs. 이 실행은 위 8-epoch 표와 공정한 비교 대상이 아니며 최적화 경로가 실제로 학습 가능한지 확인하는 용도다.

- Test FC MAE: 0.01188166 eV/Å²
- Test 0e / 1e / 2e MAE: 0.05136297 / 0.00052349 / 0.00411475
- Test D MAE: 0.00211941 eV/Å²/amu
- Test frequency MAE: 0.08621493 THz (고정 3 q-points)
- Rotation / reciprocity / ASR / Hermiticity error: 1.110e-16 / 0.000e+00 / 4.441e-16 / 1.430e-17
- Imaginary modes: 0; CPU training 81.41 s; parameter count 19033

Checkpoint: `artifacts/fit_check/best.pt`. 전체 history/config/split/metrics를 저장했다. `artifacts/fit_check/phonons/dispersion.png`는 synthetic_000의 예시 q-path plot이며 test 3-q-point metric과 동일한 평가 집합이 아니다. 초기 8+2 epoch smoke plot은 `artifacts/phonons/dispersion.png`에 보존했다. reference가 합성이므로 DFT 비교 그래프라고 부르지 않는다.

## cutoff 오차와 설치 재현성

합성 데이터 평균 cutoff D MAE는 5.5 Å에서 1.889284e-17, 3.0 Å에서 1.987242e-02 eV/Å²/amu다. 원래 합성 FC는 5.5 Å 내에 생성되었으므로 앞의 값이 거의 0인 것은 정의상 예상된다. 실제 물질의 cutoff 수렴성을 입증하지 않는다.

CPU venv 설치와 `scripts/setup.sh` 재실행이 성공했고 `pip check`는 broken requirements가 없다고 보고했다. checkpoint 추론, compact export, Phonopy plot, ASE CIF 입력 추론을 실제 실행했다. 이 프로젝트에 상시 서비스는 필요 없다.

환경 draft에 install_script와 start_skill, zenodo.org/arxiv.org allowlist 추가를 저장했다. draft 저장은 publication이나 새 task 복원 검증이 아니다.

## 남아 있는 실제 연구 작업

1. 환경 설정을 저장·게시하여 Zenodo/arXiv 접속 허용을 적용한 뒤 metadata·license·단위·파일 용량을 확인하고 실제 데이터를 checksum 검증하여 내려받기.
2. 실제 compact index, supercell alias, raw ASR/reciprocity, q convention, cutoff tail, 부모 물질 단위 split 감사 후 DFT 학습.
3. 원본 baseline 재현, parameter-matched 및 다중 seed A–F/attention-only/shared 비교, 충분한 q-grid에서 phonon 오차와 불안정 mode 평가.
4. 논문 본문 비교 및 최신 선행 연구 조사 후 신규성 판단. 완전한 MACE basis와 recursive TP의 차이를 성능·비용으로 검증.
5. magnetic/polar/2D 입력 및 장거리 물리, rotational sum rule, 에너지 적분가능성, GPU scale 검증.

현재 완료한 것은 실행 가능한 연구 프로토타입과 독립 물리 테스트, CPU 학습·추론 실험이다. 실제 데이터 검증과 최종 연구 목표 달성은 아직 완료되지 않았다.
