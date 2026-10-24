# Phase 1 — 첨부 코드와 공개 구현 감사

## 확인 범위와 출처

첨부 1은 설계 대화(1,671줄), 첨부 2는 노트북 형식 코드와 설명(7,876줄)이다. 전체 함수·구간 구조를 조사하고 데이터 매핑, backbone, pair decoder, scaling/split, loss, FC 복원 및 포논 비교의 핵심 구간을 직접 읽었다. 노트북 전체를 실행하거나 모든 plotting 경로의 정확성을 검증한 것은 아니다. 첨부 대화의 “코드를 확인했다”는 서술은 이번 검증의 증거로 사용하지 않았다.

실제 내려받아 확인한 공개 코드의 정확한 commit은 `source_versions.json`에 기록했다.

| 구현 | 읽은 핵심 파일 | 라이선스 파일 확인 | 적용 방식 |
|---|---|---|---|
| [HamGNN](https://github.com/QuantumLab-ZY/HamGNN) | `hamgnn/models/hamgnn_conv.py`, `nn/convolution.py`, `nn/interaction_blocks.py`, `nn/attention.py` | GPL-3.0-only, 파일 SPDX와 LICENSE | 개념 비교만 수행, 코드 복사·import 없음 |
| [MACE](https://github.com/ACEsuit/mace) | `mace/modules/blocks.py:EquivariantProductBasisBlock`, `symmetric_contraction.py` | MIT, `LICENSE.md` | 고차 density correlation 개념 참고, 코드 복사 없음 |
| [e3nn](https://github.com/e3nn/e3nn) | `o3/_tensor_product`, `io/_cartesian_tensor.py` 및 설치된 API | MIT | 공개 라이브러리 의존성으로 사용 |
| [Phonopy](https://github.com/phonopy/phonopy) | `harmonic/dynamical_matrix.py`, `structure/cells.py`와 설치된 API | BSD-3-Clause | 구조·FC 입출력 및 독립 수치 기준 |

MACE의 README에 남은 `MIT.md` 링크 대신 실제 `LICENSE.md`를 확인했다. HamGNN의 현재 main에는 correlation과 attention 경로가 이미 존재한다. 이를 추가했다는 사실만으로 신규성을 주장할 수 없다. “HamGNN-V2”라는 이름의 별도 버전을 재현한 것은 아니며, 기록한 main commit을 조사했다. pretrained 모델의 라이선스가 코드 라이선스와 같다고 가정하지 않는다. 프로젝트 자체의 배포 라이선스는 사용자 결정 전까지 지정하지 않았다.

논문 후보: [MACE, 2206.07697](https://arxiv.org/abs/2206.07697), [e3nn, 2207.09453](https://arxiv.org/abs/2207.09453), [Universal SOC Hamiltonian, 2504.19586](https://arxiv.org/abs/2504.19586). 본문 HTML 요청은 환경 프록시의 403으로 실패했다. 논문 본문을 읽었다고 주장하지 않으며 최신 문헌 전체 비교와 신규성 판정은 미완료다.

사용자 제공 데이터: https://zenodo.org/records/20196565 . API와 페이지 모두 프록시 403으로 실패했다. 데이터 제목·파일 크기·라이선스·내용·DFT 방법은 아직 검증하지 못했다. 이 기록을 실제 학습에 사용했다는 주장은 하지 않는다.

## 첨부 모델의 계산 구조

atomic embedding + 원자 물성 → scalar chemical embedding → periodic node/edge message passing → 모든 layer의 node state를 이용하는 persistent target pair → symmetric / antisymmetric 독립 backbone → 0e,1e,2e readout → forward/reverse 평균·차 → compact FC → onsite ASR → D(q), 주파수/DOS.

TRAIN 전용 RMS scaling과 material-folder split, convolution graph와 FC pair의 분리, shortest-vector mask, cutoff target과 full target의 분리는 유지할 가치가 있다. 기존 property table, magnitude sampling, 3-stage 채널별 freezing, 캐시 pickle, MP API/DOS UI는 이번 최소 구현에 옮기지 않았다. 표준 데이터 계약과 수학적 역할을 재사용하고 코드는 독립 작성했다.

## 확인한 오류와 위험

1. **onsite ASR와 reciprocity의 동시 만족 문제** — 첨부 478–486, 1652–1700, 2838–2863줄. offsite reciprocity만 맞춘 뒤 `Phi_ii=-sum_offsite`를 적용하면 onsite가 비대칭일 수 있다. 1e row-divergence가 0이어야 한다. 새 테스트는 이 반례를 생성하고 raw D가 non-Hermitian임을 검출한다. 새 구현은 skew divergence projection 후 onsite를 복원한다.
2. **Hermitian repair에 의한 오류 은폐** — 첨부 498–545, 1698 및 reciprocal loss. 항상 `(D+D†)/2`를 적용하므로 문제가 숨겨진다. 새 `dynamical_matrix`에는 보정이 없고 eigenvalue 계산 전에 잔차를 검사한다. 외부 Phonopy가 보정하는 경우에도 원본 잔차를 보고한다.
3. **shortest-vector 축 추측의 모호성** — 366–391줄. compact dense format에서 `(n_prim,n_super)`이면 swap하는 분기는 두 수가 같을 때 올바른 축을 잘못 뒤집을 수 있다. 새 코드는 공식 `[super,primitive]` 계약만 허용하고 미지원 layout은 명시적으로 실패한다. `1×1×1`과 `2×2×2`, conventional→primitive 및 full FC 변환을 검사한다.
4. **compact label과 image label의 차이** — 548–706, 1255–1360, 3016 이후. 기존 decoder는 shortest image 예측을 평균하여 compact label과 맞추고, 이후 explicit image inference에서는 multiplicity=1로 바꾼다. 이를 무한 격자 개별 IFC와 동일시할 근거가 없다. 새 모델은 image tensor를 합산해 compact label을 예측한다. 여전히 finite-supercell alias를 유일하게 풀 수 없으므로 이 불확실성을 출력 계약에 표시한다.
5. **phase convention 탐색의 식별성** — 578–607줄. ±phase와 transpose 후보를 고르는 방식은 실수/centrosymmetric fixture에서 여러 후보를 구분하지 못한다. 새 코드는 공식 `exp(+2π i q·s)`와 Cartesian block 순서를 고정하고 imaginary D가 0이 아닌 다원자 fixture로 검증한다.
6. **attention cutoff** — 공개 HamGNN `AttentionAggregation*`에서 cutoff를 logit에 곱한 뒤 softmax한다. logit 0도 정규화에 참여하므로 이 연산 자체는 완전 제외를 보장하지 않는다. 전체 호출자가 edge를 미리 제거할 수 있으므로 모델 전체의 버그라고 단정하지 않는다. 새 코드는 inactive edge를 normalization 전에 제거하고 내부에서도 log-envelope를 사용한다.
7. **neighbor cap과 동률** — 기존 max-neighbor 절단은 거리 동률에서 원자 번호에 의존할 수 있다. 새 그래프는 cutoff 내 이웃을 모두 사용한다. 초대형 구조의 비용은 별도 문제다.
8. **raw target 변조** — 기존 PROJECT_TARGET_ASR은 레이블을 바꿀 수 있다. 새 FC loss는 raw label을 유지하고 constraint로 표현할 수 없는 오차를 별도 기록한다. reciprocal loss만 모델과 동일한 cutoff+constraint 공간의 reference를 사용한다.
9. **split 의미** — material folder 분리만으로 같은 물질의 여러 변형·supercell 누수를 막을 수 없다. 새 loader는 상대 경로 ID를 사용하며 `--split`을 지원한다. 실제 데이터에서는 parent-material ID로 폴더/분할을 먼저 정리해야 한다. 합성 6개 구조는 유사 계열의 변형이므로 보편적 물질 일반화 실험이 아니다.
10. **기타 물리** — 구조와 원자번호만으로 서로 다른 자성 상태, 전하 상태, 전자 온도의 IFC를 구별할 수 없다. polar crystal의 장거리 dipole 항과 2D Coulomb 경계조건은 local cutoff 모델로 해결되지 않는다. BORN/NAC 입력은 비교에 사용하지만 예측하지 않는다.

## 감사 후 결론

첨부 코드를 ground truth로 사용하지 않는다. 테스트 기준은 선형대수 항등식, 독립적인 Phonopy 계산 및 대칭 변환이다. 첨부 baseline을 동일하게 재현했다는 주장도 하지 않는다. A는 그 핵심 구성(독립 symmetric/antisymmetric fitting, equivariant messages, persistent pairs)을 사용하는 축소 재구현이며, 기존 checkpoint/전체 학습법과의 직접 성능 비교는 아직 남아 있다.
