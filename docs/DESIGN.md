# Phase 2 — Equivariant Many-Body Pair Transformer for IFC2

연구 가칭이다. 신규성이나 성능 우위가 증명되었다는 뜻이 아니다. 현재 구현은 작은 기준 구현이며 MACE 전체 구현, 완전한 graph transformer, 에너지 모델의 Hessian을 재현하지 않는다.

## 물리량과 변환

`Phi_ij(R) = ∂²E/∂u_i∂u_j = -∂F_i/∂u_j`, 단위 eV/Å².
극벡터 변위의 rank-2 tensor이므로 `Phi' = Q Phi Qᵀ`이며 Q는 det=±1인 O(3) 행렬이다.
`1o ⊗ 1o = 0e ⊕ 1e ⊕ 2e`. 1e는 **축벡터**이고 반전에 부호가 바뀌지 않는다. 개별 offsite block은 비대칭일 수 있으며 전체 Hessian의 대칭성과 모순되지 않는다.

좌표 회전 등변성은 좌표계를 바꿨을 때 tensor가 어떻게 바뀌는가의 조건이다. 회전 sum rule은 평형 구조에서 실제 무한소 강체 회전 변위에 대한 힘/토크 응답 조건이다. 예를 들어 적절한 평형·응력 조건 아래 `sum_j Phi_ij^{alpha beta} r_ij^gamma = sum_j Phi_ij^{alpha gamma} r_ij^beta`. 전자는 아래 irrep 연산으로 보장하지만 후자는 현재 강제하지 않는다. ASR·reciprocity를 만족해도 안정성, rotational sum rule, 모든 구조에 걸친 에너지 적분가능성이 보장되지는 않는다.

## 수식 → shape → 계산

N: primitive atoms, E: convolution images, P: compact offsite groups, M: 모든 shortest images, C: 채널 수, L: geometric lmax, H: attention heads, d: hidden irrep dimension.

| 모듈 | 입력 → 출력 | 수학/역할 |
|---|---|---|
| data.graph | Z[N], x[N,3], cell[3,3], PBC → index[2,E], r[E,3] | ASE 주기 neighbor enumeration, FC 값에 의존하지 않음 |
| geometry | r[E,3] → radial[E,K], Y[E,(L+1)²], envelope[E] | sinc radial과 compact smooth envelope, 거리·각도 분리 |
| embedding | Z[N] → h[N,d] | 0e 원소 정보, 다른 irrep는 초기 0 |
| RadialTP | h_j[E,d], Y, radial → m[E,d] | CG 허용 경로를 radial MLP로 가중 |
| Interaction | m[E,d] → h[N,d] | density sum + optional attention + higher correlation + gate/residual |
| pair_init | h_i,h_j[M,d], Y_pair → p[M,d] | label-free shortest image geometry를 node 환경과 결합 |
| pair update | p[M,d], updated endpoints → p'[M,d] | 모든 layer에서 지속 pair state 갱신 |
| enhanced fusion | TP(p,Linear(h_i⊕h_j)) → [M,d] | 기존 pair state와 최신 양 끝 환경의 곱 상관 |
| readout | p[M,d] → c[M,9] → Phi_image[M,3,3] | 0e/1e/2e 별 equivariant Linear, orthonormal basis 복원 |
| constraint | image tensors → group[P,3,3], image[M,3,3] | reciprocity, image sum, skew divergence projection |
| compact export | pairs[P,3,3] → [N,Nsuper,3,3] | cutoff 밖 0, onsite = 음의 offsite 합 |
| dynamical_matrix | compact FC, q[Q,3] → complex[Q,3N,3N] | phase 평균과 실제 원자 질량, Hermitian repair 없음 |

기본 hidden irreps는 `C×0e + Σ_{l=1..L}(C×lo + C×le)`이다. 따라서 d=C[1+2L(L+2)]이고 C=3,L=2에서 d=51이다. 출력 readout에는 같은 (l,parity)의 copy만 혼합하므로 일반 dense matrix로 irreps를 섞지 않는다. C·L·layers는 config로 변경한다.

```
Z → scalar embedding ─→ node layer 1 ─→ node layer 2
                       ↑ Σm, attention, TP(A,A) ↑
periodic conv graph ────┴───────────────────────┘
shortest image graph → pair init → pair update → pair update → 0e+1e+2e
                                                                ↓
                reciprocity → compact sum → skew projection → onsite ASR → D(q)
```

## 기법별 채택 근거와 비용

아래 T는 CG tensor-product 유효 경로당 contraction을 모두 합친 비용, Kν는 correlation ν의 coupling 경로 수다. 실제 비용은 irrep multiplicity와 e3nn contraction 구현에 따라 달라지므로 단일 O(C²)로 뭉뚱그리지 않는다. 그래프 저장 E·d와 pair 저장 M·d가 기본 메모리다.

| 기법 | 수학과 담는 정보 | IFC 장점·한계 및 중복 | 시간/메모리와 결정 |
|---|---|---|---|
| TP / CG | `(a⊗b)_{LM}=Σ CG a_lm b_l'm'` | 방향 간 결합을 대칭에 맞게 생성. 물리 법칙 자체나 유한 cutoff 밖 상호작용은 학습하지 못함. correlation·message·pair fusion의 공통 연산 | edge O(E T), node O(N T); 채택 |
| SH + radial | `f_k(r)Y_lm(rhat)` | 방향과 거리의 분리. 유한 L/K의 해상도 한계. TP의 입력 기저로 중복 대신 필수 보완 | O(E(K+L²)), 저장 동일; 채택 |
| message passing | `A_i=Σ_j m_ij` | 배위수·이웃 화학환경. 여러 hop도 다체 정보를 만들지만 깊이와 oversmoothing 비용 | O(E T), 저장 O(E d+N d); 기본 경로 |
| MACE correlation | `B_i^{ν,L}=Σ generalized-CG Π_{t=1..ν} A_i` | density trick으로 이웃 ν조합을 명시 열거하지 않고 각도·다체 상관. message depth와 일부 중복. correlation order와 receptive field는 서로 다름 | MACE는 Kν와 channel별 contraction에 의존, 일반 ν거듭제곱 dense 전개는 비쌈 |
| 이번 recursive correlation | `B₂=TP(A,A); B₃=TP(B₂,A)` | 반복 projected product이므로 MACE complete symmetric basis와 같지 않음. 낮은 L에서 중간 coupling 정보를 잃을 수 있음. 이전 layer·비선형 후에는 ν를 엄밀한 전체 body order로 부를 수 없음 | O(N(ν−1)T), 중간 O(Ndν); B/D/E/F 후보 |
| local multi-head attention | invariant scalar score, `a=softmax(score+log envelope)`, equivariant value | 환경별 가중. 현재 scalar embedding과 radial로 score를 만듦; 모든 irrep contraction query/key를 구현한 것은 아님. sum 경로와 중복 가능한 가중 조절 | tensor 비용 O(E T + H E d), 기준 Python segment 구현은 O(H N E) mask 비용. 큰 N 최적화 필요 |
| graph transformer | attention+FFN+residual+position encodings | local attention block을 사용하지만 dense global transformer는 불필요한 O(N²) 연결·메모리와 locality 손실 가능 | global path 미채택, local path C/D/E/F |
| node–edge/pair | `p'=p+g(p,h_i,h_j,Y)` | 두 원자 환경의 기억을 유지해 pair 출력에 적합. 기존 HamGNN에도 있어 신규성 근거 아님 | O(M T), 저장 O(Md·layers); E는 곱 fusion 추가 |
| gate/residual | scalar sigmoid×irrep + skip | 비선형을 좌표 성분별 임의 적용하지 않음. 안정적 최적화, 새로운 공간 정보를 추가하지는 않음 | O((N+M)d)와 작은 gate MLP; 기본 |
| tensor decomposition | trace / skew / symmetric traceless | 정확한 9성분 표현, 작은 1e 별도 loss 가능. 분해 자체는 새 물리 prior가 아님 | O(P·9), 채택 |
| physics residual | `k_parallel uuᵀ+k_perp(I−uuᵀ)+ΔPhi` | bond 방향 강성의 초기 구조. noncentral tensor는 Δ가 표현. coefficients 양수 강제 없음, 안정성 보장 안 함. direct head와 표현 중복 | O(MK) MLP + O(M·9); F에서 실험 |
| 기타 물리 유도 접근 | 에너지 Hessian, long-range electrostatics, equivariant pretrained backbone | Hessian은 global integrability를 주지만 2차 autodiff 비용; NAC는 local GNN으로 대체 불가. pretraining과 delta learning 이득은 데이터 의존 | 이번 구현은 direct IFC. 최신 전체 문헌 검토는 네트워크 제한으로 미완료 |

## sum과 attention, shared와 independent

`W1 Σm + W2 concat_h Σa_h V_h(m)`에서 W1,W2,Vh는 모두 `o3.Linear`다. attention_only config에서는 W1 경로를 제거해 비교할 수 있다. softmax는 coordination count를 정규화하므로 sum 경로를 기본으로 유지한다. cutoff 밖 edge는 logit 계산 전에 제거하고 active edge에도 smooth envelope를 적용한다. tensor component마다 다른 임의 scalar를 곱하지 않는다.

shared 모델은 한 backbone에서 9성분을 출력한다. independent 모델은 symmetric(0e+2e)과 skew(1e)를 각각 별도 backbone으로 계산한다. 잔여 readout 성분은 사용하지 않으므로 총 파라미터와 유효 fitting budget은 완전히 같지 않다. 이 차이를 감추지 않고 parameter count를 기록한다. TRAIN RMS로 각 irrep의 component MSE를 정규화하며 1e=0에 가까운 데이터에서는 1e-4 eV/Å² floor를 쓴다.

## ASR·reciprocity·onsite symmetry를 함께 강제

먼저 image reverse mapping으로 `Phi_e ← (Phi_e+Phi_revᵀ)/2`. compact group별로 image를 **합산**한다. A_e=skew(Phi_e), d_i=Σ_{e:i→j} A_e, L=diag(out_degree)−adjacency. `v=L⁺d`, `Phi_e ← Phi_e−(v_i−v_j)`로 antisymmetric row divergence를 제거한다.

reverse-closed 그래프에서 `A_rev=−A_e`, 따라서 각 connected component의 Σd=0. 위 보정은 reciprocity를 유지하고 L v=d이므로 행 skew sum이 0이다. onsite를 음의 offsite 합으로 정하면 ASR와 onsite symmetry가 동시에 성립한다. cycle-space의 1e는 유지된다(테스트 포함). self-image edge는 L에 기여하지 않지만 reverse와 상쇄된다. 이 투영은 좌표 component에 같은 graph operator를 적용하므로 O(3) 등변성을 유지한다.

L⁺는 구조마다 한 번 geometry만으로 계산한다. 현재 dense pseudoinverse O(N³) 시간/O(N²) 메모리이므로 대규모 primitive cell에는 sparse solver가 필요하다. cut-off 경계에서 graph가 바뀌는 미분과 geometry에 대한 force/Hessian autodiff는 지원 목표가 아니다. FC 파라미터에 대한 gradient는 검증했다.

## compact alias와 D(q)

Phonopy 계약은 `D_ij(q)=Σ_s Phi_{i,s}/sqrt(m_i m_j) × mean_{v∈shortest(s,i)} exp(+2π i q·v)`. v는 primitive fractional displacement이며 basis 원자 위치를 포함한다. 단위는 eV/Å²/amu이고 THz 변환에는 `phonon.unit_conversion_factor`를 사용한다. 음의 eigenvalue는 음의 signed frequency로 표시한다.

image별 예측의 합을 compact label과 맞춘다. compact supervision만으로 각 image IFC는 식별되지 않는다. `pairs.npz:image_fc`는 학습된 분배이며 유일한 정답이라는 의미가 아니다. 모델 group correction을 image에 균등 분배해 합과 reciprocity를 유지한다. compact export와 phonon loss는 Phonopy의 equal phase-average convention을 사용한다. image별 tensor와 phase를 직접 곱한 무한격자 D는 서로 다를 수 있다. supercell이 작아 interaction alias가 심한 데이터에서는 충분한 supercell 또는 image-resolved label이 필요하다. 최단 이미지 밖의 진짜 IFC를 이 자료만으로 복원한다고 주장하지 않는다.

## 학습과 ablation 해석

기본 `L = mean_{l=0,1,2} MSE(c_l(pred)−c_l(raw))/TRAIN_RMS_l²`. 이후 선택적 reciprocal 단계에서 D와 eigenvalue loss를 더한다. 제약은 hard projection이므로 별도의 physics loss 가중치는 현재 없다. 모든 checkpoint는 validation FC loss로 선택한다. 안정화된 실제 연구에는 reciprocal 단계의 checkpoint selection도 validation phonon 성능으로 별도 실험해야 한다.

raw full D, cutoff+onsite D, cutoff+constraints D, predicted D를 구분한다. `cutoff_dm_mae`, `constraint_dm_shift`, `dm_model_mae`는 각각 cutoff 근사·제약 보정·학습 오차를 나타내며 절댓값 평균이므로 합이 total MAE와 같지는 않다.

A: 독립 채널 재구현, B: A+correlation, C: A+attention, D: 둘 다, E: D+enhanced pair product, F: E+physics residual. 모든 A–F에 기본 persistent pair가 있다. E_shared는 E의 shared-backbone 대조군이다. 같은 split/seed/epochs/width/lr를 사용하지만 파라미터 수는 같지 않다. 현재 비교는 기능 검증이며 parameter-matched multi-seed DFT 실험은 미완료다. small 1e, tensor anisotropy, phonon 오차가 실제로 어느 경로에서 개선되는지를 별도로 평가해야 한다.

Polar: supplied Born charge/dielectric를 동일하게 두 포논 계산에 적용하되 모델이 이를 예측하는 것은 아니다. 2D: vacuum supercell 검증은 가능하지만 2D NAC와 flexural rotational constraints 미구현. Magnetic: spin state 입력 없음. 이러한 조건을 가진 universal accuracy는 아직 주장할 수 없다.
