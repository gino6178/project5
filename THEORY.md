# ReRoom — Theoretical Framing

**Topology-preserving Hierarchical Geometry-conditioned Continuous Flow for
Reference-guided Scene Retargeting.**

論文定位:把工程做法抽象成理論主軸。ReRoom 的核心不是「一堆優化技巧」,而是一個
**層級、幾何條件化的連續流**,其歸納偏置在<em>度量形變</em>下保持場景拓撲。資料生產
與 optimizer 精簡為 Implementation Details;下面三個命題是理論主軸,每個都有數值 /
實證證據(見 §5)。

---

## 1. Notation

| 符號 | 意義 |
|---|---|
| \(\mathcal{G}=(\mathcal{V},\mathcal{E})\) | 場景拓撲圖:節點=物件,邊=關係(帶型別 + 彈性 \(\alpha_{ij}\)) |
| \(\pi:\text{child}\to\text{head}\) | motif 層級誘導的 parent map(head = motif 首件) |
| \(x_i=(p_i,\theta_i)\in SE(2)\) | 物件 \(i\) 狀態:位置 + 朝向,正規化到房間 MRR frame |
| \(P_t\) | 目標房邊界(32 取樣點 × 6 維,含法線) |
| \(\lambda\) | REL_SCALE = 3 m,child 度量偏移正規化常數 |
| \(v_\theta\) | flow 速度場;\(\tau\in[0,1]\) flow-matching 時間 |
| \(\Phi\) | 可行性能量(boundary / collision / clearance / wall-pull) |
| \(S_\text{rel},R_\text{col}\) | 關係保持分數、碰撞率(評估用) |

---

## 2. Hierarchical Reparameterization

定義層級狀態映射 \(\Phi_\mathcal{G}:x\mapsto z\)。head 保絕對座標;child 改成
**parent-local 度量**座標:

$$
z_h = x_h\ (\text{絕對}),\qquad
z_c = \Big(\tfrac{1}{\lambda}R(-\theta_{\pi(c)})(p_c-p_{\pi(c)}),\ \theta_c-\theta_{\pi(c)}\Big).
$$

因 parent 皆為 head,\(\Phi_\mathcal{G}\) 單步可逆(\(\Psi_\mathcal{G}=\Phi_\mathcal{G}^{-1}\);
數值 round-trip 誤差 \(1.3\times10^{-7}\))。Continuous Flow Matching 在 \(z\) 空間:

$$
z_\tau=(1-\tau)z_0+\tau z_1,\qquad
\mathcal{L}=\big\|v_\theta(z_\tau,\tau,\mathcal{G},P_t)-(z_1-z_0)\big\|^2,
$$

且用 **informative prior**(非高斯):\(z_0=\Phi_\mathcal{G}(\mathcal{T}(S_\text{ref},P_t))+\sigma\epsilon\),
\(\mathcal{T}\) 為 reference 佈局投影進目標房的仿射。這把「生成」重構為「整流」。

---

## 3. Propositions

### Proposition 1 — Metric-stretch invariance of intra-motif topology

**Statement.** 設目標房受各向異性縮放 \(A=\mathrm{diag}(s_x,s_y)\)(長寬比改變),
\(P_t\to A\!\cdot\!P_t\)。在層級(度量-相對)參數化下,motif 內部相對構型不變:

$$
z_c(A\!\cdot\!P_t)=z_c(P_t)\quad\Longrightarrow\quad
\|p_c-p_{\pi(c)}\|\ \text{保持(至多至 parent 位姿)}.
$$

**Proof (sketch).** \(z_c\) 儲存的是度量偏移 \(\|p_c-p_{\pi(c)}\|/\lambda\) 及相對朝向,
二者皆與房間尺度無關。縮放 \(A\) 僅作用於絕對量,被 parent 的 \(z_h\) 吸收;child 的
\(z_c\) 不含房間尺度變數,故不變。解碼 \(\Psi_\mathcal{G}\) 在該(尺度無關的)度量偏移處
重建 child,即得結論。∎

**Contrast.** 絕對座標流預測 child 的 *normalized* 偏移,其度量大小 \(\propto\) 房間尺寸,
放大房時 intra-motif 距離按比例拉大 → 拓撲<em>不</em>保持(即觀察到的「大房把 motif 拉散」)。

**Evidence.** 數值:0.6 m 偏移在 3 m 與 4.5 m 房都給 \(z_c=0.2\)。實證:1.35× 大房
motif_pass 25%→42%、1.0× raw \(S_\text{rel}=0.98\)(raw = 純 flow,無後處理)。

---

### Proposition 2 — Energy-guided flow with Jacobian-aware pullback

**Statement.** 取樣是能量引導 ODE:每步用可行性能量 \(\Phi\)(定義於<em>絕對</em>座標)
在預測終點 \(\hat z_1\) 的 score 修正速度。因流在重參數化流形 \(z\) 上,梯度須經 pullback:

$$
z\leftarrow z+d\tau\,v_\theta-d\tau\,\mathrm{ramp}(\tau)\,\nabla_z\Phi\big(\Psi_\mathcal{G}(\hat z_1)\big),
\qquad
\nabla_z\Phi=J_{\Psi_\mathcal{G}}^{\top}\,\nabla_x\Phi,
$$

其中 \(\hat z_1=z_\tau+(1-\tau)v_\theta\),\(\mathrm{ramp}(\tau)=\max(0,(\tau-0.5)/0.5)\)。

**Jacobian.** \(J_{\Psi_\mathcal{G}}=\partial x/\partial z\):
- head:\(\partial x_h/\partial z_h=I\)。
- child 對自身:\(\partial p_c/\partial z_c=\lambda R(\theta_{\pi(c)})\)。
- child 對 parent:\(\partial p_c/\partial z_{\pi(c)}\) 含 parent-motion 項
  \(\big(I+\tfrac{\partial R(\theta_\pi)}{\partial\theta_\pi}\lambda z_c\big)\) &mdash; 即「動 parent 帶動整組 child」的鏈式法則顯式化。

**Implementation.** 即 <code>decode → nudge (world) → encode</code>(<code>sample.py</code>):
先 \(\Psi_\mathcal{G}\) 解到世界座標算 \(\nabla_x\Phi\),施加後再 \(\Phi_\mathcal{G}\) 編回 \(z\)。
這是 pullback 的離散實現,把 in-sampling guidance 放進標準 energy-guided flow 理論。

---

### Proposition 3 — Topology-preserving projection

**Statement.** 輸出 \(x^\star=\Pi(\Psi_\mathcal{G}(z_1))\),\(\Pi\) 為**單步、確定性、無梯度**
的幾何投影(Regularity snap:槽位 / 貼牆 flush / Manhattan,對應 eq (37) 的 constraint
projection)。\(\Pi\) 保持關係拓撲:

$$
\big|S_\text{rel}(\Pi(x))-S_\text{rel}(x)\big|\le 0.005\quad(\text{實測 }\Delta=-0.004),
$$

施加 Manhattan / 貼牆對齊(平均位移 19 cm),不改拓撲。(碰撞數字見 Prop 3' 的 36-cell 消融;早期 4-seed 的 2.4%/1.8% 為小樣本雜訊,已以大樣本 + 變異數取代。)

**Interpretation.** 生成模型 \(v_\theta\) 擁有拓撲與近可行分佈;\(\Pi\) 只貢獻一個<em>拓撲中性</em>
的**單步**度量對齊。線上系統即 **單一網路 \(v_\theta\) + 單步投影 \(\Pi\)**(跟 LEGO-Net
一樣輕盈),無需 test-time 多步能量優化。故 \(\Pi\) 在論文中降格為 Implementation Detail,
而非結構來源 &mdash; 且不主張「\(\Pi\) 可忽略 / <5%」(位移 19 cm 不小),而主張
「\(\Pi\) topology-preserving」。多步 polish 保留為選用精修(碰撞數字見 Prop 3' 36-cell 表)。

---

### Proposition 3′ — Differentiable form of the projection (D3)

**Statement.** \(\Pi\) 的三個規整(貼牆 flush + 朝向 + Manhattan + 非碰撞)可寫成平滑能量,
以 \(K\) 步展開式梯度下降實現為一個**可微**投影 \(\Pi_\theta\):

$$
\Pi_\theta(x)=\big(K\text{ 步展開梯度下降}\big)\ \text{on}\
\underbrace{w_a\lVert x-x_0\rVert^2}_{\text{錨定提案}}
+w_f E_\text{flush}+w_o E_\text{ortho}+w_c E_\text{col}.
$$

可微 ⇒ 既能推理時取代硬 snap,又能在訓練時展開進 loss(OptNet / implicit-diff 精神,
顯式展開、無外部依賴)。

**Evidence.**（raw `flow_bfresh` 輸出,36 cells = 12 refs × 3 尺寸,mean ± per-cell std)

| 投影 | \(R_\text{col}\) | snap% | \(S_\text{rel}\) |
|---|---|---|---|
| 無(raw flow) | 1.15±2.3% | 80±24% | **0.707**±.32 |
| 硬 snap(shipped) | 1.59±2.6% | 87±20% | 0.676±.31 |
| **可微投影 \(\Pi_\theta\)** | 1.04±1.9% | 80±23% | **0.703**±.32 |
| 25 步 polish | 1.23±2.3% | 86±23% | 0.680±.31 |

**Finding.** 碰撞差異都在一個標準差內,**不主張顯著的碰撞勝出**。穩健的方向性效果是
**拓撲保持**:拓撲保持組(raw、\(\Pi_\theta\))守住 \(S_\text{rel}\!\approx\!0.70\),而硬 snap / polish
用約 0.03 的 \(S_\text{rel}\) 換 +6–7 個百分點的貼牆率。\(\Pi_\theta\) 的價值是「在無碰撞代價、
極低延遲下的拓撲保持對齊」,而非碰撞 SOTA。關鍵是**錨定提案**項 —— 少了它純能量下降會
漂移、扯散 motif(\(S_\text{rel}\) 掉到 0.566),有了它 \(\Pi_\theta\) 才是「最近可行點」的投影。把 \(\Pi\)
從工程後處理升級為**神經生成器 + 可微凸投影**的自洽算子。實作 `reroom/retarget/diffproj.py`;
研究延伸,尚未設為 shipped 預設。細節見 `EXTENSIONS.md`。

---

## 4. Architecture as inductive bias

理論主軸的三個歸納偏置,對應具體模組:

| 歸納偏置 | 模組 | 對應命題 |
|---|---|---|
| 度量-相對層級(尺度不變) | parent-relative state + 異質節點 Type-ID + child 遮罩牆面 | Prop 1 |
| 幾何條件化(牆為一等公民) | WallCrossAttention(邊界點為牆 token)+ dense pairwise bias | Prop 2 引導、對齊 |
| 整流先驗(非盲猜) | informative prior \(z_0=\Phi_\mathcal{G}(\mathcal{T}(\cdot))+\sigma\epsilon\) | 全域定位、低 raw \(R_\text{col}\) |

訓練規範(Prop 1 收斂所需):fresh(warm-start 進相對空間 = 壞初始化)、異質節點
Type-ID、child loss 尺度補償 \(\lambda_\text{child}=10\approx1/0.3^2\)(否則 child 局部尺度
梯度被 parent 房間尺度淹沒)。

---

## 5. Empirical evidence(消融與固定測試集)

| 主張 | 證據 | 出處 |
|---|---|---|
| 拓撲是模型本體學的 | \(\Delta S_\text{rel}=-0.004\) 經 polish | polish 消融(4 場景 × 3 尺寸) |
| raw flow 低碰撞 | raw \(R_\text{col}=1.15\pm2.3\%\)(36-cell) | 同上 |
| 度量-相對 → motif 不散架 | 1.35× motif_pass 25→42%、0.75× 50→83%(raw) | 固定測試集,bfresh vs r2m2full |
| raw flow 本體足夠強 | 1.0× raw \(S_\text{rel}=0.98\) | three_sizes_bfresh_raw |
| scale-invariance | 0.6 m → \(z=0.2\) in 3 m & 4.5 m | to_relative/to_world 單元測 |
| 逐步驗證每個模組 | bnd200→r2→r2m2→r2m2full→bfresh 進程表 | 網站「六、結果」 |

固定測試集:12 跨場景 cross pair(真實人類 GT）+ 4 three-sizes,凍結於
<code>outputs/fixed_testset.json</code>;主標的為 PDF §15 關係型指標(\(S_\text{rel},S_\text{motif},R_\text{OOB},R_\text{col}\)),
位置 MAE 僅參考(PDF §4:design intent ≠ 絕對座標)。

---

## 6. Paper positioning

- **主軸(寫成理論)**:§2 層級 continuous flow + Prop 1–3。標題方向
  *Topology-preserving Hierarchical Flow Matching for Scene Retargeting*。
- **精簡為 Implementation Details**:資料生產(forward-deform 因果對 + 守門)、
  optimizer/polish 六能量、regularity snap。以「bounded topology-preserving projection」
  一句帶過,細節進 appendix。
- **與相關工作的區隔**:
  - vs LEGO-Net:借「從擾動整流」但改為<em>跨房 retargeting</em> + 層級度量流(非同房去噪)。
  - vs PhyScene:借 energy-guided 可行性,但物件集來自 reference 遷移而非生成;walkability 為 opt-in。
  - vs flow-matching layout works:新增<em>hierarchical metric-relative</em> 參數化與其形變不變性證明。

實作檔案地圖見 <code>SHIPPED.md</code>;可重現 checkpoint <code>flow_bfresh</code>。
