# 深度學習期中考重點筆記（理論加強版）

## 深度學習五個核心步驟（老師指定必背）
1. **資料準備**：收集 → 清理 → 正規化/標準化 → 資料增強 → train/val/test 劃分 → 打包成 batch  
2. **模型定義**：設計網路層數、激活函數、參數初始化方式  
3. **損失函數與優化器設定**：選擇 Loss + Optimizer + 學習率  
4. **訓練超參數設定**：epoch、batch size、learning rate schedule、正則化、early stopping  
5. **評估與推論**：驗證集指標、測試集評估、inference 時關閉 dropout/BN 訓練模式

---

## 1. 神經網路基礎

### 1-1 單層神經元
$$
z = \mathbf{w}^T \mathbf{x} + b, \quad a = \sigma(z)
$$

### 1-2 常見激活函數（考試最愛這六個）

| 函數         | 公式                                      | 導數                                      | 特性與考點                                 |
|--------------|-------------------------------------------|-------------------------------------------|--------------------------------------------|
| Sigmoid      | $\sigma(z) = \frac{1}{1+e^{-z}}$         | $\sigma'(z) = \sigma(z)(1-\sigma(z))$    | 輸出 0~1，易梯度消失                       |
| Tanh         | $\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}$ | $1 - \tanh^2(z)$                        | 零中心，但仍梯度消失                       |
| ReLU         | $f(z) = \max(0, z)$                      | $f'(z) = \begin{cases}0 & z \le 0 \\ 1 & z > 0\end{cases}$ | 計算快、不飽和，Dying ReLU 問題           |
| Leaky ReLU   | $f(z) = \begin{cases}z & z > 0 \\ \alpha z & z \le 0\end{cases}$ ($\alpha=0.01$) | 同左                                      | 解決 Dying ReLU                            |
| Hard Tanh    | $f(z) = \begin{cases}-1 & z < -1 \\ z & -1 \le z \le 1 \\ 1 & z > 1\end{cases}$ | 在 (-1,1) 內導數為 1                     | 輸出範圍固定 [-1,1]，早期 RNN 常用         |
| Softplus     | $f(z) = \log(1 + e^z)$                   | $f'(z) = \frac{1}{1+e^{-z}} = \sigma(z)$ | 平滑版的 ReLU，數學性質好                  |

### 1-3 反向傳播四個核心公式（必手推）

第 $l$ 層的局部梯度 $\delta^l = \frac{\partial L}{\partial z^l}$

1. 輸出層：  
   $\delta^L = \nabla_a L \odot \sigma'(z^L)$
2. 隱藏層往回傳：  
   $\delta^l = (W^{l+1}^T \delta^{l+1}) \odot \sigma'(z^l)$
3. 權重梯度：  
   $\frac{\partial L}{\partial W^l} = \delta^l (a^{l-1})^T$
4. 偏置梯度：  
   $\frac{\partial L}{\partial b^l} = \delta^l$

---

## 2. 梯度下降變種（表格直接背）

| 優化器       | 更新公式                                                                                  | 重點考點                              |
|--------------|-------------------------------------------------------------------------------------------|---------------------------------------|
| SGD          | $w \leftarrow w - \eta \nabla L$                                                         | 最基礎                                |
| Momentum     | $v_t = \beta v_{t-1} + \nabla L \quad ; \quad w \leftarrow w - \eta v_t$                 | 慣性，$\beta$ 通常 0.9                |
| AdaGrad      | $G_t = G_{t-1} + (\nabla L)^2 \quad ; \quad w \leftarrow w - \frac{\eta}{\sqrt{G_t+\epsilon}}\nabla L$ | 學習率自動衰減，適合稀疏梯度          |
| RMSProp      | $E[g^2]_t = \beta E[g^2]_{t-1} + (1-\beta)(\nabla L)^2 \quad ; \quad w \leftarrow w - \frac{\eta}{\sqrt{E[g^2]_t+\epsilon}}\nabla L$ | 解決 AdaGrad 衰減太快                |
| Adam         | $m_t = \beta_1 m_{t-1} + (1-\beta_1)\nabla L$<br>$v_t = \beta_2 v_{t-1} + (1-\beta_2)(\nabla L)^2$<br>$w \leftarrow w - \eta \frac{m_t}{\sqrt{v_t}+\epsilon}$ | 目前最強，$\beta_1=0.9, \beta_2=0.999$ |

---

## 3. 正則化技術（簡要背誦）
- L2 正則化：在 loss 加上 $\frac{\lambda}{2} \|w\|_2^2$ → 梯度多 $+\lambda w$
- Dropout：訓練時隨機丟 p 比例，測試時乘 (1-p)
- Early Stopping
- Data Augmentation

---

## 4. Batch Normalization（超詳細版，老師最愛考推導）

### 為什麼需要 BN？
解決 **Internal Covariate Shift**：每一層輸入分佈在訓練過程中一直變 → 後面層很難學

### 訓練時（mini-batch）
對第 $l$ 層某個通道的 $B$ 個激活值 $x_1...x_B$：

1. 計算 batch 均值  
   $$\mu_B = \frac{1}{B} \sum_{i=1}^B x_i$$
2. 計算 batch 變異數  
   $$\sigma_B^2 = \frac{1}{B} \sum_{i=1}^B (x_i - \mu_B)^2$$
3. 正規化  
   $$\hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}$$
4. 縮放與平移（可學習參數）  
   $$y_i = \gamma \hat{x}_i + \beta = \text{BN}_{\gamma,\beta}(x_i)$$

### 測試時（Inference）
用整個訓練集的移動平均（running average）：
$$
\hat{x} = \frac{x - \mathbb{E}[x]}{\sqrt{\text{Var}[x]} + \epsilon}, \quad 
y = \gamma \hat{x} + \beta
$$

### BN 的梯度反向傳播（期中考最愛出的大題）
$$
\frac{\partial L}{\partial \hat{x}_i} = \frac{\partial L}{\partial y_i} \cdot \gamma
$$
$$
\frac{\partial L}{\partial \sigma_B^2} = -\frac{1}{2} \sum_i \frac{\partial L}{\partial \hat{x}_i} \cdot (x_i - \mu_B) \cdot (\sigma_B^2 + \epsilon)^{-3/2}
$$
$$
\frac{\partial L}{\partial \mu_B} = \sum_i \frac{\partial L}{\partial \hat{x}_i} \cdot \frac{-1}{\sqrt{\sigma_B^2 + \epsilon}} + \frac{\partial L}{\partial \sigma_B^2} \cdot \frac{-2}{B} \sum_i (x_i - \mu_B)
$$
$$
\frac{\partial L}{\partial x_i} = \frac{\partial L}{\partial \hat{x}_i} \cdot \frac{1}{\sqrt{\sigma_B^2 + \epsilon}} + \frac{\partial L}{\partial \sigma_B^2} \cdot \frac{2(x_i - \mu_B)}{B} + \frac{\partial L}{\partial \mu_B} \cdot \frac{1}{B}
$$

現在實作位置：**Conv/FC → BN → ReLU**（主流）

---

## 5. CNN、ResNet、DenseNet、Transformer（簡要公式版）

### CNN 卷積輸出大小
$$
O = \left\lfloor \frac{W - K + 2P}{S} \right\rfloor + 1
$$

### ResNet 核心
$$
y = F(x, \{W_i\}) + x \quad \text{(identity shortcut)}
$$
當維度不合時用 1×1 卷積調整 x

### DenseNet 核心
$$
x_l = H_l \big( [x_0, x_1, \dots, x_{l-1}] \big) \quad \text{[ ] 表示 concatenation}
$$

### Transformer - Self-Attention（必推）
$$
\text{Attention}(Q,K,V) = \text{softmax}\left( \frac{QK^T}{\sqrt{d_k}} \right) V
$$
除 $\sqrt{d_k}$：防止點積過大導致 softmax 梯度接近 0

### Positional Encoding
$$
PE_{(pos,2i)} = \sin\left(\frac{pos}{10000^{2i/d}}\right), \quad
PE_{(pos,2i+1)} = \cos\left(\frac{pos}{10000^{2i/d}}\right)
$$
