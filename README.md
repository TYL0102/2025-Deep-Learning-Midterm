# 深度學習期中考最強申論版筆記（可直接拿來回答大題 + 公式完美顯示）

## 老師的五大步驟（開頭申論直接背）
1. 資料準備：資料清理、正規化、增強、train/val/test 劃分、打包成 batch  
2. 模型定義：設計網路架構、層數、激活函數、參數初始化  
3. 損失函數與優化器設定：選 Loss + Optimizer + 初始學習率  
4. 訓練超參數設定：epoch、batch size、lr schedule、weight decay、dropout、early stopping  
5. 評估與推論：計算指標、畫 loss/acc 曲線、model.eval() + no_grad()

下面每一節都用「如果申論題問 XXX，我會這樣答」的完整方式寫，保證教授看了會想給高分。

## 1. 神經網路基礎

**申論題可能問法：請說明單一神經元的前向計算公式，並解釋為什麼需要非線性激活函數？**

答：  
單一神經元的計算公式為  
$$z = \mathbf{w}^\top \mathbf{x} + b \quad ,\quad a = \sigma(z)$$  
其中 $\mathbf{w}$ 為權重向量，$b$ 為偏差，$\sigma(\cdot)$ 為非線性激活函數。  
若無非線性激活函數，多層網路無論疊多深，最終都只相當於一層線性變換，無法逼近複雜非線性函數（Universal Approximation Theorem 的前提就是要有非線性）。

**申論題可能問法：請列出常見激活函數並說明其優缺點**

答：  
| 激活函數     | 公式                                                 | 導數                                       | 優點                               | 缺點                                 |
|--------------|------------------------------------------------------|--------------------------------------------|------------------------------------|--------------------------------------|
| Sigmoid      | $$\sigma(z)=\frac{1}{1+e^{-z}}$$                    | $$\sigma(z)(1-\sigma(z))$$                | 輸出 0~1，可解釋為機率            | 兩端飽和 → 梯度消失                  |
| Tanh         | $$\tanh(z)=\frac{e^z-e^{-z}}{e^z+e^{-z}}$$          | $$1-\tanh^2(z)$$                          | 輸出 -1~1，零中心                  | 仍會飽和，梯度消失                   |
| ReLU         | $$f(z)=\max(0,z)$$                                  | $$0(z\le0),\;1(z>0)$$                     | 計算極快、不飽和、稀疏激活         | Dying ReLU（負區死掉）               |
| Leaky ReLU   | $$f(z)=\begin{cases}z & z>0 \\ \alpha z & z\le0\end{cases}$$ | 同上                                      | 解決 Dying ReLU                    | 需要手調 $\alpha$                    |
| Hard Tanh    | $$f(z)=\begin{cases}-1 & z<-1 \\ z & -1\le z\le1 \\ 1 & z>1\end{cases}$$ | 中間區間導數=1                           | 輸出有界，早期 RNN 常用            | 不連續可導                           |
| Softplus     | $$f(z)=\ln(1+e^z)$$                                 | $$\frac{1}{1+e^{-z}}$$                    | 光滑、可導無處不連續               | 計算較慢                             |

**申論題必考：請完整推導反向傳播的四個核心公式**

答：  
定義局部梯度 $\delta^l = \frac{\partial L}{\partial z^l}$  
(1) 輸出層  
$$\delta^L = \frac{\partial L}{\partial a^L} \odot \sigma'(z^L)$$  
(2) 隱藏層遞迴  
$$\delta^l = (W^{l+1}^\top \delta^{l+1}) \odot \sigma'(z^l)$$  
(3) 權重梯度  
$$\frac{\partial L}{\partial W^l} = \delta^l (a^{l-1})^\top$$  
(4) 偏差梯度  
$$\frac{\partial L}{\partial b^l} = \sum_{\text{batch}} \delta^l$$

## 2. 梯度下降變種（申論題最愛叫你比較）

**申論題可能問法：請比較 SGD、Momentum、AdaGrad、RMSProp、Adam 的差異與優缺點**

答：  
- **SGD**：$$w \leftarrow w - \eta \nabla L$$  
  最基礎，容易震盪  
- **Momentum**：加入速度項 $$v_t = \beta v_{t-1} + \nabla L$$  
  像推車有慣性，能加速通過峽谷  
- **AdaGrad**：$$G_t = G_{t-1} + (\nabla L)^2,\;\; w \leftarrow w - \frac{\eta}{\sqrt{G_t+\varepsilon}}\nabla L$$  
  稀疏特徵自動大學習率，但後期學習率幾乎變 0  
- **RMSProp**：用指數移動平均改 AdaGrad  
  $$E[g^2]_t = \beta E[g^2]_{t-1} + (1-\beta)(\nabla L)^2$$  
  解決學習率過早衰減問題  
- **Adam**：Momentum + RMSProp 的結合，加上偏差修正，目前實務與學術首選

## 3. 正則化技術（常考選擇+申論）

L2 正則化（weight decay）→ 梯度多出 $\lambda w$  
Dropout → 訓練時隨機丟神經元，測試時 scaling  
Early Stopping → val loss 不降就停  
Data Augmentation → 增加資料多樣性

## 4. Batch Normalization（這題如果出現就是 20~30 分的大題）

**申論題經典問法：請詳細說明 Batch Normalization 的原理、訓練與測試階段的差異，並推導其反向傳播公式**

答：  

**原理**  
在深層網路中，每一層的輸入分佈會隨著前層參數更新而不斷改變（Internal Covariate Shift），導致訓練困難。BN 強制將每層輸入正規化為均值 0、變異數 1，再經過可學習的 $\gamma$、$\beta$ 縮放與平移，讓網路自己決定要不要正規化。

**訓練階段（mini-batch）**  
對 mini-batch $B=\{x_1,\dots,x_m\}$：  
1. $$\mu_B = \frac{1}{m}\sum_{i=1}^m x_i$$  
2. $$\sigma_B^2 = \frac{1}{m}\sum_{i=1}^m (x_i - \mu_B)^2$$  
3. $$\hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \varepsilon}}$$  
4. $$y_i = \gamma \hat{x}_i + \beta$$

同時維護全域移動平均（預設 momentum=0.1）  
$$\mathbb{E}[x] \leftarrow 0.1 \cdot \mathbb{E}[x] + 0.9 \cdot \mu_B$$

**測試階段**  
直接使用訓練時累積的 $\mathbb{E}[x]$ 與 $\text{Var}[x]$ 進行正規化。

**反向傳播完整推導（教授最愛看）**  
$$\frac{\partial L}{\partial \hat{x}_i} = \frac{\partial L}{\partial y_i} \cdot \gamma$$  
$$\frac{\partial L}{\partial \sigma_B^2} = \sum_i \frac{\partial L}{\partial \hat{x}_i} (x_i-\mu_B) \left(-\frac{1}{2}\right)(\sigma_B^2+\varepsilon)^{-3/2}$$  
$$\frac{\partial L}{\partial \mu_B} = \sum_i \frac{\partial L}{\partial \hat{x}_i} \left(-\frac{1}{\sqrt{\sigma_B^2+\varepsilon}}\right) + \frac{\partial L}{\partial \sigma_B^2} \cdot \frac{-2}{m}\sum_i(x_i-\mu_B)$$  
$$\frac{\partial L}{\partial x_i} = \frac{\partial L}{\partial \hat{x}_i} \cdot \frac{1}{\sqrt{\sigma_B^2+\varepsilon}} + \frac{\partial L}{\partial \sigma_B^2} \cdot \frac{2(x_i-\mu_B)}{m} + \frac{\partial L}{\partial \mu_B} \cdot \frac{1}{m}$$

**現在主流位置**：Conv/FC → BN → ReLU

## 5. ResNet

**申論題：為什麼深層網路很難訓練？ResNet 如何解決？請寫出核心公式**

答：  
網路越深，梯度消失/爆炸越嚴重，導致訓練失敗。  
ResNet 提出 residual learning：  
$$y = \mathcal{F}(x,\{W_i\}) + x$$  
即使 $\mathcal{F}(x)=0$，也至少是恆等映射，確保深層網路不會比淺層差。  
當維度不匹配時，用 1×1 卷積做 projection。

## 6. DenseNet

**申論題：DenseNet 的特點與公式**

答：  
每層都直接連接到後面所有層（dense connectivity）  
$$x_l = H_l([x_0,x_1,\dots,x_{l-1}])$$  
優點：極致特徵重用、參數效率高、緩解梯度消失；缺點：記憶體需求大。

## 7. Transformer - Self-Attention

**申論題：請完整說明 Scaled Dot-Product Attention 的公式與除以 $\sqrt{d_k}$ 的原因**

答：  
$$\text{Attention}(Q,K,V) = \text{softmax}\left( \frac{QK^\top}{\sqrt{d_k}} \right)V$$  
原因：Q、K 點積的期望與變異數皆與 $d_k$ 成正比，不除 $\sqrt{d_k}$ 會讓點積數值過大，softmax 後梯度趨近於 0，無法學習。

**Positional Encoding**  
使用固定 sin/cos 函數讓模型知道序列順序。

筆記結束！

把上面這整段直接複製到任何支援 Markdown 的地方（包括 GitHub README）都會完美顯示。  
現在你就算遇到「請詳細說明...」這種 20~30 分的大題，也可以直接照抄上面的段落，教授看了絕對會傻眼等級的高分。

要我再幫你出 15 題「最可能出現在你學校期中考」的申論題＋參考答案，還是把這份轉成 PDF？隨時說！  
衝吧！！今天穩滿分！！！
