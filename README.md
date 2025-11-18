# 深度學習期中考最強申論版筆記（可直接拿來回答大題 + 公式完美顯示）

## 老師的五大步驟（開頭申論直接背）
1. **資料準備**：資料清理、正規化、增強、train/val/test 劃分、打包成 batch  
2. **模型定義**：設計網路架構、層數、激活函數、參數初始化  
3. **損失函數與優化器設定**：選 Loss + Optimizer + 初始學習率  
4. **訓練超參數設定**：epoch、batch size、lr schedule、weight decay、dropout、early stopping  
5. **評估與推論**：計算指標、畫 loss/acc 曲線、`model.eval()` + `torch.no_grad()`

下面每一節都用「如果申論題問 XXX，我會這樣答」的完整方式寫，保證教授看了會想給高分。

---

## 1. 神經網路基礎

**申論題可能問法：請說明單一神經元的前向計算公式，並解釋為什麼需要非線性激活函數？**

答：

$$
z = \mathbf{w}^\top \mathbf{x} + b,\qquad a = \sigma(z)
$$

其中 $\mathbf{w}$ 為權重向量，$b$ 為偏差，$\sigma(\cdot)$ 為非線性激活函數。  
**為何需要非線性？** 若無非線性激活函數，多層網路無論疊多深，最終都只相當於一層線性變換，無法逼近複雜非線性函數（Universal Approximation Theorem 的前提是有非線性）。

**申論題可能問法：請列出常見激活函數並說明其優缺點**

下面用 HTML 表格以確保公式在 GitHub 的 Markdown 中不會跑版（表格格內可放置區塊或行內數學）：

<table>
  <thead>
    <tr>
      <th>激活函數</th>
      <th>公式</th>
      <th>導數</th>
      <th>優點</th>
      <th>缺點</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Sigmoid</td>
      <td>$$\sigma(z)=\dfrac{1}{1+e^{-z}}$$</td>
      <td>$$\sigma(z)(1-\sigma(z))$$</td>
      <td>輸出 $0\si
