# 股票當沖策略回測系統

## 系統概述

這個Python程式是一個完整的股票當沖策略回測系統，專為台灣股市設計。它使用機器學習模型分析大量市場特徵，識別具有當日獲利潛力的股票。系統包含數據獲取、特徵工程、模型訓練與評估，以及交易策略回測等完整功能。

## 資料庫撈取

系統從PostgreSQL資料庫（位於140.113.87.91的finDB）中撈取以下資料表的數據：

1. **台股日交易資料 (tw_stock_daily_price)**
   - 欄位：date, stock_id, open, high, low, close, t_volume, t_money
   - 用途：獲取基本的價格和成交量資訊

2. **融資融券資料 (tw_stock_margin_purchase_short_sale)**
   - 欄位：date, stock_id, margin_purchase_buy, margin_purchase_cash_repayment, margin_purchase_limit, margin_purchase_sell, margin_purchase_today_balance, margin_purchase_yesterday_balance, offset_loan_and_short, short_sale_buy, short_sale_cash_repayment, short_sale_limit, short_sale_sell, short_sale_today_balance, short_sale_yesterday_balance
   - 用途：計算融資融券相關指標和比率

3. **三大法人買賣超資料 (tw_stock_institutional_investors_buy_sell)**
   - 欄位：date, stock_id, buy, sell, name (篩選 Foreign_Investor)
   - 用途：追蹤外資買賣行為，計算外資買賣比例和累積買賣

這些資料透過`connect_db()`函數建立的資料庫連接進行撈取，並在程式中轉換為特徵用於訓練模型和進行回測。

## 主要功能

1. **數據獲取**：從Postgres資料庫獲取台灣股市交易數據、融資融券資料和外資交易數據
2. **特徵工程**：生成超過100個交易相關特徵，包括價格形態、成交量變化、融資融券指標等
3. **模型訓練**：支援XGBoost、LightGBM和TabNet三種機器學習模型
4. **策略回測**：評估模型在歷史數據上的表現，生成績效指標和權益曲線
5. **結果視覺化**：生成特徵重要性、訓練曲線和權益曲線等視覺化圖表

## 系統需求

- Python 3.7或更高版本
- PostgreSQL資料庫（用於存儲股票資料）
- 以下Python套件：
  - pandas
  - numpy
  - matplotlib
  - seaborn
  - psycopg2
  - xgboost
  - lightgbm
  - pytorch-tabnet
  - scikit-learn
  - torch

## 安裝步驟

1. 克隆或下載程式碼
2. 安裝必要的Python套件：
   ```bash
   pip install pandas numpy matplotlib seaborn psycopg2 xgboost lightgbm pytorch-tabnet scikit-learn torch
   ```
3. 設定PostgreSQL資料庫連接參數（在`connect_db`函數中）

## 使用方法

### 基本用法

程式的主要入口點是`main`函數，可以通過以下方式運行：

```python
# 使用預設參數運行
if __name__ == "__main__":
    main()
```

或者指定自訂參數：

```python
if __name__ == "__main__":
    main(train_start_date='2020-01-01',
         train_end_date='2022-12-31',
         test_start_date='2023-01-01',
         test_end_date='2024-12-31',
         model_type='xgboost')
```

### 參數說明

- `train_start_date`：訓練數據開始日期（格式：YYYY-MM-DD）
- `train_end_date`：訓練數據結束日期
- `test_start_date`：測試數據開始日期
- `test_end_date`：測試數據結束日期
- `model_type`：使用的模型類型，可選值為 'xgboost'、'lightgbm' 或 'tabnet'

## 系統流程說明

1. **數據準備階段**
   - 建立資料庫連接並獲取股票數據
   - 計算各種技術指標和特徵
   - 生成訓練標籤（次日開盤到收盤的價格變動）
   - 過濾不符合條件的股票（如成交量過低、非4碼股票等）

2. **特徵工程階段**
   - 計算價格相關特徵（開高低收變化率、上下影線等）
   - 計算成交量相關特徵（成交量變化、相對移動平均等）
   - 計算融資融券特徵（融資餘額變動、資券比等）
   - 計算外資買賣特徵（外資買進賣出比例、累積買賣等）
   - 計算歷史特徵（前N天的價格和量能變化）

3. **模型訓練階段**
   - 特徵選擇（選出最具預測力的特徵子集）
   - 特徵標準化
   - 使用選定的模型進行訓練（XGBoost/LightGBM/TabNet）
   - 評估模型表現並分析特徵重要性

4. **回測階段**
   - 對測試數據進行預測
   - 每日選取前30名高機率股票
   - 計算交易盈虧（考慮開盤漲幅門檻和手續費）
   - 生成權益曲線和績效指標

## 輸出結果

程式會在以下目錄生成結果：

- `./models/`：保存訓練好的模型和標準化參數
- `./analysis_results/`：包含回測結果、權益曲線和特徵重要性等
  - `features/`：特徵重要性分析
  - `detailed_trades.csv`：詳細交易記錄
  - `daily_top30_stocks_{train_year}_{test_year}.csv`：每日選股結果
  - `equity_curve{train_year}_{test_year}.png`：權益曲線圖表

## 注意事項

1. 請確保資料庫連接參數正確，以便成功獲取數據
2. 在正式運行前，請確認系統中存在`./output2020_2025/`目錄，其中包含不可當沖股票的清單
3. 如果使用GPU加速模型訓練，請確保已正確安裝CUDA和相關驅動
4. 回測結果僅供參考，實際交易可能受到市場流動性、滑點等因素影響

## 參數調整

若要調整模型和交易策略，可以修改以下參數：

- 特徵工程參數：可在各個特徵計算函數中調整
- 模型參數：可在`train_model`、`train_model_lightgbm`或`train_model_tabnet`函數中調整
- 交易策略參數：可在`calculate_equity_curve`函數中調整開盤漲幅門檻等

## 示例

以下是運行程式的典型流程：

1. 確保資料庫連接可用
2. 運行主程式，設定適當的訓練和測試期間
3. 檢查輸出的特徵重要性分析，了解哪些因素影響當沖獲利
4. 查看權益曲線和績效指標，評估策略效果
5. 調整參數後重新運行，優化策略表現

## 自訂開發

若要擴展系統功能，可以：

1. 添加新的特徵計算函數
2. 整合其他機器學習模型
3. 實現更複雜的交易策略和風險管理邏輯
4. 添加實時交易接口，將回測系統轉為實盤交易系統

此回測系統提供了一個完整的框架，可以根據個人需求進行擴展和優化。