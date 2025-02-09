import psycopg2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
import os
from matplotlib.font_manager import FontProperties
warnings.filterwarnings('ignore')
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from matplotlib import gridspec
import matplotlib.ticker as mtick
import math
# 設定中文字型
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

def create_folders():
    """創建儲存結果的資料夾"""
    folders = [
        './analysis_results',
        './analysis_results/features',
        
    ]
    for folder in folders:
        if not os.path.exists(folder):
            os.makedirs(folder)

def connect_db() -> psycopg2.extensions.connection:
    """建立資料庫連接"""
    conn = psycopg2.connect(
        host='140.113.87.91',
        database='finDB',
        user='nycu_findb_user',
        password='NYCUd@t@b@se8791'
    )
    conn.autocommit = True
    return conn
def get_stock_data(start_date: str, end_date: str) -> pd.DataFrame:
    """從資料庫獲取股票數據"""
    conn = connect_db()
    query = f"""
    SELECT date, stock_id, open, high, low, close, t_volume, t_money
    FROM tw_stock_daily_price
    WHERE date BETWEEN '{start_date}' AND '{end_date}'
    ORDER BY date, stock_id
    """
    df = pd.read_sql(query, conn)
    conn.close()
    return df
def calculate_volume_ma(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """計算成交量移動平均"""
    df['volume_ma'] = df.groupby('stock_id')['t_volume'].rolling(window=window).mean().reset_index(0, drop=True)
    return df

def calculate_volume_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """計算成交量比率（當日成交量/前日成交量）"""
    df['prev_volume'] = df.groupby('stock_id')['t_volume'].shift(1)
    df['volume_ratio'] = df['t_volume'] / df['prev_volume']
    return df

def calculate_upper_shadow(df: pd.DataFrame) -> pd.DataFrame:
    """計算上影線比率"""
    df['upper_shadow_ratio'] = (df['high'] - df['open']) / (df['high'] - df['low'])
    return df

def calculate_price_ma_bias(df: pd.DataFrame) -> pd.DataFrame:
    """計算價格乖離率
    使用5日和10日乖離率
    乖離率 = (目前股價 - MA價格) / MA價格 * 100%
    """
    # 計算5日和10日移動平均
    df['ma5'] = df.groupby('stock_id')['close'].rolling(window=5).mean().reset_index(0, drop=True)
    df['ma10'] = df.groupby('stock_id')['close'].rolling(window=10).mean().reset_index(0, drop=True)
    
    # 計算乖離率
    df['bias_5'] = (df['close'] - df['ma5']) / df['ma5'] * 100
    df['bias_10'] = (df['close'] - df['ma10']) / df['ma10'] * 100
    
    return df

def filter_stocks(df: pd.DataFrame) -> pd.DataFrame:
    """篩選符合條件的股票"""
    # 只保留4碼股票
    df['stock_id'] = df['stock_id'].astype(str)
    stock_code_filter = df['stock_id'].str.len() == 4
    
    # 成交量和金額條件
    trading_filter = (
        (df['volume_ma'] > 1000) #& # # 5日均量大於1000
       # (df['t_money'] > 100000000)  # 成交金額大於1億
    )
    
    # 合併所有條件
    conditions = stock_code_filter & trading_filter
    
    filtered_df = df[conditions].copy()
    
    # 印出過濾資訊
    print(f"原始股票數量: {len(df['stock_id'].unique())}")
    print(f"過濾後股票數量: {len(filtered_df['stock_id'].unique())}")
    print(f"原始資料筆數: {len(df)}")
    print(f"過濾後資料筆數: {len(filtered_df)}")
    
    return filtered_df
def calculate_daily_return(df: pd.DataFrame) -> pd.DataFrame:
    """計算當沖報酬率 (open - close)"""
    df['daily_return'] = (df['open'] - df['close']) / df['open']
    return df

def calculate_margin_features(margin_data: pd.DataFrame) -> pd.DataFrame:
    """
    計算融資融券相關的特徵，並將 NaN 值替換為 0
    
    Parameters:
    -----------
    margin_data : pd.DataFrame
        融資融券資料，從 get_margin_data() 獲取的原始資料
        
    Returns:
    --------
    pd.DataFrame
        包含融資融券特徵的DataFrame
    """
    # 複製資料以避免修改原始資料
    df = margin_data.copy()
    df['date'] = pd.to_datetime(df['date'])
    
    # 1. 基礎融資比率
    # 融資使用率
    df['margin_usage_ratio'] = df['margin_purchase_today_balance'] / df['margin_purchase_limit']
    
    # 融資餘額變動率
    df['margin_balance_change_ratio'] = (df['margin_purchase_today_balance'] - 
                                       df['margin_purchase_yesterday_balance']) / \
                                       df['margin_purchase_yesterday_balance']
    
    # 融資買進強度
    df['margin_buy_strength'] = df['margin_purchase_buy'] / df['margin_purchase_yesterday_balance']
    
    # 融資賣出強度
    df['margin_sell_strength'] = df['margin_purchase_sell'] / df['margin_purchase_yesterday_balance']
    
    # 2. 基礎融券比率
    # 融券使用率
    df['short_usage_ratio'] = df['short_sale_today_balance'] / df['short_sale_limit']
    
    # 融券餘額變動率
    df['short_balance_change_ratio'] = (df['short_sale_today_balance'] - 
                                      df['short_sale_yesterday_balance']) / \
                                      df['short_sale_yesterday_balance']
    
    # 融券賣出強度
    df['short_sell_strength'] = df['short_sale_sell'] / df['short_sale_yesterday_balance']
    
    # 融券買進強度
    df['short_buy_strength'] = df['short_sale_buy'] / df['short_sale_yesterday_balance']
    
    # 3. 綜合指標
    # 資券比
    df['margin_short_ratio'] = df['margin_purchase_today_balance'] / df['short_sale_today_balance']
    
    # 資券維持率差異
    df['margin_short_maintain_spread'] = (df['margin_purchase_today_balance'] / df['margin_purchase_limit']) - \
                                       (df['short_sale_today_balance'] / df['short_sale_limit'])
    
    # 將無限值替換為 NaN
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # 定義要處理的比率欄位
    ratio_columns = [
        'margin_usage_ratio', 'margin_balance_change_ratio', 
        'margin_buy_strength', 'margin_sell_strength',
        'short_usage_ratio', 'short_balance_change_ratio', 
        'short_sell_strength', 'short_buy_strength',
        'margin_short_ratio', 'margin_short_maintain_spread'
    ]
    
    # 4. 計算5日移動平均 (在填充NaN之前)
    for col in ratio_columns:
        df[f'{col}_5d_ma'] = df.groupby('stock_id')[col].rolling(window=5).mean().reset_index(0, drop=True)
    
    # 5. 將所有 NaN 值替換為 0
    columns_to_fill = ratio_columns + [f'{col}_5d_ma' for col in ratio_columns]
    df[columns_to_fill] = df[columns_to_fill].fillna(0)
    
    # 6. 保留原始日期和股票代碼
    result_df = df[['date', 'stock_id'] + columns_to_fill]
    
    return result_df
def get_margin_data(start_date: str, end_date: str) -> pd.DataFrame:
    """從資料庫獲取融資融券數據"""
    conn = connect_db()
    query = f"""
    SELECT 
        date, 
        stock_id,
        margin_purchase_buy,
        margin_purchase_cash_repayment,
        margin_purchase_limit,
        margin_purchase_sell,
        margin_purchase_today_balance,
        margin_purchase_yesterday_balance,
        offset_loan_and_short,
        short_sale_buy,
        short_sale_cash_repayment,
        short_sale_limit,
        short_sale_sell,
        short_sale_today_balance,
        short_sale_yesterday_balance
    FROM tw_stock_margin_purchase_short_sale
    WHERE date BETWEEN '{start_date}' AND '{end_date}'
    ORDER BY date, stock_id
    """
    df = pd.read_sql(query, conn)
    conn.close()
    return df

def calculate_turnover_rate(stock_daily_data: pd.DataFrame, stock_info: pd.DataFrame) -> pd.DataFrame:
    """
    計算股票的日周轉率
    
    Parameters:
    -----------
    stock_daily_data : pd.DataFrame
        日交易資料，需包含 stock_id, date, t_volume 欄位
    stock_info : pd.DataFrame
        股票基本資料，需包含 代號, 發行張數 欄位
        
    Returns:
    --------
    pd.DataFrame
        包含周轉率的DataFrame
    """
    # 重命名股票基本資料的欄位，以便合併
    stock_info = stock_info.rename(columns={
        '代號': 'stock_id',
        '發行張數': 'shares_issued'
    })
    
    # 確保stock_id的型態一致
    stock_daily_data['stock_id'] = stock_daily_data['stock_id'].astype(str)
    stock_info['stock_id'] = stock_info['stock_id'].astype(str)
    
    # 合併資料
    merged_data = pd.merge(
        stock_daily_data,
        stock_info[['stock_id', 'shares_issued']],
        on='stock_id',
        how='left'
    )
    
    # 計算周轉率 (交易量/發行張數)
    merged_data['turnover_rate'] = merged_data['t_volume'] / merged_data['shares_issued']
    
    return merged_data
def get_foreign_trade(start_date: str, end_date: str) -> pd.DataFrame:
    """從資料庫獲取外資交易數據"""
    conn = connect_db()
    query = f"""
    SELECT date, stock_id, buy, sell
    FROM tw_stock_institutional_investors_buy_sell
    WHERE date BETWEEN '{start_date}' AND '{end_date}'
    AND name = 'Foreign_Investor'
    ORDER BY date, stock_id
    """
    df = pd.read_sql(query, conn)
    conn.close()
    return df
def generate_foreign_features(start_date: str, end_date: str) -> pd.DataFrame:
    """生成外資買賣相關特徵"""
    
    # 獲取數據
    stock_df = get_stock_data(start_date, end_date)
    foreign_df = get_foreign_trade(start_date, end_date)
    # 確保日期格式
    stock_df['date'] = pd.to_datetime(stock_df['date'])
    foreign_df['date'] = pd.to_datetime(foreign_df['date'])
    # 合併資料
    merged_df = pd.merge(
        stock_df[['date', 'stock_id', 't_volume']], 
        foreign_df[['date', 'stock_id', 'buy', 'sell']], 
        on=['date', 'stock_id'], 
        how='inner'
    )
    
    # 計算每日外資買賣比例
    merged_df['foreign_buy_ratio'] = merged_df['buy'] / merged_df['t_volume']  # 外資買進比例
    merged_df['foreign_sell_ratio'] = merged_df['sell'] / merged_df['t_volume']  # 外資賣出比例
    merged_df['foreign_net_ratio'] = (merged_df['buy'] - merged_df['sell']) / merged_df['t_volume']  # 外資淨買超比例
    
    # 排序資料以確保時間序列正確
    merged_df = merged_df.sort_values(['stock_id', 'date'])
    
    # 計算5天內的累積數據
    for i in range(5):
        day = i + 1
        # 累積買進量
        merged_df[f'foreign_buy_sum_{day}d'] = merged_df.groupby('stock_id')['buy'].rolling(
            window=day, min_periods=1
        ).sum().reset_index(0, drop=True)
        
        # 累積賣出量
        merged_df[f'foreign_sell_sum_{day}d'] = merged_df.groupby('stock_id')['sell'].rolling(
            window=day, min_periods=1
        ).sum().reset_index(0, drop=True)
        
        # 累積成交量
        merged_df[f't_volume_sum_{day}d'] = merged_df.groupby('stock_id')['t_volume'].rolling(
            window=day, min_periods=1
        ).sum().reset_index(0, drop=True)
        
        # 計算比例
        merged_df[f'foreign_buy_ratio_{day}d'] = merged_df[f'foreign_buy_sum_{day}d'] / merged_df[f't_volume_sum_{day}d']
        merged_df[f'foreign_sell_ratio_{day}d'] = merged_df[f'foreign_sell_sum_{day}d'] / merged_df[f't_volume_sum_{day}d']
        merged_df[f'foreign_net_ratio_{day}d'] = (
            merged_df[f'foreign_buy_sum_{day}d'] - merged_df[f'foreign_sell_sum_{day}d']
        ) / merged_df[f't_volume_sum_{day}d']
    
    # 選擇要保留的特徵列
    feature_columns = [
        'date', 
        'stock_id',
        'foreign_buy_ratio', 
        'foreign_sell_ratio', 
        'foreign_net_ratio'
    ]
    
    # 加入5天累積比例特徵
    for i in range(5):
        day = i + 1
        feature_columns.extend([
            f'foreign_buy_ratio_{day}d',
            f'foreign_sell_ratio_{day}d',
            f'foreign_net_ratio_{day}d'
        ])
    
    # 選擇特徵並填充缺失值
    result_df = merged_df[feature_columns].copy()
    result_df = result_df.fillna(0)
    
    return result_df
def calculate_historical_features(df: pd.DataFrame, window_size: int = 5) -> pd.DataFrame:
    """
    計算個股前N天的開高低收和成交量
    
    Parameters:
    -----------
    df : pd.DataFrame
        需包含 stock_id, date, open, high, low, close, t_volume 欄位的DataFrame
    window_size : int, default=5
        要往前看幾天的資料
        
    Returns:
    --------
    pd.DataFrame
        包含歷史特徵的DataFrame
    """
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    
    # 排序資料
    df = df.sort_values(['stock_id', 'date'])
    
    # 對每個股票分別計算歷史特徵
    for i in range(1, window_size + 1):
        # 計算前N天的價格資訊
        df[f'open_d{i}'] = df.groupby('stock_id')['open'].shift(i)
        df[f'high_d{i}'] = df.groupby('stock_id')['high'].shift(i)
        df[f'low_d{i}'] = df.groupby('stock_id')['low'].shift(i)
        df[f'close_d{i}'] = df.groupby('stock_id')['close'].shift(i)
        df[f'volume_d{i}'] = df.groupby('stock_id')['t_volume'].shift(i)
        
        # 計算與前N天的價格變化比例
        df[f'price_change_d{i}'] = (df['close'] - df[f'close_d{i}']) / df[f'close_d{i}']
        df[f'volume_change_d{i}'] = (df['t_volume'] - df[f'volume_d{i}']) / df[f'volume_d{i}']
    
    return df
def calculate_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    計算價格相關的特徵
    """
    df = df.copy()
    df = df.sort_values(['stock_id', 'date'])
    
    # 計算昨日收盤價
    df['prev_close'] = df.groupby('stock_id')['close'].shift(1)
    
    # 當日OHLC相對於昨日收盤的變化幅度
    df['open_change'] = (df['open'] - df['prev_close']) / df['prev_close']
    df['high_change'] = (df['high'] - df['prev_close']) / df['prev_close']
    df['low_change'] = (df['low'] - df['prev_close']) / df['prev_close']
    df['close_change'] = (df['close'] - df['prev_close']) / df['prev_close']
    
    # 當日價格區間特徵
    df['day_range'] = (df['high'] - df['low']) / df['prev_close']
    df['up_shadow'] = (df['high'] - df[['open', 'close']].max(axis=1)) / df['prev_close']
    df['down_shadow'] = (df[['open', 'close']].min(axis=1) - df['low']) / df['prev_close']
    df['body'] = abs(df['close'] - df['open']) / df['prev_close']
    
    return df

def calculate_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    計算成交量相關的特徵
    """
    df = df.copy()
    
    # 計算前N日平均成交量和金額
    for days in [5, 10, 20]:
        df[f'volume_ma{days}'] = df.groupby('stock_id')['t_volume'].rolling(days).mean().reset_index(0, drop=True)
        df[f'money_ma{days}'] = df.groupby('stock_id')['t_money'].rolling(days).mean().reset_index(0, drop=True)
    
    # 計算昨日成交量和金額
    df['prev_volume'] = df.groupby('stock_id')['t_volume'].shift(1)
    df['prev_money'] = df.groupby('stock_id')['t_money'].shift(1)
    
    # 相對於前一日的變化
    df['volume_change'] = (df['t_volume'] - df['prev_volume']) / df['prev_volume']
    df['money_change'] = (df['t_money'] - df['prev_money']) / df['prev_money']
    
    # 相對於移動平均的變化
    df['volume_vs_5ma'] = df['t_volume'] / df['volume_ma5']
    df['volume_vs_20ma'] = df['t_volume'] / df['volume_ma20']
    
    return df

def calculate_historical_features(df: pd.DataFrame, window_size: int = 5) -> pd.DataFrame:
    """
    計算歷史特徵
    """
    df = df.copy()
    
    # 對每個股票分別計算歷史特徵
    for i in range(1, window_size + 1):
        # 計算前N天的變化率
        df[f'open_change_d{i}'] = df.groupby('stock_id')['open_change'].shift(i)
        df[f'high_change_d{i}'] = df.groupby('stock_id')['high_change'].shift(i)
        df[f'low_change_d{i}'] = df.groupby('stock_id')['low_change'].shift(i)
        df[f'close_change_d{i}'] = df.groupby('stock_id')['close_change'].shift(i)
        df[f'volume_change_d{i}'] = df.groupby('stock_id')['volume_change'].shift(i)
        
        # 計算N天累積漲跌幅
        df[f'cumulative_return_d{i}'] = df.groupby('stock_id')['close_change'].rolling(window=i).sum().reset_index(0, drop=True)
        
        # 計算N天累積成交量變化
        df[f'cumulative_volume_d{i}'] = df.groupby('stock_id')['volume_change'].rolling(window=i).sum().reset_index(0, drop=True)
    
    return df

def get_no_intraday_stocks(date):
    """
    讀取指定日期的不可當沖股票清單
    """
    date_str = date.strftime('%Y%m%d')
    file_path = f'./output/{date_str}_no_intraday.csv'
    try:
        no_intraday_df = pd.read_csv(file_path)
        return set(no_intraday_df['stock_id'].astype(str))
    except FileNotFoundError:
        print(f"Warning: No intraday file not found for date {date_str}")
        return set()

def prepare_training_data(start_date: str, end_date: str) -> pd.DataFrame:
    """
    準備訓練資料
    """
    # 獲取原始資料
    stock_data = get_stock_data(start_date, end_date)
    
    # 確保日期格式一致
    stock_data['date'] = pd.to_datetime(stock_data['date'])
    
    # 計算價格和成交量特徵
    df = calculate_price_features(stock_data)
    df = calculate_volume_features(df)
    
    # 過濾條件：4碼股票、成交量和金額門檻
    df['stock_id'] = df['stock_id'].astype(str)
    stock_filter = df['stock_id'].str.len() == 4
    volume_filter = df['volume_ma5'] > 1000
    money_filter = df['t_money'] > 100000000
    
    df = df[stock_filter & volume_filter & money_filter].copy()
    
    # 計算歷史特徵
    df = calculate_historical_features(df)
    
    # 獲取其他特徵資料並統一日期格式
    margin_data = get_margin_data(start_date, end_date)
    margin_data['date'] = pd.to_datetime(margin_data['date'])
    
    foreign_features = generate_foreign_features(start_date, end_date)
    foreign_features['date'] = pd.to_datetime(foreign_features['date'])
    
    # 計算融資融券特徵
    margin_features = calculate_margin_features(margin_data)
    
    #周轉率
    #turnover_feature = calculate_turnover_rate(df)

    # 確保所有要合併的DataFrame都有相同的日期格式
    print("Date types:")
    print("df date type:", df['date'].dtype)
    print("margin_features date type:", margin_features['date'].dtype)
    print("foreign_features date type:", foreign_features['date'].dtype)
    
    # 合併特徵
    #df = pd.merge(df, turnover_feature, on=['date', 'stock_id'], how='left')
    df = pd.merge(df, margin_features, on=['date', 'stock_id'], how='left')
    df = pd.merge(df, foreign_features, on=['date', 'stock_id'], how='left')
    
    # 計算標籤
    df['next_day_high'] = df.groupby('stock_id')['high'].shift(-1)
    df['next_day_low'] = df.groupby('stock_id')['low'].shift(-1)
    df['next_day_open'] = df.groupby('stock_id')['open'].shift(-1)
    df['next_day_close'] = df.groupby('stock_id')['close'].shift(-1)
    #df['price_range_ratio'] = (df['next_day_high'] - df['next_day_low']) / df['close'] * 100
    df['price_range_ratio'] = (df['next_day_open'] - df['next_day_close']) / df['close'] * 100
    df['label'] = (df['price_range_ratio'] > 0.2).astype(int)
    df['profit'] = (df['next_day_open'] - df['next_day_close']) 
    
    # 移除不需要的欄位
    cols_to_drop = [
        'open', 'high', 'low', 't_volume', 't_money',
        'prev_close', 'prev_volume', 'prev_money',
         'next_day_low', 'next_day_close', 'volume_ma5', 'volume_ma10', 'volume_ma20', 'money_ma5', 'money_ma10', 'money_ma20'
    ]
    df = df.drop(columns=cols_to_drop)
    
    # 填充缺失值
    df = df.fillna(0)
    
    # 資料檢查
    print("\n資料集基本資訊:")
    print(f"資料筆數: {len(df)}")
    print(f"特徵數量: {len(df.columns)}")
    print(f"日期範圍: {df['date'].min()} to {df['date'].max()}")
    print(f"股票數量: {len(df['stock_id'].unique())}")
    print("\n標籤分布:")
    print(df['label'].value_counts(normalize=True))
    
    # 檢查特徵
    print("\n特徵列表:")
    feature_cols = [col for col in df.columns if col not in ['date', 'stock_id', 'label', 'price_range_ratio']]
    for col in feature_cols:
        print(f"- {col}")
    # 確保資料按照日期排序
    df = df.sort_values('date')
    
    # 處理無限值和極端值
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # 對數值型欄位進行極端值處理
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    for col in numeric_columns:
        if col not in ['date', 'stock_id', 'label']:
            # 計算上下界
            Q1 = df[col].quantile(0.01)
            Q3 = df[col].quantile(0.99)
            df[col] = df[col].clip(Q1, Q3)
    
    # 填充缺失值
    df = df.fillna(0)
    
    # 轉換日期列為datetime格式
    df['date'] = pd.to_datetime(df['date'])
    
    # 過濾日期範圍
    mask = (df['date'] >= pd.to_datetime(start_date)) & (df['date'] <= pd.to_datetime(end_date))
    df = df[mask].copy()
    
    # 針對每一天過濾不可當沖的股票
    filtered_dfs = []
    for date in df['date'].unique():
        # 獲取當天的資料
        daily_data = df[df['date'] == date].copy()
        
        # 獲取當天不可當沖的股票清單
        no_intraday_stocks = get_no_intraday_stocks(pd.to_datetime(date))
        
        # 過濾掉不可當沖的股票
        daily_data = daily_data[~daily_data['stock_id'].astype(str).isin(no_intraday_stocks)]
        
        filtered_dfs.append(daily_data)
    
    # 合併所有過濾後的資料
    filtered_df = pd.concat(filtered_dfs, ignore_index=True)
    
    return filtered_df

def train_model(X_train_scaled, X_test_scaled, y_train, y_test,feature_columns):
    """
    訓練XGBoost模型並評估效果
    """
    
    # 設定XGBoost參數
    params = {
        'max_depth': 6,
        'learning_rate': 0.01,
        'n_estimators': 1000,
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'scale_pos_weight': len(y_train[y_train==0]) / len(y_train[y_train==1]),
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 3,
        'gamma': 0.1
    }

    # 訓練模型
    model = xgb.XGBClassifier(**params)
    
    # 使用eval_set進行訓練
    eval_set = [(X_train_scaled, y_train), (X_test_scaled, y_test)]
    model.fit(
        X_train_scaled, 
        y_train,
        eval_set=eval_set,
        verbose=True
    )
        # 計算特徵重要性
    feature_importance = pd.DataFrame({
        'feature': feature_columns,
        'importance': model.feature_importances_
    })
    feature_importance = feature_importance.sort_values('importance', ascending=False)
    
    # 繪製特徵重要性熱圖
    plt.figure(figsize=(12, 8))
    top_20_features = feature_importance.head(20)
    sns.barplot(data=top_20_features, x='importance', y='feature')
    plt.title('Top 20 Most Important Features')
    plt.xlabel('Importance Score')
    plt.ylabel('Features')
    plt.tight_layout()
    plt.savefig('./analysis_results/features/feature_importance.png')
    
    # 打印特徵重要性
    print("\n前20個最重要的特徵:")
    print(feature_importance.head(20))
    
    # 將特徵重要性保存到CSV
    feature_importance.to_csv('./analysis_results/features/feature_importance.csv', index=False)
    
    
    return model
def calculate_equity_curve(predictions, y_true, df, start_idx, end_idx):
    """
    計算權益曲線，包含停損邏輯
    """
    trades = pd.DataFrame({
        'date': df['date'].iloc[start_idx:end_idx].values,
        'stock_id': df['stock_id'].iloc[start_idx:end_idx].values,
        'predicted': predictions,
        'actual': y_true,
        'profit': df['profit'].iloc[start_idx:end_idx].values,
        'next_day_high': df['next_day_high'].iloc[start_idx:end_idx].values,
        'next_day_open': df['next_day_open'].iloc[start_idx:end_idx].values,
        'close': df['close'].iloc[start_idx:end_idx].values,
    })

    # 計算停損條件：High > Close * 1.08
    trades['stop_loss'] = trades['next_day_high'] > trades['close'] * 1.08

    # 初始化交易損益
    trades['trade_profit'] = 0.0

    # 處理停損的情況
    stop_loss_mask = (trades['predicted'] == 1) & trades['stop_loss']
    trades.loc[stop_loss_mask, 'trade_profit'] = trades['next_day_open'] - trades['close'] * 1.08

    # 處理未停損的交易
    normal_trade_mask = (trades['predicted'] == 1) & ~trades['stop_loss']
    trades.loc[normal_trade_mask & (trades['profit'] > 0), 'trade_profit'] = trades['profit'] * 0.998  # 獲利扣手續費
    trades.loc[normal_trade_mask & (trades['profit'] <= 0), 'trade_profit'] = trades['profit'] * 1.002  # 虧損扣手續費

    # 儲存交易記錄
    trades.to_csv('./trades.csv', index=False)

    # 計算每日損益和累積獲利
    daily_profits = trades.groupby('date').agg({
        'trade_profit': 'sum',
        'predicted': 'sum'
    }).reset_index()
    daily_profits['cumulative_profit'] = daily_profits['trade_profit'].cumsum()

    # 交易統計資訊
    total_trades = (trades['predicted'] == 1).sum()
    winning_trades = ((trades['predicted'] == 1) & (trades['trade_profit'] > 0)).sum()
    losing_trades = ((trades['predicted'] == 1) & (trades['trade_profit'] < 0)).sum()
    win_rate = winning_trades / total_trades if total_trades > 0 else 0
    total_profit = trades['trade_profit'].sum()

    print(f"\n交易統計:")
    print(f"總交易次數: {total_trades}")
    print(f"獲利次數: {winning_trades}")
    print(f"虧損次數: {losing_trades}")
    print(f"勝率: {win_rate:.2%}")
    print(f"總獲利: {total_profit:.2f}")

    return daily_profits


def equity_plot(data, Strategy, initial_cash):
    """
    繪製權益曲線和回撤圖，使用修正後的夏普比率計算
    
    Parameters:
    -----------
    data : pd.DataFrame
        包含日期和每日獲利的DataFrame
    Strategy : str
        策略名稱
    initial_cash : float
        初始資金
    """
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import matplotlib.ticker as mtick
    
    # 確保日期索引
    data.index = pd.to_datetime(data.index)
    
    # 計算每日權益
    equity_series = initial_cash + data['Return'].cumsum()
    
    # 修正：計算每日報酬率
    # 使用當日獲利除以前一日權益，以計算實際的日報酬率
    previous_equity = equity_series.shift(1)
    previous_equity.iloc[0] = initial_cash  # 設定第一天的前一日權益為初始資金
    daily_returns = data['Return'] / previous_equity
    
    # 計算年化報酬率
    total_days = len(daily_returns)
    trading_days_per_year = 252
    total_profit = data['Return'].sum()
    final_equity = equity_series.iloc[-1]
    annual_return = (final_equity / initial_cash) ** (trading_days_per_year / total_days) - 1
    
    # 修正：計算夏普比率
    risk_free_rate = 0.02  # 假設無風險利率為2%
    excess_daily_returns = daily_returns - risk_free_rate/trading_days_per_year
    annualized_excess_return = excess_daily_returns.mean() * trading_days_per_year
    annualized_std = daily_returns.std() * np.sqrt(trading_days_per_year)
    sharpe_ratio = annualized_excess_return / annualized_std if annualized_std != 0 else 0
    
    # 修正：計算索提諾比率
    downside_returns = daily_returns[daily_returns < risk_free_rate/trading_days_per_year]
    downside_std = downside_returns.std() * np.sqrt(trading_days_per_year)
    sortino_ratio = annualized_excess_return / downside_std if downside_std != 0 else 0
    
    # 計算最大回撤
    cummax = equity_series.cummax()
    drawdown = (cummax - equity_series) / cummax
    max_drawdown = drawdown.max() * 100
    
    # 找出新高點
    new_highs = equity_series[equity_series == equity_series.cummax()]
    
    # 繪圖
    fig = plt.figure(figsize=(16, 9))
    spec = gridspec.GridSpec(ncols=1, nrows=2, height_ratios=[3, 1])
    
    # 上半部：權益曲線
    ax0 = fig.add_subplot(spec[0])
    ax0.plot(equity_series.index, equity_series.values, label='Equity Curve', color='blue')
    ax0.scatter(new_highs.index, new_highs.values,
               c='#02ff0f', s=50, alpha=1, edgecolor='green', 
               label='New Equity High')
    
    # 設置y軸為金額格式
    def format_amount(x, p):
        if x >= 1e8:
            return f'{x/1e8:.1f}億'
        elif x >= 1e4:
            return f'{x/1e4:.1f}萬'
        else:
            return f'{x:.0f}'
    
    ax0.yaxis.set_major_formatter(plt.FuncFormatter(format_amount))
    ax0.grid(True, alpha=0.3)
    ax0.legend(loc='upper left')
    
    # 添加績效指標
    avg_daily_return = daily_returns.mean()
    performance_text = (
        f'{Strategy} Performance Metrics\n'
        f'總獲利: {total_profit:,.0f}\n'
        f'年化報酬率: {annual_return:.2%}\n'
        f'夏普比率: {sharpe_ratio:.2f}\n'
        f'索提諾比率: {sortino_ratio:.2f}\n'
        f'最大回撤: {max_drawdown:.2f}%\n'
        f'平均日報酬率: {avg_daily_return:.2%}'
    )
    ax0.set_title(performance_text, fontsize=12, pad=20)
    
    # 下半部：回撤圖
    ax1 = fig.add_subplot(spec[1])
    ax1.fill_between(drawdown.index, drawdown * 100, 0, color='red', alpha=0.5, label='Drawdown')
    ax1.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='lower left')
    ax1.set_ylabel('Drawdown %')
    
    plt.tight_layout()
    plt.savefig('./analysis_results/equity_curve.png', bbox_inches='tight', dpi=300)
    plt.close()
    
    # 打印詳細的績效指標
    print("\n策略績效統計:")
    print(f"總獲利: {total_profit:,.0f}")
    print(f"年化報酬率: {annual_return:.2%}")
    print(f"夏普比率: {sharpe_ratio:.2f}")
    print(f"索提諾比率: {sortino_ratio:.2f}")
    print(f"最大回撤: {max_drawdown:.2f}%")
    print(f"平均日報酬率: {avg_daily_return:.4%}")
    
    # 額外的統計信息
    winning_days = len(daily_returns[daily_returns > 0])
    losing_days = len(daily_returns[daily_returns < 0])
    win_rate = winning_days / len(daily_returns)
    
    print("\n其他統計:")
    print(f"獲利天數: {winning_days}")
    print(f"虧損天數: {losing_days}")
    print(f"勝率: {win_rate:.2%}")
    print(f"日報酬率標準差: {daily_returns.std():.4%}")
    
    return {
        'total_profit': total_profit,
        'annual_return': annual_return,
        'sharpe_ratio': sharpe_ratio,
        'sortino_ratio': sortino_ratio,
        'max_drawdown': max_drawdown,
        'avg_daily_return': avg_daily_return,
        'win_rate': win_rate
    }
def main():
    # 儲存結果
    create_folders()
    # 設定分析期間
    start_date = '2023-01-01'
    end_date = '2024-12-31'
    
    # 準備資料
    print("準備訓練資料...")
    df = prepare_training_data(start_date, end_date)
    
    # 資料分割（按時間順序）
    train_size = int(len(df) * 0.6)
    train_data = df.iloc[:train_size]
    test_data = df.iloc[train_size:]
    
    def process_features(X):
        # 處理無限值
        X = X.replace([np.inf, -np.inf], np.nan)
        # 填充缺失值
        X = X.fillna(0)
        return X
    
    # 準備特徵
    exclude_columns = [
        'date', 'stock_id', 'label', 'close',
        'price_range_ratio', 'next_day_high', 'next_day_low',
        'next_day_open', 'next_day_close', 'profit'
    ]
    feature_columns = [col for col in df.columns if col not in exclude_columns]
    
    # 分割特徵和標籤
    X_train = process_features(train_data[feature_columns])
    y_train = train_data['label']
    X_test = process_features(test_data[feature_columns])
    y_test = test_data['label']
    
    # 檢查特徵是否含有無限值或極端值
    print("檢查特徵:")
    print("Training set null values:", X_train.isnull().sum().sum())
    print("Testing set null values:", X_test.isnull().sum().sum())
    print("Training set infinite values:", np.isinf(X_train.values).sum())
    print("Testing set infinite values:", np.isinf(X_test.values).sum())
    
    # 特徵標準化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 訓練模型
    model = train_model(X_train_scaled, X_test_scaled, y_train, y_test, feature_columns)
    # 儲存訓練和測試資料
    train_data.to_csv('./analysis_results/train_data.csv', index=False)
    test_data.to_csv('./analysis_results/test_data.csv', index=False)
    # 獲取訓練集預測結果
    train_pred = model.predict(X_train_scaled)
    train_proba = model.predict_proba(X_train_scaled)[:, 1]
    train_equity = calculate_equity_curve(train_pred, y_train, train_data, 0, len(train_data))
    # 儲存訓練集預測結果
    pd.DataFrame({
        'date': train_data['date'],
        'stock_id': train_data['stock_id'],
        'predicted': train_pred,
        'probability': train_proba,
        'actual': y_train,
        'profit': train_data['profit']
    }).to_csv('./analysis_results/train_predictions.csv', index=False)
    
    # 獲取測試集預測結果和機率
    test_pred = model.predict(X_test_scaled)
    test_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    # 儲存測試集預測結果
    pd.DataFrame({
        'date': test_data['date'],
        'stock_id': test_data['stock_id'],
        'predicted': test_pred,
        'probability': test_proba,
        'actual': y_test,
        'profit': test_data['profit']
    }).to_csv('./analysis_results/test_predictions.csv', index=False)
    # 處理測試集 - 選擇每日前10檔股票
    filtered_test_data = pd.DataFrame()
    dates = test_data['date'].unique()
    
    for date in dates:
        # 獲取當日資料
        daily_data = test_data[test_data['date'] == date].copy()
        
        # 獲取當日不可當沖的股票
        no_intraday_file = f'./output1231/{date.strftime("%Y%m%d")}_no_intraday.csv'
        try:
            no_intraday_df = pd.read_csv(no_intraday_file)
            no_intraday_stocks = set(no_intraday_df['stock_id'].astype(str))
            # 過濾掉不可當沖的股票
            daily_data = daily_data[~daily_data['stock_id'].astype(str).isin(no_intraday_stocks)]
        except FileNotFoundError:
            print(f"Warning: No intraday file not found for date {date}")
        
        # 獲取當日的預測機率
        daily_X = process_features(daily_data[feature_columns])
        daily_X_scaled = scaler.transform(daily_X)
        daily_proba = model.predict_proba(daily_X_scaled)[:, 1]  # 獲取正類別的機率
        
        daily_data['pred_probability'] = daily_proba
        
        # 選擇前10檔股票
        top_10_stocks = daily_data.nlargest(30, 'pred_probability')
        filtered_test_data = pd.concat([filtered_test_data, top_10_stocks])
    
    # 重新準備測試集特徵
    X_test_filtered = process_features(filtered_test_data[feature_columns])
    X_test_filtered_scaled = scaler.transform(X_test_filtered)
    # 儲存模型
    print("儲存模型...")
    if not os.path.exists('./models'):
        os.makedirs('./models')
    
    # 儲存 XGBoost 模型
    model.save_model('./models/trained_model.json')
    
    # 儲存 StandardScaler 參數
    scaler_params = {
        'mean': scaler.mean_,
        'scale': scaler.scale_,
        'var': scaler.var_,
        'n_samples_seen': scaler.n_samples_seen_
    }
    np.savez('./models/scaler_params.npz', **scaler_params)
    
    print("模型和標準化參數已儲存至 ./models 目錄")
    # 獲取過濾後的測試集預測結果
    test_pred = model.predict(X_test_filtered_scaled)
    
   # 計算權益曲線
    test_equity = calculate_equity_curve(
        test_pred, 
        filtered_test_data['label'], 
        filtered_test_data, 
        0, 
        len(filtered_test_data)
    )
    
    equity_data = pd.DataFrame({
    'Return': test_equity['trade_profit'].values
    }, index=pd.to_datetime(test_equity['date']))

    performance_metrics = equity_plot(
        data=equity_data * 1000,
        Strategy='當沖策略',
        initial_cash=10000000
    )
    # 打印績效指標
    print("\n策略績效指標：")
    print(f"年化報酬率: {performance_metrics['annual_return']:.2%}")
    print(f"夏普比率: {performance_metrics['sharpe_ratio']:.2f}")
    print(f"索提諾比率: {performance_metrics['sortino_ratio']:.2f}")
    print(f"最大回撤: {performance_metrics['max_drawdown']:.2f}%")
        # 儲存每日選股結果
    filtered_test_data.to_csv('./analysis_results/daily_top10_stocks.csv', index=False)
    
    # 顯示每日選股數量統計
    daily_counts = filtered_test_data.groupby('date')['stock_id'].count()
    print("\n每日選股統計：")
    print(f"平均選股數量: {daily_counts.mean():.2f}")
    print(f"最少選股數量: {daily_counts.min()}")
    print(f"最多選股數量: {daily_counts.max()}")
    
    return model, train_equity, test_equity

if __name__ == "__main__":
    main()
