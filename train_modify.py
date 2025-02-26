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
from pytorch_tabnet.tab_model import TabNetClassifier
import lightgbm as lgb
import torch
import numpy as np
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
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
    file_path = f'./output2020_2025/{date_str}_no_intraday.csv'
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
    df['label'] = (df['price_range_ratio'] >= 0.2).astype(int)
    df['profit'] = (df['next_day_open'] - df['next_day_close']) 
    df['profit_volume'] = (df['next_day_open'] * 0.0015 + (df['next_day_open']  + df['next_day_close']) * 0.001425 *0.3) 

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
    # 添加标签分布检查
    print("\n标签分布检查:")
    label_dist = df['label'].value_counts(normalize=True)
    print(label_dist)
    
    # 添加特征相关性检查
    correlation_matrix = df[feature_cols].corr()
    highly_correlated = np.where(np.abs(correlation_matrix) > 0.85)
    highly_correlated = [(feature_cols[x], feature_cols[y], correlation_matrix.iloc[x, y]) 
                        for x, y in zip(*highly_correlated) if x != y]
    
    if highly_correlated:
        print("\n高相关性特征对:")
        for feat1, feat2, corr in highly_correlated:
            print(f"{feat1} - {feat2}: {corr:.3f}")
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



import xgboost as xgb
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# 3. 添加特征选择步骤
def select_features(X_train, y_train, feature_columns):
    from sklearn.feature_selection import SelectFromModel
    
    # 使用轻量级模型进行特征选择
    selector = SelectFromModel(
        estimator=xgb.XGBClassifier(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=4
        ),
        threshold='median'
    )
    
    selector.fit(X_train, y_train)
    selected_features_mask = selector.get_support()
    selected_features = [feature_columns[i] for i in range(len(feature_columns)) 
                        if selected_features_mask[i]]
    
    print(f"\n选择的特征数量: {len(selected_features)}")
    print("选择的特征:", selected_features)
    
    return selected_features

def train_model(X_train_scaled, X_test_scaled, y_train, y_test, feature_columns):
    """
    訓練 XGBoost 模型並評估效果，包含 GPU 支援和錯誤處理
    """
    import xgboost as xgb
    import numpy as np
    from sklearn.metrics import accuracy_score
    
    print(f"Using XGBoost version: {xgb.__version__}")
    
    # 檢查輸入數據
    print(f"Training data shape: {X_train_scaled.shape}")
    print(f"Number of positive samples: {np.sum(y_train == 1)}")
    print(f"Number of negative samples: {np.sum(y_train == 0)}")
    
    # 檢查是否有 NaN 或無限值
    print("Checking for NaN/Inf in training data...")
    print(f"NaN in X_train: {np.isnan(X_train_scaled).any()}")
    print(f"Inf in X_train: {np.isinf(X_train_scaled).any()}")
    
    # 記錄訓練過程中的指標
    train_metrics = {'loss': [], 'accuracy': []}
    test_metrics = {'loss': [], 'accuracy': []}
    
    # 創建自定義的 callback 類
    class MetricCallback(xgb.callback.TrainingCallback):
        def after_iteration(self, model, epoch, evals_log):
            """每次迭代後計算和記錄指標"""
            # 獲取當前迭代的預測結果
            dtrain_pred = model.predict(dtrain)
            dtest_pred = model.predict(dtest)
            
            # 計算accuracy
            train_acc = accuracy_score(y_train, (dtrain_pred >= 0.5).astype(int))
            test_acc = accuracy_score(y_test, (dtest_pred >= 0.5).astype(int))
            
            # 記錄指標
            if evals_log:
                train_metrics['loss'].append(evals_log['train']['logloss'][-1])
                test_metrics['loss'].append(evals_log['eval']['logloss'][-1])
                train_metrics['accuracy'].append(train_acc)
                test_metrics['accuracy'].append(test_acc)
            
            return False
    neg_pos_ratio = np.sum(y_train == 0) / np.sum(y_train == 1)
    
    params = {
        'max_depth': 6,  # 降低树的深度
        'learning_rate': 0.01,  # 降低学习率
        'n_estimators': 2000,
        'objective': 'binary:logistic',
        'scale_pos_weight': neg_pos_ratio,  # 根据实际比例设置
        'subsample': 0.7,  # 降低采样比例
        'colsample_bytree': 0.7,  # 降低特征采样比例
        'min_child_weight': 5,  # 增加最小子节点权重
        'gamma': 0.3,  # 增加分裂阈值
        'reg_lambda': 3,  # 增加L2正则化
        'reg_alpha': 2,  # 增加L1正则化
        'eval_metric': ['logloss', 'auc']  # 添加AUC评估
    }
    # # 設定基本參數
    # params = {
    #     'max_depth': 20,
    #     'learning_rate': 0.005,  # 調整學習率
    #     'n_estimators': 3000,
    #     'objective': 'binary:logistic',
    #     'scale_pos_weight': len(y_train[y_train == 0]) / len(y_train[y_train == 1]),
    #     'subsample': 0.85,
    #     'colsample_bytree': 0.85,
    #     'min_child_weight': 2,
    #     'gamma': 0.2,
    #     'reg_lambda': 2,
    #     'reg_alpha': 1,
    #     'eval_metric': 'logloss'
    # }
    
    # 嘗試使用 GPU
    try:
        test_matrix = xgb.DMatrix(X_train_scaled[:10], y_train[:10])
        test_model = xgb.train({'tree_method': 'gpu_hist'}, test_matrix, num_boost_round=1)
        params['tree_method'] = 'gpu_hist'
        print("Successfully initialized GPU training")
    except Exception as e:
        print(f"GPU initialization failed: {e}")
        print("Falling back to CPU training")
        params['tree_method'] = 'hist'
    
    # 創建 DMatrix
    try:
        dtrain = xgb.DMatrix(X_train_scaled, label=y_train, feature_names=feature_columns)
        dtest = xgb.DMatrix(X_test_scaled, label=y_test, feature_names=feature_columns)
    except Exception as e:
        print(f"Error creating DMatrix: {e}")
        return None
    
    # 設定評估集
    eval_set = [(dtrain, "train"), (dtest, "eval")]
    
    # 訓練模型
    try:
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=2000,
            evals=eval_set,
            early_stopping_rounds=150,
            verbose_eval=50,  # 每50輪打印一次
            callbacks=[MetricCallback()]
        )
        
        print("Training completed successfully")
        
        # 檢查模型是否成功訓練
        print("Model attributes:")
        print(f"Best iteration: {model.best_iteration}")
        print(f"Best score: {model.best_score}")
        
        # 繪製訓練指標圖
        plt.figure(figsize=(15, 5))
        
        # 損失曲線
        plt.subplot(1, 2, 1)
        plt.plot(train_metrics['loss'], label='Training Loss', color='blue')
        plt.plot(test_metrics['loss'], label='Validation Loss', color='red')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        
        # 準確率曲線
        plt.subplot(1, 2, 2)
        plt.plot(train_metrics['accuracy'], label='Training Accuracy', color='blue')
        plt.plot(test_metrics['accuracy'], label='Validation Accuracy', color='red')
        plt.xlabel('Iteration')
        plt.ylabel('Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('./analysis_results/training_metrics.png')
        plt.close()
        
        # 計算最終的accuracy
        final_train_pred = (model.predict(dtrain) >= 0.5).astype(int)
        final_test_pred = (model.predict(dtest) >= 0.5).astype(int)
        
        print("\n最終準確率:")
        print(f"Training Accuracy: {accuracy_score(y_train, final_train_pred):.4f}")
        print(f"Validation Accuracy: {accuracy_score(y_test, final_test_pred):.4f}")
        
        # 分析特徵重要性
        feature_importance = analyze_feature_importance(model, feature_columns)
        
        # 保存模型
        model.save_model('./models/best_xgboost_model.json')
        
        return model
        
    except Exception as e:
        print(f"Error during training: {e}")
        return None
def train_model_tabnet(X_train_scaled, X_test_scaled, y_train, y_test, feature_columns):
    """
    使用 TabNet 訓練模型並評估效果
    """
    print("Training with TabNet...")
    # 檢查 GPU 可用性
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    # 檢查輸入數據
    print(f"Training data shape: {X_train_scaled.shape}")
    print(f"Number of positive samples: {np.sum(y_train == 1)}")
    print(f"Number of negative samples: {np.sum(y_train == 0)}")
    
    # 記錄訓練過程中的指標
    train_metrics = {'loss': [], 'accuracy': []}
    test_metrics = {'loss': [], 'accuracy': []}
    
    # 設定 TabNet 參數
    tabnet_params = {
        'n_d': 32,  # 決策層維度
        'n_a': 32,  # 注意力層維度
        'n_steps': 3,  # 決策步驟數
        'gamma': 1.5,  # 特徵選擇的係數
        'n_independent': 2,  # 獨立層數量
        'n_shared': 2,  # 共享層數量
        'cat_idxs': [],  # 類別特徵的索引
        'cat_dims': [],  # 類別特徵的維度
        'optimizer_fn': torch.optim.Adam,
        'optimizer_params': dict(lr=2e-2),
        'scheduler_params': dict(step_size=50, gamma=0.9),
        'scheduler_fn': torch.optim.lr_scheduler.StepLR,
        'mask_type': 'entmax',
        'device_name': device  # 指定設備
    }
    
    # 初始化模型
    model = TabNetClassifier(**tabnet_params)
    
    # 轉換數據格式
    X_train_scaled = X_train_scaled.astype(np.float32)
    X_test_scaled = X_test_scaled.astype(np.float32)
    
    # 訓練模型
    model.fit(
        X_train=X_train_scaled, 
        y_train=y_train,
        eval_set=[(X_train_scaled, y_train), (X_test_scaled, y_test)],
        eval_name=['train', 'valid'],
        max_epochs=200,
        patience=50,
        batch_size=1024,
        virtual_batch_size=128,
        num_workers=0,
        drop_last=False
    )
    
    # 獲取訓練歷史
    history = model.history
    
    # 繪製訓練指標圖
    plt.figure(figsize=(15, 5))
    
    # 損失曲線
    plt.subplot(1, 2, 1)
    plt.plot(history['loss'], label='Training Loss', color='blue')
    plt.plot(history['val_loss'], label='Validation Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    
    # 準確率曲線
    plt.subplot(1, 2, 2)
    plt.plot(history['accuracy'], label='Training Accuracy', color='blue')
    plt.plot(history['val_accuracy'], label='Validation Accuracy', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('./analysis_results/tabnet_training_metrics.png')
    plt.close()
    
    # 計算最終的 accuracy
    train_pred = model.predict(X_train_scaled)
    test_pred = model.predict(X_test_scaled)
    
    print("\n最終準確率:")
    print(f"Training Accuracy: {accuracy_score(y_train, train_pred):.4f}")
    print(f"Validation Accuracy: {accuracy_score(y_test, test_pred):.4f}")
    
    # 特徵重要性
    feature_importance = pd.DataFrame({
        'feature': feature_columns,
        'importance': model.feature_importances_
    })
    
    feature_importance = feature_importance.sort_values('importance', ascending=False)
    
    # 保存模型
    torch.save(model.state_dict(), './models/best_tabnet_model.pth')
    
    return model

def train_model_lightgbm(X_train_scaled, X_test_scaled, y_train, y_test, feature_columns):
    """
    使用 LightGBM 訓練模型並評估效果 (GPU 支援)
    """
    print("Training with LightGBM...")
    
    # 記錄訓練過程中的指標
    train_metrics = {'loss': [], 'accuracy': []}
    test_metrics = {'loss': [], 'accuracy': []}
    
    try:
        # 創建數據集
        train_data = lgb.Dataset(X_train_scaled, label=y_train, feature_name=feature_columns)
        test_data = lgb.Dataset(X_test_scaled, label=y_test, feature_name=feature_columns)
        
        # 設定參數（包含 GPU 支援）
        params = {
            'objective': 'binary',
            'metric': ['binary_logloss', 'binary_error'],
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'early_stopping_round': 50,
            'verbose': 0
        }
        
        # 嘗試啟用 GPU
        try:
            test_data_small = lgb.Dataset(X_train_scaled[:100], label=y_train[:100])
            test_params = params.copy()
            test_params['device'] = 'gpu'
            model = lgb.train(test_params, test_data_small, num_boost_round=1)
            
            print("Successfully initialized GPU training for LightGBM")
            params['device'] = 'gpu'
            params['gpu_platform_id'] = 0
            params['gpu_device_id'] = 0
            
        except Exception as e:
            print(f"GPU initialization failed for LightGBM: {e}")
            print("Falling back to CPU training")
            params['device'] = 'cpu'

        class MetricCallback:
            def __init__(self, train_metrics, test_metrics):
                self.train_metrics = train_metrics
                self.test_metrics = test_metrics

            def __call__(self, env):
                try:
                    # 獲取train/valid的loss
                    train_loss = env.evaluation_result_list[0][2]  # training binary_logloss
                    train_error = env.evaluation_result_list[1][2]  # training binary_error
                    valid_loss = env.evaluation_result_list[2][2]  # valid binary_logloss
                    valid_error = env.evaluation_result_list[3][2]  # valid binary_error
                    
                    # 記錄指標
                    self.train_metrics['loss'].append(train_loss)
                    self.train_metrics['accuracy'].append(1 - train_error)
                    self.test_metrics['loss'].append(valid_loss)
                    self.test_metrics['accuracy'].append(1 - valid_error)
                    
                except Exception as e:
                    print(f"Error in callback: {e}")
                return False

        # 訓練模型
        model = lgb.train(
            params,
            train_data,
            num_boost_round=2000,
            valid_sets=[train_data, test_data],
            valid_names=['training', 'valid_1'],
            callbacks=[MetricCallback(train_metrics, test_metrics)]
        )
        
        # 繪製訓練指標圖
        plt.figure(figsize=(15, 5))
        
        # 損失曲線
        plt.subplot(1, 2, 1)
        plt.plot(train_metrics['loss'], label='Training Loss', color='blue')
        plt.plot(test_metrics['loss'], label='Validation Loss', color='red')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        
        # 準確率曲線
        plt.subplot(1, 2, 2)
        plt.plot(train_metrics['accuracy'], label='Training Accuracy', color='blue')
        plt.plot(test_metrics['accuracy'], label='Validation Accuracy', color='red')
        plt.xlabel('Iteration')
        plt.ylabel('Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('./analysis_results/lightgbm_training_metrics.png')
        plt.close()
        
        # 計算最終的 accuracy
        train_pred = (model.predict(X_train_scaled) >= 0.5).astype(int)
        test_pred = (model.predict(X_test_scaled) >= 0.5).astype(int)
        
        print("\n最終準確率:")
        print(f"Training Accuracy: {accuracy_score(y_train, train_pred):.4f}")
        print(f"Validation Accuracy: {accuracy_score(y_test, test_pred):.4f}")
        
        # 特徵重要性
        feature_importance = pd.DataFrame({
            'feature': feature_columns,
            'importance': model.feature_importance()
        })
        feature_importance = feature_importance.sort_values('importance', ascending=False)
        
        # 保存模型
        model.save_model('./models/best_lightgbm_model.txt')
        
        return model
        
    except Exception as e:
        print(f"Error during LightGBM training: {e}")
        import traceback
        traceback.print_exc()
        return None
 #修改 main 函數中的模型選擇部分
def train_selected_model(model_type, X_train_scaled, X_test_scaled, y_train, y_test, feature_columns):
    """
    根據選擇的模型類型訓練相應的模型
    """
    if model_type == 'tabnet':
        return train_model_tabnet(X_train_scaled, X_test_scaled, y_train, y_test, feature_columns)
    elif model_type == 'lightgbm':
        return train_model_lightgbm(X_train_scaled, X_test_scaled, y_train, y_test, feature_columns)
    else:  # 默認使用原來的 XGBoost
        return train_model(X_train_scaled, X_test_scaled, y_train, y_test, feature_columns)
def analyze_feature_importance(model, feature_columns):
    """
    分析並呈現特徵重要性
    
    Parameters:
    -----------
    model : xgb.Booster
        訓練好的XGBoost模型
    feature_columns : list
        特徵名稱列表
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    
    # 獲取特徵重要性
    importance_dict = model.get_score(importance_type='weight')
    
    # 創建特徵重要性DataFrame
    feature_importance = pd.DataFrame({
        'feature': feature_columns,
        'importance': [importance_dict.get(f, 0) for f in feature_columns]
    })
    
    # 正規化重要性分數
    feature_importance['importance_normalized'] = feature_importance['importance'] / feature_importance['importance'].sum() * 100
    
    # 按重要性排序
    feature_importance = feature_importance.sort_values('importance_normalized', ascending=False)
    
    # 繪製前30個重要特徵的柱狀圖
    plt.figure(figsize=(15, 10))
    plt.subplot(1, 1, 1)
    
    # 使用seaborn繪製柱狀圖
    sns.barplot(
        data=feature_importance.head(30),
        x='importance_normalized',
        y='feature',
        palette='viridis'
    )
    
    plt.title('Top 30 Most Important Features (Normalized %)', pad=20)
    plt.xlabel('Importance Score (%)')
    plt.ylabel('Features')
    
    # 在每個柱子後面添加數值標籤
    for i, v in enumerate(feature_importance.head(30)['importance_normalized']):
        plt.text(v, i, f'{v:.2f}%', va='center')
    
    plt.tight_layout()
    plt.savefig('./analysis_results/features/feature_importance_detailed.png', 
                bbox_inches='tight', dpi=300)
    
    # 將完整的特徵重要性保存到CSV
    feature_importance.to_csv('./analysis_results/features/feature_importance_detailed.csv', 
                            index=False)
    
    # 打印前30個最重要的特徵及其重要性分數
    print("\n前30個最重要的特徵及其重要性分數：")
    print("=====================================")
    for idx, row in feature_importance.head(30).iterrows():
        print(f"{row['feature']}: {row['importance_normalized']:.2f}%")
    
    # 按特徵類型分組分析
    feature_types = {
        '價格相關': ['open_change', 'high_change', 'low_change', 'close_change', 
                  'day_range', 'up_shadow', 'down_shadow', 'body'],
        '成交量相關': ['volume_change', 'money_change', 'volume_vs_5ma', 'volume_vs_20ma'],
        '歷史資料': [f for f in feature_columns if '_d' in f],
        '融資融券': [f for f in feature_columns if any(x in f for x in ['margin', 'short'])],
        '外資': [f for f in feature_columns if 'foreign' in f]
    }
    
    print("\n特徵類型重要性分析：")
    print("===================")
    for feature_type, features in feature_types.items():
        type_importance = feature_importance[
            feature_importance['feature'].isin(features)
        ]['importance_normalized'].sum()
        print(f"{feature_type}: {type_importance:.2f}%")
    
    return feature_importance

def calculate_equity_curve(predictions, y_true, df, start_idx, end_idx, open_threshold=0.03):
    """
    計算權益曲線，包含開盤漲幅條件和停損邏輯
    
    Parameters:
    -----------
    predictions : array-like
        模型預測結果
    y_true : array-like
        實際標籤
    df : pd.DataFrame
        包含交易資料的DataFrame
    start_idx : int
        起始索引
    end_idx : int
        結束索引
    open_threshold : float, default=0.03
        開盤漲幅門檻，預設為3%
    stop_loss_threshold : float, default=0.08
        停損門檻，預設為8%
    """
    trades = pd.DataFrame({
        'date': df['date'].iloc[start_idx:end_idx].values,
        'stock_id': df['stock_id'].iloc[start_idx:end_idx].values,
        'predicted': predictions,
        'actual': y_true,
        'next_day_high': df['next_day_high'].iloc[start_idx:end_idx].values,
        'next_day_open': df['next_day_open'].iloc[start_idx:end_idx].values,
        'profit': df['profit'].iloc[start_idx:end_idx].values,
        'close': df['close'].iloc[start_idx:end_idx].values,
        'profit_volume': df['profit_volume'].iloc[start_idx:end_idx].values
    })

    # 初始化交易損益
    trades['trade_profit'] = 0.0

    # 計算隔日開盤漲幅
    trades['open_rise'] = (trades['next_day_open'] - trades['close']) / trades['close']
    
    # 只在開盤漲幅超過門檻時進行交易
    trades['valid_trade'] = (trades['predicted'] == 1) & (trades['open_rise'] >= open_threshold)

    # 計算停損條件：High > Close * 1.08
    trades['stop_loss'] = trades['next_day_high'] > trades['close'] * 1.08

    # 處理停損的情況
    stop_loss_mask = (trades['valid_trade']) & trades['stop_loss']
    trades.loc[stop_loss_mask, 'trade_profit'] = (trades['next_day_open'] - trades['close'] * 1.08) - (df['next_day_open'] * 0.0015 + (df['next_day_open']  + df['close'] * 1.08) * 0.001425 *0.3) 

    # 處理未停損的交易
    normal_trade_mask = (trades['valid_trade']) & ~trades['stop_loss']
    trades.loc[normal_trade_mask, 'trade_profit'] = (trades['profit']) - trades['profit_volume']

    # 統計交易結果
    total_trades = trades['valid_trade'].sum()
    profitable_trades = ((trades['valid_trade']) & (trades['trade_profit'] > 0)).sum()
    losing_trades = ((trades['valid_trade']) & (trades['trade_profit'] < 0)).sum()
    win_rate = profitable_trades / total_trades if total_trades > 0 else 0
    total_profit = trades['trade_profit'].sum()

    print(f"\n交易統計 (開盤漲幅門檻: {open_threshold:.2%}):")
    print(f"總交易次數: {total_trades}")
    print(f"有效交易次數: {trades['valid_trade'].sum()}")
    print(f"獲利次數: {profitable_trades}")
    print(f"虧損次數: {losing_trades}")
    print(f"勝率: {win_rate:.2%}")
    print(f"總獲利: {total_profit:.2f}")

    # 計算每日損益和累積獲利
    daily_profits = trades.groupby('date').agg({
        'trade_profit': 'sum',
        'predicted': 'sum',
        'valid_trade': 'sum'
    }).reset_index()
    
    daily_profits['cumulative_profit'] = daily_profits['trade_profit'].cumsum()

    # 儲存詳細的交易記錄
    trades.to_csv('./analysis_results/detailed_trades.csv', index=False)
    
    return daily_profits

def equity_plot(data, Strategy, initial_cash, train_start_date, test_start_date):
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
    ax1.fill_between(drawdown.index, -drawdown * 100, 0, color='red', alpha=0.5, label='Drawdown')
    ax1.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='lower left')
    ax1.set_ylabel('Drawdown %')
    
    plt.tight_layout()
    plt.savefig(f'./analysis_results/equity_curve{train_start_date[:4]}_{test_start_date[:4]}.png', bbox_inches='tight', dpi=300)
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
import os
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import StandardScaler

def main(train_start_date='2024-01-01', 
         train_end_date='2024-12-31',
         test_start_date='2023-01-01',
         test_end_date='2023-12-31',
         model_type='xgboost'):
    """
    主函數，可自定義訓練和測試期間
    """
    # 建立資料夾
    create_folders()
    
    # 讀取訓練和測試資料
    print("準備訓練資料...")
    train_df = prepare_training_data(train_start_date, train_end_date)
    
    print("準備測試資料...")
    test_df = prepare_training_data(test_start_date, test_end_date)

    def process_features(X):
        """處理無限值、NaN"""
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(0)
        return X

    # 篩選特徵
    exclude_columns = [
        'date', 'stock_id', 'label', 'close',
        'price_range_ratio', 'next_day_high', 'next_day_low',
        'next_day_open', 'next_day_close', 'profit', 'profit_volume'
    ]
    feature_columns = [col for col in train_df.columns if col not in exclude_columns]

    # 分割特徵與標籤
    X_train = process_features(train_df[feature_columns])
    y_train = train_df['label']
    X_test = process_features(test_df[feature_columns])
    y_test = test_df['label']
    # 在这里添加特征选择步骤
    print("执行特征选择...")
    selected_features = select_features(X_train, y_train, feature_columns)
    
    # 使用选择后的特征
    X_train = X_train[selected_features]
    X_test = X_test[selected_features]
    feature_columns = selected_features

    # 標準化特徵
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 訓練模型
    model = train_selected_model(model_type, X_train_scaled, X_test_scaled, y_train, y_test, feature_columns)

    # 儲存訓練和測試資料
    train_df.to_csv('./analysis_results/train_data.csv', index=False)
    test_df.to_csv('./analysis_results/test_data.csv', index=False)

    # 訓練集預測
    print("進行訓練集預測...")
    dtrain_pred = xgb.DMatrix(X_train_scaled, feature_names=feature_columns)  # 添加特徵名稱
    train_pred = model.predict(dtrain_pred)
    train_proba = model.predict(dtrain_pred)

    train_equity = calculate_equity_curve(
        train_pred, y_train, train_df, 0, len(train_df), open_threshold=0.03
    )

    # 儲存訓練集預測結果
    pd.DataFrame({
        'date': train_df['date'],
        'stock_id': train_df['stock_id'],
        'predicted': train_pred,
        'probability': train_proba,
        'actual': y_train,
        'profit': train_df['profit']
    }).to_csv('./analysis_results/train_predictions.csv', index=False)

        # 訓練集預測部分
    print("進行訓練集預測...")
    dtrain_pred = xgb.DMatrix(X_train_scaled, feature_names=feature_columns)
    train_proba = model.predict(dtrain_pred)  # 獲取機率值
    train_pred = (train_proba >= 0.5).astype(int)  # 轉換為0/1標籤

    train_equity = calculate_equity_curve(
        train_pred, y_train, train_df, 0, len(train_df), open_threshold= 0.00
    )

    # 儲存訓練集預測結果
    pd.DataFrame({
        'date': train_df['date'],
        'stock_id': train_df['stock_id'],
        'predicted': train_pred,
        'probability': train_proba,
        'actual': y_train,
        'profit': train_df['profit']
    }).to_csv('./analysis_results/train_predictions.csv', index=False)

    # 測試集預測部分
    print("進行測試集預測...")
    filtered_test_data = pd.DataFrame()
    dates = test_df['date'].unique()

    for date in dates:
        daily_data = test_df[test_df['date'] == date].copy()
        daily_X = process_features(daily_data[feature_columns])
        daily_X_scaled = scaler.transform(daily_X)
        
        dtest_daily = xgb.DMatrix(daily_X_scaled, feature_names=feature_columns)
        daily_proba = model.predict(dtest_daily)  # 獲取機率值
        daily_pred = (daily_proba >= 0.5).astype(int)  # 轉換為0/1標籤

        daily_data['pred_probability'] = daily_proba
        daily_data['predicted'] = daily_pred
        top_30_stocks = daily_data.nlargest(30, 'pred_probability', 'all')
        filtered_test_data = pd.concat([filtered_test_data, top_30_stocks])

    # 最終測試集預測
    X_test_filtered = process_features(filtered_test_data[feature_columns])
    X_test_filtered_scaled = scaler.transform(X_test_filtered)

    dtest_filtered = xgb.DMatrix(X_test_filtered_scaled, feature_names=feature_columns)
    test_proba = model.predict(dtest_filtered)  # 獲取機率值
    test_pred = (test_proba >= 0.5).astype(int)  # 轉換為0/1標籤

    # 計算測試集權益曲線
    test_equity = calculate_equity_curve(
        test_pred, 
        filtered_test_data['label'], 
        filtered_test_data, 
        0, 
        len(filtered_test_data),
        open_threshold=0.00
)
    # 儲存模型和標準化參數
    print("儲存模型...")
    os.makedirs('./models', exist_ok=True)
    
    model_filename = f'./models/trained_model_{train_start_date[:4]}_{test_start_date[:4]}.json'
    model.save_model(model_filename)
    
    scaler_filename = f'./models/scaler_params_{train_start_date[:4]}_{test_start_date[:4]}.npz'
    np.savez(scaler_filename, 
             mean=scaler.mean_, 
             scale=scaler.scale_, 
             var=scaler.var_, 
             n_samples_seen=scaler.n_samples_seen_)

    # 計算績效指標
    equity_data = pd.DataFrame({
        'Return': test_equity['trade_profit'].values
    }, index=pd.to_datetime(test_equity['date']))

    performance_metrics = equity_plot(
        data=equity_data * 1000,
        Strategy='當沖策略',
        initial_cash=10000000,
        train_start_date=train_start_date,
        test_start_date=test_start_date
    )

    # 儲存每日選股結果
    result_filename = f'./analysis_results/daily_top30_stocks_{train_start_date[:4]}_{test_start_date[:4]}.csv'
    filtered_test_data.to_csv(result_filename, index=False)

    return model, train_equity, test_equity
if __name__ == "__main__":
    # 設定訓練期間為2024年全年，測試期間為2023年全年
    main(train_start_date='2020-01-01',
     train_end_date='2022-12-31',
     test_start_date='2023-01-01',
     test_end_date='2024-12-31',
     model_type='xgboost')