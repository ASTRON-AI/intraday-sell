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
import pickle
import time
# 引入自定義快取管理系統
from cache_system import CacheManager
import argparse
# 確保快取系統已初始化
CacheManager.initialize()
# 設定中文字型
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
def get_cache_path(cache_type, sub_type=None):
    """
    Get path to the appropriate cache directory
    
    Parameters:
    -----------
    cache_type : str
        Main cache type ('raw_data', 'features', 'models', 'predictions')
    sub_type : str, optional
        Subcategory of cache type (e.g., 'stock_price', 'broker_features')
        
    Returns:
    --------
    str
        Full path to the cache directory
    """
    base_cache_dir = 'D:/data_cache'
    
    if sub_type:
        cache_dir = os.path.join(base_cache_dir, cache_type, sub_type)
    else:
        cache_dir = os.path.join(base_cache_dir, cache_type)
    
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir
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
def get_stock_info() -> pd.DataFrame:
    """獲取股票基本資料"""
    conn = connect_db()
    query = """
    SELECT stock_id, stock_name, industry_category, type
    FROM tw_stock_info
    """
    df = pd.read_sql(query, conn)
    conn.close()
    return df
def create_cache_folder():
    """建立快取資料夾"""
    cache_folder = 'D:/data_cache'
    if not os.path.exists(cache_folder):
        os.makedirs(cache_folder)
    return cache_folder
def get_stock_data_with_lookback_cached(start_date: str, end_date: str, lookback_days: int = 30) -> pd.DataFrame:
    """從快取或資料庫獲取股票數據，包含回溯期間的數據
    
    Parameters:
    -----------
    start_date : str
        分析開始日期，格式為 'YYYY-MM-DD'
    end_date : str
        分析結束日期，格式為 'YYYY-MM-DD'
    lookback_days : int, default=30
        向前回溯的天數，用於計算滾動特徵
        
    Returns:
    --------
    pd.DataFrame
        包含回溯期間的股票數據
    """
    # 建立快取資料夾
    cache_folder = get_cache_path('raw_data', 'stock_price')
    
    # 建立快取檔案名稱
    cache_filename = f"{cache_folder}/stock_data_{start_date}_{end_date}_{lookback_days}.pkl"
    
    # 檢查快取檔案是否存在
    if os.path.exists(cache_filename):
        print(f"從快取讀取股票資料: {cache_filename}")
        with open(cache_filename, 'rb') as f:
            df = pickle.load(f)
        return df
    
    # 如果快取不存在，從資料庫讀取
    print(f"快取不存在，從資料庫讀取股票資料...")
    
    # 計算實際查詢的開始日期（回溯lookback_days天）
    start_date_dt = datetime.strptime(start_date, '%Y-%m-%d')
    lookback_start_date = start_date_dt - timedelta(days=lookback_days)
    lookback_start_str = lookback_start_date.strftime('%Y-%m-%d')
    
    print(f"原始開始日期: {start_date}")
    print(f"回溯後開始日期: {lookback_start_str} (回溯 {lookback_days} 天)")
    
    # 使用修改後的日期從資料庫獲取數據
    conn = connect_db()
    query = f"""
    SELECT date, stock_id, open, high, low, close, t_volume, t_money
    FROM tw_stock_daily_price
    WHERE date BETWEEN '{lookback_start_str}' AND '{end_date}'
    ORDER BY date, stock_id
    """
    df = pd.read_sql(query, conn)
    conn.close()
    
    # 標記哪些資料是實際分析期間的資料（非回溯期間）
    df['date'] = pd.to_datetime(df['date'])
    df['is_analysis_period'] = df['date'] >= pd.to_datetime(start_date)
    
    # 儲存到快取
    print(f"儲存股票資料到快取: {cache_filename}")
    with open(cache_filename, 'wb') as f:
        pickle.dump(df, f)
    
    return df

def get_margin_data_with_lookback_cached(start_date: str, end_date: str, lookback_days: int = 30) -> pd.DataFrame:
    """從快取或資料庫獲取融資融券數據，包含回溯期間的數據"""
    # 建立快取資料夾
    cache_folder = get_cache_path('raw_data', 'margin_data')
    
    # 建立快取檔案名稱
    cache_filename = f"{cache_folder}/margin_data_{start_date}_{end_date}_{lookback_days}.pkl"
    
    # 檢查快取檔案是否存在
    if os.path.exists(cache_filename):
        print(f"從快取讀取融資融券資料: {cache_filename}")
        with open(cache_filename, 'rb') as f:
            df = pickle.load(f)
        return df
    
    # 如果快取不存在，從資料庫讀取
    print(f"快取不存在，從資料庫讀取融資融券資料...")
    
    start_date_dt = datetime.strptime(start_date, '%Y-%m-%d')
    lookback_start_date = start_date_dt - timedelta(days=lookback_days)
    lookback_start_str = lookback_start_date.strftime('%Y-%m-%d')
    
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
    WHERE date BETWEEN '{lookback_start_str}' AND '{end_date}'
    ORDER BY date, stock_id
    """
    df = pd.read_sql(query, conn)
    conn.close()
    
    df['date'] = pd.to_datetime(df['date'])
    df['is_analysis_period'] = df['date'] >= pd.to_datetime(start_date)
    
    # 儲存到快取
    print(f"儲存融資融券資料到快取: {cache_filename}")
    with open(cache_filename, 'wb') as f:
        pickle.dump(df, f)
    
    return df

def get_foreign_trade_with_lookback_cached(start_date: str, end_date: str, lookback_days: int = 30) -> pd.DataFrame:
    """從快取或資料庫獲取外資交易數據，包含回溯期間的數據"""
    # 建立快取資料夾
    cache_folder = get_cache_path('raw_data', 'foreign_trade')
    
    # 建立快取檔案名稱
    cache_filename = f"{cache_folder}/foreign_trade_{start_date}_{end_date}_{lookback_days}.pkl"
    
    # 檢查快取檔案是否存在
    if os.path.exists(cache_filename):
        print(f"從快取讀取外資交易資料: {cache_filename}")
        with open(cache_filename, 'rb') as f:
            df = pickle.load(f)
        return df
    
    # 如果快取不存在，從資料庫讀取
    print(f"快取不存在，從資料庫讀取外資交易資料...")
    
    start_date_dt = datetime.strptime(start_date, '%Y-%m-%d')
    lookback_start_date = start_date_dt - timedelta(days=lookback_days)
    lookback_start_str = lookback_start_date.strftime('%Y-%m-%d')
    
    conn = connect_db()
    query = f"""
    SELECT date, stock_id, buy, sell
    FROM tw_stock_institutional_investors_buy_sell
    WHERE date BETWEEN '{lookback_start_str}' AND '{end_date}'
    AND name = 'Foreign_Investor'
    ORDER BY date, stock_id
    """
    df = pd.read_sql(query, conn)
    conn.close()
    
    df['date'] = pd.to_datetime(df['date'])
    df['is_analysis_period'] = df['date'] >= pd.to_datetime(start_date)
    
    # 儲存到快取
    print(f"儲存外資交易資料到快取: {cache_filename}")
    with open(cache_filename, 'wb') as f:
        pickle.dump(df, f)
    
    return df
def get_stock_info_cached() -> pd.DataFrame:
    """獲取股票基本資料，使用快取"""
    # 建立快取資料夾
    cache_folder = create_cache_folder()
    
    # 建立快取檔案名稱
    cache_filename = f"{cache_folder}/stock_info.pkl"
    
    # 檢查快取檔案是否存在
    if os.path.exists(cache_filename):
        print(f"從快取讀取股票基本資料: {cache_filename}")
        with open(cache_filename, 'rb') as f:
            df = pickle.load(f)
        return df
    
    # 如果快取不存在，從資料庫讀取
    print(f"快取不存在，從資料庫讀取股票基本資料...")
    
    conn = connect_db()
    query = """
    SELECT stock_id, stock_name, industry_category, type
    FROM tw_stock_info
    """
    df = pd.read_sql(query, conn)
    conn.close()
    
    # 儲存到快取
    print(f"儲存股票基本資料到快取: {cache_filename}")
    with open(cache_filename, 'wb') as f:
        pickle.dump(df, f)
    
    return df

def clear_cache(start_date=None, end_date=None, confirm=True):
    """清除快取資料
    
    Parameters:
    -----------
    start_date : str, optional
        如果提供，僅清除包含此開始日期的快取
    end_date : str, optional
        如果提供，僅清除包含此結束日期的快取
    confirm : bool, default=True
        如果為True，會要求使用者確認清除操作
    """
    cache_folder = 'D:/data_cache'
    if not os.path.exists(cache_folder):
        print("快取資料夾不存在，無需清除。")
        return
    
    cache_files = os.listdir(cache_folder)
    
    # 篩選檔案
    if start_date or end_date:
        filtered_files = []
        for file in cache_files:
            if start_date and start_date in file:
                filtered_files.append(file)
            elif end_date and end_date in file:
                filtered_files.append(file)
        cache_files = filtered_files
    
    if not cache_files:
        print("沒有符合條件的快取檔案。")
        return
    
    # 詢問確認
    if confirm:
        print(f"將清除以下 {len(cache_files)} 個快取檔案:")
        for file in cache_files[:10]:  # 僅顯示前10個
            print(f"- {file}")
        if len(cache_files) > 10:
            print(f"... 還有 {len(cache_files) - 10} 個檔案")
        
        confirm_input = input("確定要清除這些快取檔案嗎？(y/n): ")
        if confirm_input.lower() != 'y':
            print("取消清除操作。")
            return
    
    # 清除檔案
    for file in cache_files:
        file_path = os.path.join(cache_folder, file)
        try:
            os.remove(file_path)
            print(f"已清除: {file}")
        except Exception as e:
            print(f"清除 {file} 時發生錯誤: {e}")
    
    print(f"已清除 {len(cache_files)} 個快取檔案。")


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

import os
import pickle
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import gc  # Garbage collection

def get_broker_data_with_lookback_cached(start_date: str, end_date: str, lookback_days: int = 30) -> pd.DataFrame:
    """
    從資料庫獲取籌碼分點數據，包含回溯期間的數據，使用快取機制減少資料庫負擔
    """
    # 創建快取目錄
    cache_dir = get_cache_path('raw_data', 'broker_data')
    
    # 計算回溯後的日期
    start_date_dt = datetime.strptime(start_date, '%Y-%m-%d')
    lookback_start_date = start_date_dt - timedelta(days=lookback_days)
    lookback_start_str = lookback_start_date.strftime('%Y-%m-%d')
    
    # 生成快取檔案名稱
    cache_filename = f"{cache_dir}/broker_data_{lookback_start_str}_to_{end_date}.pkl"
    
    print(f"籌碼分點原始開始日期: {start_date}")
    print(f"籌碼分點回溯後開始日期: {lookback_start_str} (回溯 {lookback_days} 天)")
    
    # 檢查快取是否存在
    if os.path.exists(cache_filename):
        print(f"找到快取檔案: {cache_filename}，直接載入資料...")
        try:
            with open(cache_filename, 'rb') as f:
                result_df = pickle.load(f)
            
            # 驗證快取資料
            if 'date' in result_df.columns and 'stock_id' in result_df.columns:
                print(f"成功載入快取，資料筆數: {len(result_df)}")
                
                # 確保日期格式正確
                result_df['date'] = pd.to_datetime(result_df['date'])
                
                # 重新標記分析期間（以防查詢參數有變）
                result_df['is_analysis_period'] = result_df['date'] >= pd.to_datetime(start_date)
                
                return result_df
            else:
                print("快取資料結構不符，重新查詢...")
        except Exception as e:
            print(f"載入快取失敗: {e}，重新查詢資料...")
    
    # 未找到快取或載入失敗，從資料庫查詢
    all_data = []
    
    # 計算日期區間，按月分批查詢
    start_dt = datetime.strptime(lookback_start_str, '%Y-%m-%d')
    end_dt = datetime.strptime(end_date, '%Y-%m-%d')
    current_start = start_dt
    
    while current_start <= end_dt:
        # 計算當前批次的結束日期 (當月最後一天或end_date中較早的日期)
        if current_start.month == 12:
            next_month = datetime(current_start.year + 1, 1, 1)
        else:
            next_month = datetime(current_start.year, current_start.month + 1, 1)
        
        current_end = min(next_month - timedelta(days=1), end_dt)
        
        # 格式化日期為字串
        current_start_str = current_start.strftime('%Y-%m-%d')
        current_end_str = current_end.strftime('%Y-%m-%d')
        
        # 檢查月份快取
        month_cache_filename = f"{cache_dir}/broker_data_monthly_{current_start_str}_to_{current_end_str}.pkl"
        
        if os.path.exists(month_cache_filename):
            print(f"載入月份快取: {current_start_str} 到 {current_end_str}")
            try:
                with open(month_cache_filename, 'rb') as f:
                    month_data = pickle.load(f)
                all_data.append(month_data)
            except Exception as e:
                print(f"載入月份快取失敗: {e}，查詢資料庫...")
                month_data = query_db_for_month(current_start_str, current_end_str)
                all_data.append(month_data)
                
                # 儲存月份快取
                with open(month_cache_filename, 'wb') as f:
                    pickle.dump(month_data, f)
        else:
            print(f"查詢資料庫: {current_start_str} 到 {current_end_str}")
            month_data = query_db_for_month(current_start_str, current_end_str)
            all_data.append(month_data)
            
            # 儲存月份快取
            with open(month_cache_filename, 'wb') as f:
                pickle.dump(month_data, f)
        
        # 更新下一個批次的開始日期
        current_start = next_month
        
        # 強制執行垃圾回收
        gc.collect()
    
    # 合併所有批次的資料
    if not all_data:
        print("警告: 未找到任何籌碼分點資料")
        return pd.DataFrame(columns=['date', 'securities_trader', 'securities_trader_id', 
                                    'stock_id', 'buy_volume', 'sell_volume', 
                                    'buy_price', 'sell_price', 'is_analysis_period'])
    
    # 分批合併以減少記憶體使用
    result_df = pd.concat(all_data, ignore_index=True)
    
    # 標記哪些資料是實際分析期間的資料（非回溯期間）
    result_df['date'] = pd.to_datetime(result_df['date'])
    result_df['is_analysis_period'] = result_df['date'] >= pd.to_datetime(start_date)
    
    print(f"籌碼分點資料查詢完成，總筆數: {len(result_df)}")
    
    # 儲存最終快取
    try:
        with open(cache_filename, 'wb') as f:
            pickle.dump(result_df, f)
        print(f"已儲存籌碼分點資料快取: {cache_filename}")
    except Exception as e:
        print(f"儲存快取失敗: {e}")
    
    return result_df

def query_db_for_month(start_date, end_date):
    """從資料庫查詢指定月份的籌碼分點資料"""
    
    print(f"正在查詢籌碼分點資料: {start_date} 到 {end_date}")
    
    # 查詢當前批次的資料
    conn = connect_db()
    query = f"""
    SELECT 
        date, 
        securities_trader, 
        securities_trader_id, 
        stock_id, 
        buy_volume, 
        sell_volume, 
        buy_price, 
        sell_price
    FROM tw_stock_trading_daily_report_secid_agg
    WHERE date BETWEEN '{start_date}' AND '{end_date}'
    ORDER BY date, stock_id, securities_trader_id
    """
    
    # 使用 chunksize 分批讀取，減少記憶體使用
    chunks = []
    for chunk in pd.read_sql(query, conn, chunksize=100000):
        chunks.append(chunk)
        # 立即釋放不需要的記憶體
        gc.collect()
    
    conn.close()
    
    # 合併這個月的資料
    if chunks:
        return pd.concat(chunks, ignore_index=True)
    else:
        return pd.DataFrame()  # 返回空的DataFrame如果沒有資料


def generate_broker_features_optimized_cached(broker_data: pd.DataFrame, stock_data: pd.DataFrame) -> pd.DataFrame:
    """
    從籌碼分點資料生成特徵，使用記憶體優化方式，並支援快取機制
    """
    import gc  # 加入垃圾回收模組
    
    # 生成快取檔案名稱
    cache_dir = get_cache_path('features', 'broker_features')
    
    # 使用broker_data的日期範圍建立快取檔案名稱
    min_date = pd.to_datetime(broker_data['date']).min().strftime('%Y-%m-%d')
    max_date = pd.to_datetime(broker_data['date']).max().strftime('%Y-%m-%d')
    cache_filename = f"{cache_dir}/broker_features_{min_date}_to_{max_date}.pkl"
    
    # 檢查快取是否存在
    if os.path.exists(cache_filename):
        print(f"找到快取檔案: {cache_filename}，直接載入特徵...")
        try:
            with open(cache_filename, 'rb') as f:
                result_df = pickle.load(f)
            
            # 驗證快取資料
            if 'date' in result_df.columns and 'stock_id' in result_df.columns:
                print(f"成功載入籌碼分點特徵快取，資料筆數: {len(result_df)}")
                
                # 確保日期格式正確
                result_df['date'] = pd.to_datetime(result_df['date'])
                
                return result_df
            else:
                print("快取資料結構不符，重新計算特徵...")
        except Exception as e:
            print(f"載入快取失敗: {e}，重新計算特徵...")
    
    print("計算籌碼分點特徵...")
    # 確保日期格式正確
    broker_data['date'] = pd.to_datetime(broker_data['date'])
    stock_data['date'] = pd.to_datetime(stock_data['date'])
    
    # 只保留需要的列，減少記憶體使用
    stock_volumes = stock_data[['date', 'stock_id', 't_volume', 'is_analysis_period']].copy()
    
    # 確保數值型別正確，使用 inplace=True 減少記憶體分配
    broker_data['buy_volume'] = pd.to_numeric(broker_data['buy_volume'], errors='coerce')
    broker_data['sell_volume'] = pd.to_numeric(broker_data['sell_volume'], errors='coerce')
    broker_data.fillna({'buy_volume': 0, 'sell_volume': 0}, inplace=True)
    
    print("計算每日每支股票的總買賣量...")
    # 使用 groupby 的高效聚合，減少中間資料產生
    daily_totals = broker_data.groupby(['date', 'stock_id', 'is_analysis_period'], observed=True).agg({
        'buy_volume': 'sum',
        'sell_volume': 'sum'
    }).reset_index()
    
    daily_totals.rename(columns={
        'buy_volume': 'total_buy_volume',
        'sell_volume': 'total_sell_volume'
    }, inplace=True)
    
    # 釋放一部分記憶體
    if 'buy_price' in broker_data.columns and 'sell_price' in broker_data.columns:
        del broker_data['buy_price'], broker_data['sell_price']
    gc.collect()
    
    print("計算主力券商買賣情況...")
    # 定義主力券商名單
    major_broker_ids = [
        '9A00', '9600', '5920', '5850', '5380', '5860', '8440', '8840', '8890', '8880',
        '9200', '9800', '9900', '9A90', '9C00', '9B00', '595P', '9300', '9E00', '9F00'
    ]
    
    # 優化判斷主力券商的方式
    broker_data['is_major_broker'] = False
    for broker_id in major_broker_ids:
        broker_data.loc[broker_data['securities_trader_id'].str.startswith(broker_id[:2]), 'is_major_broker'] = True
    
    # 使用高效的篩選和聚合
    major_broker_stats = broker_data[broker_data['is_major_broker']].groupby(
        ['date', 'stock_id', 'is_analysis_period'], observed=True
    ).agg({
        'buy_volume': 'sum',
        'sell_volume': 'sum'
    }).reset_index()
    
    major_broker_stats.rename(columns={
        'buy_volume': 'major_broker_buy_volume',
        'sell_volume': 'major_broker_sell_volume'
    }, inplace=True)
    
    # 釋放記憶體
    broker_data.drop('is_major_broker', axis=1, inplace=True)
    gc.collect()
    
    print("計算券商淨買賣量...")
    # 計算每個券商的淨買賣量
    broker_data['net_buy_volume'] = broker_data['buy_volume'] - broker_data['sell_volume']
    
    # 使用更高效的方法處理前五大買賣超券商
    print("識別前五大買超券商...")
    
    # 分批處理日期，減少記憶體使用
    unique_dates = broker_data['date'].unique()
    batch_size = 10  # 每批處理10天
    
    top5_buyer_results = []
    top5_seller_results = []
    
    for i in range(0, len(unique_dates), batch_size):
        batch_dates = unique_dates[i:i+batch_size]
        batch_data = broker_data[broker_data['date'].isin(batch_dates)]
        
        # 對每個日期每個股票計算淨買賣量並排序
        grouped = batch_data.groupby(['date', 'stock_id', 'is_analysis_period'])
        
        # 使用apply函數找出前5大買超
        def get_top5_buyers(group):
            return group.nlargest(5, 'net_buy_volume')[['buy_volume', 'sell_volume', 'net_buy_volume']]
        
        # 使用apply函數找出前5大賣超
        def get_top5_sellers(group):
            return group.nsmallest(5, 'net_buy_volume')[['buy_volume', 'sell_volume', 'net_buy_volume']]
        
        buyer_stats = grouped.apply(get_top5_buyers).reset_index()
        seller_stats = grouped.apply(get_top5_sellers).reset_index()
        
        # 合併相同日期和股票的前5大買超券商資料
        buyer_agg = buyer_stats.groupby(['date', 'stock_id', 'is_analysis_period']).agg({
            'buy_volume': 'sum',
            'sell_volume': 'sum',
            'net_buy_volume': 'sum'
        }).reset_index()
        
        # 合併相同日期和股票的前5大賣超券商資料
        seller_agg = seller_stats.groupby(['date', 'stock_id', 'is_analysis_period']).agg({
            'buy_volume': 'sum',
            'sell_volume': 'sum',
            'net_buy_volume': 'sum'
        }).reset_index()
        
        top5_buyer_results.append(buyer_agg)
        top5_seller_results.append(seller_agg)
        
        # 釋放記憶體
        del batch_data, grouped, buyer_stats, seller_stats, buyer_agg, seller_agg
        gc.collect()
    
    # 合併結果
    top5_buyer_stats = pd.concat(top5_buyer_results, ignore_index=True)
    top5_seller_stats = pd.concat(top5_seller_results, ignore_index=True)
    
    # 重命名列
    top5_buyer_stats.rename(columns={
        'buy_volume': 'top5_buyer_buy_volume',
        'sell_volume': 'top5_buyer_sell_volume',
        'net_buy_volume': 'top5_buyer_net_volume'
    }, inplace=True)
    
    top5_seller_stats.rename(columns={
        'buy_volume': 'top5_seller_buy_volume',
        'sell_volume': 'top5_seller_sell_volume',
        'net_buy_volume': 'top5_seller_net_volume'
    }, inplace=True)
    
    # 釋放記憶體
    del top5_buyer_results, top5_seller_results
    gc.collect()
    
    print("計算券商集中度指標...")
    # 分批處理券商集中度計算
    broker_concentration_results = []
    
    for i in range(0, len(unique_dates), batch_size):
        batch_dates = unique_dates[i:i+batch_size]
        batch_data = broker_data[broker_data['date'].isin(batch_dates)]
        
        # 對每個日期每個股票計算買賣集中度
        def calc_concentration(group):
            # 買入集中度
            buy_total = group['buy_volume'].sum()
            if buy_total > 0:
                top3_buy = group.nlargest(3, 'buy_volume')['buy_volume'].sum()
                buy_concentration = top3_buy / buy_total
            else:
                buy_concentration = 0
                
            # 賣出集中度
            sell_total = group['sell_volume'].sum()
            if sell_total > 0:
                top3_sell = group.nlargest(3, 'sell_volume')['sell_volume'].sum()
                sell_concentration = top3_sell / sell_total
            else:
                sell_concentration = 0
                
            return pd.Series({
                'broker_buy_concentration': buy_concentration,
                'broker_sell_concentration': sell_concentration
            })
        
        # 計算每個日期每個股票的集中度
        concentration_stats = batch_data.groupby(['date', 'stock_id', 'is_analysis_period']).apply(calc_concentration).reset_index()
        broker_concentration_results.append(concentration_stats)
        
        # 釋放記憶體
        del batch_data
        gc.collect()
    
    # 合併結果
    broker_concentration = pd.concat(broker_concentration_results, ignore_index=True)
    
    # 釋放記憶體
    del broker_concentration_results
    gc.collect()
    
    print("計算外資券商指標...")
    # 判斷外資券商
    broker_data['is_foreign_broker'] = broker_data['securities_trader_id'].str.startswith('9')
    
    # 分批處理外資券商統計
    foreign_broker_results = []
    
    for i in range(0, len(unique_dates), batch_size):
        batch_dates = unique_dates[i:i+batch_size]
        batch_data = broker_data[broker_data['date'].isin(batch_dates) & broker_data['is_foreign_broker']]
        
        # 計算每個日期每個股票的外資券商買賣情況
        foreign_stats = batch_data.groupby(['date', 'stock_id', 'is_analysis_period']).agg({
            'buy_volume': 'sum',
            'sell_volume': 'sum'
        }).reset_index()
        
        foreign_stats.rename(columns={
            'buy_volume': 'foreign_broker_buy_volume',
            'sell_volume': 'foreign_broker_sell_volume'
        }, inplace=True)
        
        foreign_broker_results.append(foreign_stats)
        
        # 釋放記憶體
        del batch_data, foreign_stats
        gc.collect()
    
    # 合併結果
    foreign_broker_stats = pd.concat(foreign_broker_results, ignore_index=True)
    
    # 釋放記憶體
    del foreign_broker_results, broker_data
    gc.collect()
    
    print("合併所有特徵...")
    # 合併所有資料 - 使用分步合併減少記憶體壓力
    result = pd.merge(daily_totals, major_broker_stats, on=['date', 'stock_id', 'is_analysis_period'], how='left')
    del major_broker_stats
    gc.collect()
    
    result = pd.merge(result, top5_buyer_stats, on=['date', 'stock_id', 'is_analysis_period'], how='left')
    del top5_buyer_stats
    gc.collect()
    
    result = pd.merge(result, top5_seller_stats, on=['date', 'stock_id', 'is_analysis_period'], how='left')
    del top5_seller_stats
    gc.collect()
    
    result = pd.merge(result, broker_concentration, on=['date', 'stock_id', 'is_analysis_period'], how='left')
    del broker_concentration
    gc.collect()
    
    result = pd.merge(result, foreign_broker_stats, on=['date', 'stock_id', 'is_analysis_period'], how='left')
    del foreign_broker_stats
    gc.collect()
    
    result = pd.merge(result, stock_volumes, on=['date', 'stock_id', 'is_analysis_period'], how='left')
    del stock_volumes
    gc.collect()
    
    # 填充缺失值
    result.fillna(0, inplace=True)
    
    print("計算籌碼分點特徵...")
    # 計算前綴為 broker_ 的特徵
    # 1. 買賣比和淨買量
    result['broker_buy_sell_ratio'] = result['total_buy_volume'] / (result['total_sell_volume'] + 1e-10)
    result['broker_net_volume'] = result['total_buy_volume'] - result['total_sell_volume']
    result['broker_net_ratio'] = result['broker_net_volume'] / (result['t_volume'] + 1e-10)
    
    # 2. 主力券商指標
    result['broker_major_buy_ratio'] = result['major_broker_buy_volume'] / (result['total_buy_volume'] + 1e-10)
    result['broker_major_sell_ratio'] = result['major_broker_sell_volume'] / (result['total_sell_volume'] + 1e-10)
    result['broker_major_net_volume'] = result['major_broker_buy_volume'] - result['major_broker_sell_volume']
    result['broker_major_net_ratio'] = result['broker_major_net_volume'] / (result['t_volume'] + 1e-10)
    
    # 3. 前五大買超券商指標
    result['broker_top5_buyer_ratio'] = result['top5_buyer_buy_volume'] / (result['total_buy_volume'] + 1e-10)
    result['broker_top5_buyer_net_ratio'] = result['top5_buyer_net_volume'] / (result['t_volume'] + 1e-10)
    
    # 4. 前五大賣超券商指標
    result['broker_top5_seller_ratio'] = result['top5_seller_sell_volume'] / (result['total_sell_volume'] + 1e-10)
    result['broker_top5_seller_net_ratio'] = result['top5_seller_net_volume'] / (result['t_volume'] + 1e-10)
    
    # 5. 外資券商指標
    result['broker_foreign_buy_ratio'] = result['foreign_broker_buy_volume'] / (result['total_buy_volume'] + 1e-10)
    result['broker_foreign_sell_ratio'] = result['foreign_broker_sell_volume'] / (result['total_sell_volume'] + 1e-10)
    result['broker_foreign_net_volume'] = result['foreign_broker_buy_volume'] - result['foreign_broker_sell_volume']
    result['broker_foreign_net_ratio'] = result['broker_foreign_net_volume'] / (result['t_volume'] + 1e-10)
    
    print("計算時間序列特徵...")
    # 計算時間序列特徵 - 使用transform減少記憶體使用
    result = result.sort_values(['stock_id', 'date'])
    
    # 使用分批處理來計算移動平均
    # 每次處理一種特徵，減少記憶體壓力
    for feature in ['broker_major_net_ratio', 'broker_buy_sell_ratio', 'broker_top5_buyer_net_ratio', 'broker_top5_seller_net_ratio']:
        for window in [3, 5, 10]:
            column_name = f'{feature}_{window}d_ma'
            print(f"計算 {column_name}...")
            
            # 使用transform更高效
            result[column_name] = result.groupby('stock_id')[feature].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean()
            )
            
            # 每計算完一個特徵就回收一次記憶體
            gc.collect()
    
    # 計算變化率特徵
    print("計算變化率特徵...")
    result['broker_major_net_ratio_change'] = result.groupby('stock_id')['broker_major_net_ratio'].diff()
    result['broker_buy_sell_ratio_change'] = result.groupby('stock_id')['broker_buy_sell_ratio'].diff()
    result['broker_major_momentum'] = result['broker_major_net_ratio'] - result['broker_major_net_ratio_5d_ma']
    
    # 處理極端值和無限值
    result.replace([np.inf, -np.inf], 0, inplace=True)
    
    # 選擇最終特徵
    broker_features = [
        'date', 'stock_id', 'is_analysis_period',
        # 總量特徵
        'broker_buy_sell_ratio', 'broker_net_ratio',
        # 主力券商特徵
        'broker_major_buy_ratio', 'broker_major_sell_ratio', 'broker_major_net_ratio',
        # 前五大券商特徵
        'broker_top5_buyer_ratio', 'broker_top5_buyer_net_ratio',
        'broker_top5_seller_ratio', 'broker_top5_seller_net_ratio',
        # 集中度特徵
        'broker_buy_concentration', 'broker_sell_concentration',
        # 時間序列特徵
        'broker_major_net_ratio_3d_ma', 'broker_major_net_ratio_5d_ma', 'broker_major_net_ratio_10d_ma',
        'broker_buy_sell_ratio_3d_ma', 'broker_buy_sell_ratio_5d_ma', 'broker_buy_sell_ratio_10d_ma',
        'broker_top5_buyer_net_ratio_5d_ma', 'broker_top5_seller_net_ratio_5d_ma',
        # 變化率特徵
        'broker_major_net_ratio_change', 'broker_buy_sell_ratio_change', 'broker_major_momentum',
        # 外資特徵
        'broker_foreign_buy_ratio', 'broker_foreign_sell_ratio', 'broker_foreign_net_ratio'
    ]
    
    # 確保所有特徵都存在
    for feature in broker_features:
        if feature not in result.columns and feature not in ['date', 'stock_id', 'is_analysis_period']:
            result[feature] = 0
    
    # 只保留需要的列
    final_result = result[broker_features].copy()
    
    # 儲存快取
    try:
        with open(cache_filename, 'wb') as f:
            pickle.dump(final_result, f)
        print(f"已儲存籌碼分點特徵快取: {cache_filename}")
    except Exception as e:
        print(f"儲存特徵快取失敗: {e}")
    
    # 釋放記憶體
    del result
    gc.collect()
    
    print("籌碼分點特徵計算完成")
    return final_result

# 修改原來的函數為直接調用快取版本的函數

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


# def calculate_historical_features(df: pd.DataFrame, window_size: int = 5) -> pd.DataFrame:
#     """
#     計算個股前N天的開高低收和成交量
    
#     Parameters:
#     -----------
#     df : pd.DataFrame
#         需包含 stock_id, date, open, high, low, close, t_volume 欄位的DataFrame
#     window_size : int, default=5
#         要往前看幾天的資料
        
#     Returns:
#     --------
#     pd.DataFrame
#         包含歷史特徵的DataFrame
#     """
#     df = df.copy()
#     df['date'] = pd.to_datetime(df['date'])
    
#     # 排序資料
#     df = df.sort_values(['stock_id', 'date'])
    
#     # 對每個股票分別計算歷史特徵
#     for i in range(1, window_size + 1):
#         # 計算前N天的價格資訊
#         df[f'open_d{i}'] = df.groupby('stock_id')['open'].shift(i)
#         df[f'high_d{i}'] = df.groupby('stock_id')['high'].shift(i)
#         df[f'low_d{i}'] = df.groupby('stock_id')['low'].shift(i)
#         df[f'close_d{i}'] = df.groupby('stock_id')['close'].shift(i)
#         df[f'volume_d{i}'] = df.groupby('stock_id')['t_volume'].shift(i)
        
#         # 計算與前N天的價格變化比例
#         df[f'price_change_d{i}'] = (df['close'] - df[f'close_d{i}']) / df[f'close_d{i}']
#         df[f'volume_change_d{i}'] = (df['t_volume'] - df[f'volume_d{i}']) / df[f'volume_d{i}']
    
#     return df
# def calculate_price_features(df: pd.DataFrame) -> pd.DataFrame:
#     """
#     計算價格相關的特徵
#     """
#     df = df.copy()
#     df = df.sort_values(['stock_id', 'date'])
    
#     # 計算昨日收盤價
#     df['prev_close'] = df.groupby('stock_id')['close'].shift(1)
    
#     # 當日OHLC相對於昨日收盤的變化幅度
#     df['open_change'] = (df['open'] - df['prev_close']) / df['prev_close']
#     df['high_change'] = (df['high'] - df['prev_close']) / df['prev_close']
#     df['low_change'] = (df['low'] - df['prev_close']) / df['prev_close']
#     df['close_change'] = (df['close'] - df['prev_close']) / df['prev_close']
    
#     # 當日價格區間特徵
#     df['day_range'] = (df['high'] - df['low']) / df['prev_close']
#     df['up_shadow'] = (df['high'] - df[['open', 'close']].max(axis=1)) / df['prev_close']
#     df['down_shadow'] = (df[['open', 'close']].min(axis=1) - df['low']) / df['prev_close']
#     df['body'] = abs(df['close'] - df['open']) / df['prev_close']
    
#     return df

# def calculate_volume_features(df: pd.DataFrame) -> pd.DataFrame:
#     """
#     計算成交量相關的特徵
#     """
#     df = df.copy()
    
#     # 計算前N日平均成交量和金額
#     for days in [5, 10, 20]:
#         df[f'volume_ma{days}'] = df.groupby('stock_id')['t_volume'].rolling(days).mean().reset_index(0, drop=True)
#         df[f'money_ma{days}'] = df.groupby('stock_id')['t_money'].rolling(days).mean().reset_index(0, drop=True)
    
#     # 計算昨日成交量和金額
#     df['prev_volume'] = df.groupby('stock_id')['t_volume'].shift(1)
#     df['prev_money'] = df.groupby('stock_id')['t_money'].shift(1)
    
#     # 相對於前一日的變化
#     df['volume_change'] = (df['t_volume'] - df['prev_volume']) / df['prev_volume']
#     df['money_change'] = (df['t_money'] - df['prev_money']) / df['prev_money']
    
#     # 相對於移動平均的變化
#     df['volume_vs_5ma'] = df['t_volume'] / df['volume_ma5']
#     df['volume_vs_20ma'] = df['t_volume'] / df['volume_ma20']
    
#     return df


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
    
def get_tpexno_intraday_stocks(date):
    """
    讀取指定日期的不可當沖股票清單
    """
    date_str = date.strftime('%Y%m%d')
    file_path = f'./output_stock_data/{date_str}_tpexno_intraday.csv'
    try:
        no_intraday_df = pd.read_csv(file_path)
        return set(no_intraday_df['stock_id'].astype(str))
    except FileNotFoundError:
        print(f"Warning: No intraday file not found for date {date_str}")
        return set()


def get_nosell_intraday_stocks(date):
    """
    讀取指定日期的不可當沖股票清單
    """
    date_str = date.strftime('%Y%m%d')
    file_path = f'./nosell_output2020_2025/{date_str}_no_intraday_all.csv'
    try:
        no_intraday_df = pd.read_csv(file_path)
        return set(no_intraday_df['stock_id'].astype(str))
    except FileNotFoundError:
        print(f"Warning: No intraday file not found for date {date_str}")
        return set()



import xgboost as xgb
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


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
        'max_depth': 8,  # 降低树的深度
        'learning_rate': 0.02,  # 降低学习率
        'n_estimators': 2000,
        'objective': 'binary:logistic',
        'scale_pos_weight': neg_pos_ratio,  # 根据实际比例设置
        'subsample': 0.8,  # 降低采样比例
        'colsample_bytree': 0.8,  # 降低特征采样比例
        'min_child_weight': 3,  # 增加最小子节点权重
        'gamma': 0.2,  # 增加分裂阈值
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

 #修改 main 函數中的模型選擇部分
def train_selected_model(model_type, X_train_scaled, X_test_scaled, y_train, y_test, feature_columns):
    """
    根據選擇的模型類型訓練相應的模型
    """
    # if model_type == 'tabnet':
    #     return train_model_tabnet(X_train_scaled, X_test_scaled, y_train, y_test, feature_columns)
    # elif model_type == 'lightgbm':
    #     return train_model_lightgbm(X_train_scaled, X_test_scaled, y_train, y_test, feature_columns)
    # else:  # 默認使用原來的 XGBoost
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

def calculate_equity_curve(predictions, y_true, df, start_idx, end_idx, start_date, end_date, capital_per_trade=1000000, open_threshold=0.03):
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
    capital_per_trade : float, default=1000000
        每筆交易的固定資金額度，預設為100萬
    open_threshold : float, default=0.03
        開盤漲幅門檻，預設為3%
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

    # 計算每筆交易的股數
    trades['shares'] = capital_per_trade / trades['close']

    # 計算停損條件：High > Close * 1.08
    trades['stop_loss'] = trades['next_day_high'] > trades['close'] * 1.08

    # 處理停損的情況
    stop_loss_mask = (trades['valid_trade']) & trades['stop_loss']
    trades.loc[stop_loss_mask, 'trade_profit'] = (
        trades['shares'] * (trades['next_day_open'] - trades['close'] * 1.08) -
        (trades['shares'] * trades['next_day_open'] * 0.0015 + 
         trades['shares'] * (trades['next_day_open'] + trades['close'] * 1.08) * 0.001425 * 0.3)
    )

    # 處理未停損的交易
    normal_trade_mask = (trades['valid_trade']) & ~trades['stop_loss']
    trades.loc[normal_trade_mask, 'trade_profit'] = (
        trades['shares'] * trades['profit'] -
        trades['shares'] * trades['profit_volume']
    )

    # 統計交易結果
    total_trades = trades['valid_trade'].sum()
    profitable_trades = ((trades['valid_trade']) & (trades['trade_profit'] > 0)).sum()
    losing_trades = ((trades['valid_trade']) & (trades['trade_profit'] < 0)).sum()
    win_rate = profitable_trades / total_trades if total_trades > 0 else 0
    total_profit = trades['trade_profit'].sum()

    print(f"\n交易統計 (每筆資金: {capital_per_trade:,.0f}, 開盤漲幅門檻: {open_threshold:.2%}):")
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
    trades_filtered = trades[trades['predicted'] == 1]
    trades_filtered.to_csv(f'./analysis_results/detailed_trades_improved_{start_date[:4]}_{end_date[:4]}_cache_withbroker.csv', index=False)
    return daily_profits


import os
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import StandardScaler


def calculate_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    計算價格相關的特徵，使用前綴標記特徵類型
    
    Parameters:
    -----------
    df : pd.DataFrame
        需包含 stock_id, date, open, high, low, close 欄位的DataFrame
        
    Returns:
    --------
    pd.DataFrame
        包含價格特徵的DataFrame
    """
    # 生成快取檔案名稱 - 使用輸入資料的日期範圍
    min_date = pd.to_datetime(df['date']).min().strftime('%Y-%m-%d')
    max_date = pd.to_datetime(df['date']).max().strftime('%Y-%m-%d')
    filename = f"price_features_{min_date}_to_{max_date}.pkl"
    
    # 檢查快取是否存在
    if CacheManager.exists('features', 'price_features', filename):
        print(f"從快取讀取價格特徵: {filename}")
        result_df = CacheManager.load('features', 'price_features', filename)
        if result_df is not None:
            return result_df
    
    print("計算價格特徵...")
    df = df.copy()
    df = df.sort_values(['stock_id', 'date'])
    
    # 計算昨日收盤價
    df['prev_close'] = df.groupby('stock_id')['close'].shift(1)
    
    # 當日OHLC相對於昨日收盤的變化幅度，加上前綴p_
    df['p_open_change'] = (df['open'] - df['prev_close']) / df['prev_close']
    df['p_high_change'] = (df['high'] - df['prev_close']) / df['prev_close']
    df['p_low_change'] = (df['low'] - df['prev_close']) / df['prev_close']
    df['p_close_change'] = (df['close'] - df['prev_close']) / df['prev_close']
    
    # 當日價格區間特徵
    df['p_day_range'] = (df['high'] - df['low']) / df['prev_close']
    df['p_up_shadow'] = (df['high'] - df[['open', 'close']].max(axis=1)) / df['prev_close']
    df['p_down_shadow'] = (df[['open', 'close']].min(axis=1) - df['low']) / df['prev_close']
    df['p_body'] = abs(df['close'] - df['open']) / df['prev_close']
    
    # 價格變化加速度 (二階導數)
    df['p_close_change_accel'] = df['p_close_change'] - df.groupby('stock_id')['p_close_change'].shift(1)
    
    # 保留部分原始價格信息 (相對於市場平均水平)
    # 計算每日所有股票的平均收盤價
    daily_avg_close = df.groupby('date')['close'].transform('mean')
    df['p_rel_price'] = df['close'] / daily_avg_close  # 相對價格水平
    
    # 添加漲跌趨勢特徵 (連續上漲/下跌天數)
    df['p_up_streak'] = 0
    df['p_down_streak'] = 0
    
    # 按股票分組處理
    for stock_id, group in df.groupby('stock_id'):
        group = group.sort_values('date')
        up_streak = 0
        down_streak = 0
        
        for i, row in group.iterrows():
            if row['p_close_change'] > 0:
                up_streak += 1
                down_streak = 0
            elif row['p_close_change'] < 0:
                down_streak += 1
                up_streak = 0
            else:
                up_streak = 0
                down_streak = 0
                
            df.loc[i, 'p_up_streak'] = up_streak
            df.loc[i, 'p_down_streak'] = down_streak
    
    # 儲存到快取
    CacheManager.save(df, 'features', 'price_features', filename)
    print(f"儲存價格特徵快取完成")
    
    return df

def calculate_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    計算成交量相關的特徵，使用前綴標記特徵類型
    
    Parameters:
    -----------
    df : pd.DataFrame
        需包含 stock_id, date, t_volume, t_money 欄位的DataFrame
        
    Returns:
    --------
    pd.DataFrame
        包含成交量特徵的DataFrame
    """
    # 生成快取檔案名稱 - 使用輸入資料的日期範圍
    min_date = pd.to_datetime(df['date']).min().strftime('%Y-%m-%d')
    max_date = pd.to_datetime(df['date']).max().strftime('%Y-%m-%d')
    filename = f"volume_features_{min_date}_to_{max_date}.pkl"
    
    # 檢查快取是否存在
    if CacheManager.exists('features', 'volume_features', filename):
        print(f"從快取讀取成交量特徵: {filename}")
        result_df = CacheManager.load('features', 'volume_features', filename)
        if result_df is not None:
            return result_df
    
    print("計算成交量特徵...")
    df = df.copy()
    
    # 計算前N日平均成交量和金額
    for days in [5, 10, 20]:
        df[f'vol_ma{days}'] = df.groupby('stock_id')['t_volume'].rolling(days).mean().reset_index(0, drop=True)
        df[f'vol_money_ma{days}'] = df.groupby('stock_id')['t_money'].rolling(days).mean().reset_index(0, drop=True)
    
    # 計算昨日成交量和金額
    df['prev_volume'] = df.groupby('stock_id')['t_volume'].shift(1)
    df['prev_money'] = df.groupby('stock_id')['t_money'].shift(1)
    
    # 相對於前一日的變化
    df['vol_change'] = (df['t_volume'] - df['prev_volume']) / df['prev_volume']
    df['vol_money_change'] = (df['t_money'] - df['prev_money']) / df['prev_money']
    
    # 相對於移動平均的變化
    df['vol_vs_5ma'] = df['t_volume'] / df['vol_ma5']
    df['vol_vs_20ma'] = df['t_volume'] / df['vol_ma20']
    
    # 成交量波動性 (過去N天成交量的標準差/平均)
    for days in [5, 10]:
        vol_std = df.groupby('stock_id')['t_volume'].rolling(days).std().reset_index(0, drop=True)
        df[f'vol_volatility_{days}d'] = vol_std / df[f'vol_ma{days}']
    
    # 成交量趨勢 (線性回歸斜率)
    # 使用簡化方法近似計算趨勢
    df['vol_trend_5d'] = (df['t_volume'] - df.groupby('stock_id')['t_volume'].shift(5)) / 5 / df['vol_ma5']
    
    # 金額/成交量比率 (平均交易價格)
    df['vol_avg_price'] = df['t_money'] / df['t_volume']
    df['vol_avg_price_change'] = df['vol_avg_price'] / df.groupby('stock_id')['vol_avg_price'].shift(1) - 1
    
    # 處理極端值
    vol_columns = [col for col in df.columns if col.startswith('vol_')]
    df[vol_columns] = df[vol_columns].replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # 儲存到快取
    CacheManager.save(df, 'features', 'volume_features', filename)
    print(f"儲存成交量特徵快取完成")
    
    return df

def calculate_margin_features(margin_data: pd.DataFrame) -> pd.DataFrame:
    """
    計算融資融券相關的特徵，並將 NaN 值替換為 0，使用前綴標記特徵類型
    
    Parameters:
    -----------
    margin_data : pd.DataFrame
        包含融資融券資料的DataFrame
        
    Returns:
    --------
    pd.DataFrame
        包含融資融券特徵的DataFrame
    """
    # 生成快取檔案名稱 - 使用輸入資料的日期範圍
    min_date = pd.to_datetime(margin_data['date']).min().strftime('%Y-%m-%d')
    max_date = pd.to_datetime(margin_data['date']).max().strftime('%Y-%m-%d')
    filename = f"margin_features_{min_date}_to_{max_date}.pkl"
    
    # 檢查快取是否存在
    if CacheManager.exists('features', 'margin_features', filename):
        print(f"從快取讀取融資融券特徵: {filename}")
        result_df = CacheManager.load('features', 'margin_features', filename)
        if result_df is not None:
            return result_df
    
    print("計算融資融券特徵...")
    # 複製資料以避免修改原始資料
    df = margin_data.copy()
    df['date'] = pd.to_datetime(df['date'])
    
    # 1. 基礎融資比率
    # 融資使用率
    df['m_usage_ratio'] = df['margin_purchase_today_balance'] / df['margin_purchase_limit']
    
    # 融資餘額變動率
    df['m_balance_change_ratio'] = (df['margin_purchase_today_balance'] - 
                                   df['margin_purchase_yesterday_balance']) / \
                                   df['margin_purchase_yesterday_balance']
    
    # 融資買進強度
    df['m_buy_strength'] = df['margin_purchase_buy'] / df['margin_purchase_yesterday_balance']
    
    # 融資賣出強度
    df['m_sell_strength'] = df['margin_purchase_sell'] / df['margin_purchase_yesterday_balance']
    
    # 2. 基礎融券比率
    # 融券使用率
    df['s_usage_ratio'] = df['short_sale_today_balance'] / df['short_sale_limit']
    
    # 融券餘額變動率
    df['s_balance_change_ratio'] = (df['short_sale_today_balance'] - 
                                  df['short_sale_yesterday_balance']) / \
                                  df['short_sale_yesterday_balance']
    
    # 融券賣出強度
    df['s_sell_strength'] = df['short_sale_sell'] / df['short_sale_yesterday_balance']
    
    # 融券買進強度
    df['s_buy_strength'] = df['short_sale_buy'] / df['short_sale_yesterday_balance']
    
    # 3. 綜合指標
    # 資券比
    df['ms_ratio'] = df['margin_purchase_today_balance'] / df['short_sale_today_balance']
    
    # 資券維持率差異
    df['ms_maintain_spread'] = (df['margin_purchase_today_balance'] / df['margin_purchase_limit']) - \
                             (df['short_sale_today_balance'] / df['short_sale_limit'])
    
    # 4. 高級融資融券特徵
    # 融資動能 (融資買進強度 - 融資賣出強度)
    df['m_momentum'] = df['m_buy_strength'] - df['m_sell_strength']
    
    # 融券動能 (融券賣出強度 - 融券買進強度)
    df['s_momentum'] = df['s_sell_strength'] - df['s_buy_strength']
    
    # 融資融券活動總強度 (衡量市場參與程度)
    df['ms_activity'] = (df['margin_purchase_buy'] + df['margin_purchase_sell'] +
                       df['short_sale_buy'] + df['short_sale_sell']) / \
                       (df['margin_purchase_yesterday_balance'] + df['short_sale_yesterday_balance'])
    
    # 融資融券淨流入率 (正值表示資金淨流入，負值表示資金淨流出)
    df['ms_net_flow'] = (df['margin_purchase_buy'] - df['margin_purchase_sell'] -
                       df['short_sale_sell'] + df['short_sale_buy']) / \
                       (df['margin_purchase_yesterday_balance'] + df['short_sale_yesterday_balance'])
    
    # 將無限值替換為 NaN
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # 定義要處理的比率欄位
    ratio_columns = [col for col in df.columns if col.startswith(('m_', 's_', 'ms_'))]
    
    # 5. 計算5日移動平均和變化速度
    for col in ratio_columns:
        # 5日移動平均
        df[f'{col}_5d_ma'] = df.groupby('stock_id')[col].rolling(window=5).mean().reset_index(0, drop=True)
        
        # 變化速度 (一階導數)
        df[f'{col}_change'] = df[col] - df.groupby('stock_id')[col].shift(1)
        
        # 加速度 (二階導數)
        df[f'{col}_accel'] = df[f'{col}_change'] - df.groupby('stock_id')[f'{col}_change'].shift(1)
    
    # 6. 將所有 NaN 值替換為 0
    columns_to_fill = ratio_columns + [f'{col}_5d_ma' for col in ratio_columns] + \
                     [f'{col}_change' for col in ratio_columns] + \
                     [f'{col}_accel' for col in ratio_columns]
    df[columns_to_fill] = df[columns_to_fill].fillna(0)
    
    # 7. 保留原始日期和股票代碼
    result_df = df[['date', 'stock_id'] + columns_to_fill]
    
    # 儲存到快取
    CacheManager.save(result_df, 'features', 'margin_features', filename)
    print(f"儲存融資融券特徵快取完成")
    
    return result_df

def generate_foreign_features_with_lookback(start_date: str, end_date: str, lookback_days: int = 30) -> pd.DataFrame:
    """
    生成外資買賣相關特徵，包含回溯期間的資料，使用前綴標記特徵類型
    
    Parameters:
    -----------
    start_date : str
        分析開始日期，格式為 'YYYY-MM-DD'
    end_date : str
        分析結束日期，格式為 'YYYY-MM-DD'
    lookback_days : int, default=30
        向前回溯的天數，用於計算滾動特徵
        
    Returns:
    --------
    pd.DataFrame
        包含外資買賣特徵的DataFrame
    """
    # 生成快取檔案名稱
    filename = f"foreign_features_{start_date}_{end_date}_{lookback_days}.pkl"
    
    # 檢查快取是否存在
    if CacheManager.exists('features', 'foreign_features', filename):
        print(f"從快取讀取外資特徵: {filename}")
        df = CacheManager.load('features', 'foreign_features', filename)
        if df is not None:
            return df
    
    # 如果快取不存在或載入失敗，重新計算特徵
    print(f"快取不存在，重新計算外資特徵...")
    
    # 使用包含回溯期間的數據
    stock_df = get_stock_data_with_lookback_cached(start_date, end_date, lookback_days)
    foreign_df = get_foreign_trade_with_lookback_cached(start_date, end_date, lookback_days)
    
    # 確保日期格式
    stock_df['date'] = pd.to_datetime(stock_df['date'])
    foreign_df['date'] = pd.to_datetime(foreign_df['date'])
    
    # 合併資料
    merged_df = pd.merge(
        stock_df[['date', 'stock_id', 't_volume', 'is_analysis_period']], 
        foreign_df[['date', 'stock_id', 'buy', 'sell']], 
        on=['date', 'stock_id'], 
        how='inner'
    )
    
    # 計算每日外資買賣比例 (使用f_前綴)
    merged_df['f_buy_ratio'] = merged_df['buy'] / merged_df['t_volume']  # 外資買進比例
    merged_df['f_sell_ratio'] = merged_df['sell'] / merged_df['t_volume']  # 外資賣出比例
    merged_df['f_net_ratio'] = (merged_df['buy'] - merged_df['sell']) / merged_df['t_volume']  # 外資淨買超比例
    
    # 排序資料以確保時間序列正確
    merged_df = merged_df.sort_values(['stock_id', 'date'])
    
    # 計算5天內的累積數據
    for i in range(5):
        day = i + 1
        # 累積買進量
        merged_df[f'f_buy_sum_{day}d'] = merged_df.groupby('stock_id')['buy'].rolling(
            window=day, min_periods=1
        ).sum().reset_index(0, drop=True)
        
        # 累積賣出量
        merged_df[f'f_sell_sum_{day}d'] = merged_df.groupby('stock_id')['sell'].rolling(
            window=day, min_periods=1
        ).sum().reset_index(0, drop=True)
        
        # 累積成交量
        merged_df[f't_volume_sum_{day}d'] = merged_df.groupby('stock_id')['t_volume'].rolling(
            window=day, min_periods=1
        ).sum().reset_index(0, drop=True)
        
        # 計算比例
        merged_df[f'f_buy_ratio_{day}d'] = merged_df[f'f_buy_sum_{day}d'] / merged_df[f't_volume_sum_{day}d']
        merged_df[f'f_sell_ratio_{day}d'] = merged_df[f'f_sell_sum_{day}d'] / merged_df[f't_volume_sum_{day}d']
        merged_df[f'f_net_ratio_{day}d'] = (
            merged_df[f'f_buy_sum_{day}d'] - merged_df[f'f_sell_sum_{day}d']
        ) / merged_df[f't_volume_sum_{day}d']
    
    # 計算外資買賣行為的變化率和加速度
    merged_df['f_buy_ratio_change'] = merged_df['f_buy_ratio'] - merged_df.groupby('stock_id')['f_buy_ratio'].shift(1)
    merged_df['f_sell_ratio_change'] = merged_df['f_sell_ratio'] - merged_df.groupby('stock_id')['f_sell_ratio'].shift(1)
    merged_df['f_net_ratio_change'] = merged_df['f_net_ratio'] - merged_df.groupby('stock_id')['f_net_ratio'].shift(1)
    
    # 計算加速度 (二階導數)
    merged_df['f_buy_ratio_accel'] = merged_df['f_buy_ratio_change'] - merged_df.groupby('stock_id')['f_buy_ratio_change'].shift(1)
    merged_df['f_sell_ratio_accel'] = merged_df['f_sell_ratio_change'] - merged_df.groupby('stock_id')['f_sell_ratio_change'].shift(1)
    merged_df['f_net_ratio_accel'] = merged_df['f_net_ratio_change'] - merged_df.groupby('stock_id')['f_net_ratio_change'].shift(1)
    
    # 外資買賣壓力比 (買進比例/賣出比例)
    merged_df['f_pressure_ratio'] = merged_df['f_buy_ratio'] / merged_df['f_sell_ratio']
    merged_df['f_pressure_ratio'] = merged_df['f_pressure_ratio'].replace([np.inf, -np.inf], np.nan).fillna(1.0)
    
    # 外資買賣不平衡度 (買賣差異的絕對值)
    merged_df['f_imbalance'] = np.abs(merged_df['f_buy_ratio'] - merged_df['f_sell_ratio'])
    
    # 選擇要保留的特徵列
    feature_columns = [
        'date', 
        'stock_id',
        'is_analysis_period',  # 保留這個欄位以便後續過濾
    ]
    
    # 加入所有以f_開頭的特徵
    f_columns = [col for col in merged_df.columns if col.startswith('f_')]
    feature_columns.extend(f_columns)
    
    # 選擇特徵並填充缺失值
    result_df = merged_df[feature_columns].copy()
    result_df = result_df.fillna(0)
    
    # 儲存到快取
    CacheManager.save(result_df, 'features', 'foreign_features', filename)
    print(f"儲存外資特徵快取完成")
    
    return result_df
# 特徵相關性分析
def analyze_feature_correlations(df: pd.DataFrame, threshold: float = 0.7) -> pd.DataFrame:
    """
    分析特徵間的相關性，找出高度相關的特徵
    
    Parameters:
    -----------
    df : pd.DataFrame
        包含特徵的DataFrame
    threshold : float, default=0.7
        相關性閾值，高於此值的特徵對將被識別
        
    Returns:
    --------
    pd.DataFrame
        包含高相關特徵對的DataFrame
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # 選擇數值型特徵
    numeric_features = df.select_dtypes(include=[np.number]).columns
    
    # 排除非特徵列
    exclude_cols = ['label', 'price_range_ratio', 'next_day_high', 'next_day_open', 'profit', 'profit_volume']
    feature_cols = [col for col in numeric_features if col not in exclude_cols]
    
    # 計算相關矩陣
    corr_matrix = df[feature_cols].corr()
    
    # 找出高度相關的特徵對
    high_correlations = []
    
    for i in range(len(feature_cols)):
        for j in range(i+1, len(feature_cols)):
            feature1 = feature_cols[i]
            feature2 = feature_cols[j]
            correlation = corr_matrix.iloc[i, j]
            
            if abs(correlation) >= threshold:
                # 獲取特徵類型前綴
                prefix1 = feature1.split('_')[0]
                prefix2 = feature2.split('_')[0]
                
                high_correlations.append({
                    'feature1': feature1,
                    'feature2': feature2,
                    'correlation': correlation,
                    'type1': prefix1,
                    'type2': prefix2,
                    'same_type': prefix1 == prefix2
                })
    
    # 創建高相關特徵對的DataFrame
    if high_correlations:
        high_corr_df = pd.DataFrame(high_correlations)
        high_corr_df = high_corr_df.sort_values('correlation', ascending=False)
        
        # 保存高相關特徵對
        high_corr_df.to_csv('./analysis_results/high_correlation_features.csv', index=False)
        
        # 繪製熱圖
        plt.figure(figsize=(16, 14))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, cmap='coolwarm', center=0,
                   square=True, linewidths=.5, cbar_kws={"shrink": .5})
        plt.title('特徵相關性熱圖')
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig('./analysis_results/feature_correlation_heatmap.png', bbox_inches='tight', dpi=300)
        plt.close()
        
        # 打印相關性統計
        print(f"\n發現 {len(high_corr_df)} 對高相關特徵 (相關係數 >= {threshold})")
        
        # 按類型分組計算相關特徵對數量
        same_type = high_corr_df[high_corr_df['same_type']].shape[0]
        diff_type = high_corr_df[~high_corr_df['same_type']].shape[0]
        
        print(f"同類型特徵間的高相關對: {same_type}")
        print(f"不同類型特徵間的高相關對: {diff_type}")
        
        # 列出不同類型間最高相關的前10個特徵對
        if diff_type > 0:
            print("\n不同類型特徵間相關性最高的前10對:")
            for _, row in high_corr_df[~high_corr_df['same_type']].head(10).iterrows():
                print(f"{row['feature1']} 與 {row['feature2']}: {row['correlation']:.2f}")
        
        return high_corr_df
    else:
        print(f"\n未發現相關係數高於 {threshold} 的特徵對")
        return pd.DataFrame()

def create_composite_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    創建不同類型特徵之間的複合特徵，支持舊版和新版指數特徵
    不使用快取，保證計算的靈活性
    
    Parameters:
    -----------
    df : pd.DataFrame
        包含價格(p_)、成交量(vol_)、融資融券(m_,s_)、外資(f_)和指數(rel_, market_)特徵的DataFrame
        
    Returns:
    --------
    pd.DataFrame
        添加了複合特徵的DataFrame
    """
    import pandas as pd
    import numpy as np
    import time
    
    print("計算複合特徵...")
    start_time = time.time()
    df = df.copy()
    
    # 確保df已按date排序
    df = df.sort_values(['date'])
    
    # 1. 外資與價格的相關特徵
    # 外資淨買超與價格變化的相對關係
    if 'f_net_ratio' in df.columns and 'p_close_change' in df.columns:
        df['comp_f_price_alignment'] = df['f_net_ratio'] * df['p_close_change']  # 正值表示外資行為與價格變化一致
        df['comp_f_price_divergence'] = df['f_net_ratio'] - df['p_close_change']  # 正值表示外資買超但價格漲幅較小，可能有上漲空間
    
    # 2. 融資融券與價格的相關特徵
    # 融資動能與價格漲跌的關係
    if 'm_momentum' in df.columns and 'p_close_change' in df.columns:
        df['comp_m_price_alignment'] = df['m_momentum'] * df['p_close_change']
        df['comp_m_price_divergence'] = df['m_momentum'] - df['p_close_change']
    
    # 3. 外資與融資融券的相關特徵
    # 外資與融資行為的一致性
    if 'f_net_ratio' in df.columns and 'm_momentum' in df.columns:
        df['comp_f_m_alignment'] = df['f_net_ratio'] * df['m_momentum']
        # 外資與融資行為的背離程度
        df['comp_f_m_divergence'] = np.abs(df['f_net_ratio'] - df['m_momentum'])
    
    # 4. 成交量與價格的相關特徵
    # 價量配合度
    if 'vol_change' in df.columns and 'p_close_change' in df.columns:
        df['comp_vol_price_ratio'] = df['p_close_change'] / df['vol_change']
        df['comp_vol_price_ratio'] = df['comp_vol_price_ratio'].replace([np.inf, -np.inf], np.nan).fillna(0)
        
        # 價量同向性 (正值表示價量同向，負值表示價量背離)
        df['comp_vol_price_alignment'] = df['p_close_change'] * df['vol_change']
        
        # 價格漲幅/成交量比
        if 'vol_vs_5ma' in df.columns:
            df['comp_price_per_vol'] = df['p_close_change'] / (df['vol_vs_5ma'] + 0.001)
    
    # 5. 多指標綜合動能
    # 將主要動能指標合併為綜合指標
    momentum_columns = []
    for col in ['p_close_change', 'f_net_ratio', 'm_momentum', 'vol_change']:
        if col in df.columns:
            momentum_columns.append(col)
    
    if len(momentum_columns) >= 2:
        # 對各指標進行加權平均
        df['comp_overall_momentum'] = df[momentum_columns].mean(axis=1)
        
        # 計算各指標間的一致性 (標準差的負值，越高表示越一致)
        df['comp_indicator_agreement'] = -df[momentum_columns].std(axis=1)
    
    # 6. 時間序列特性
    # 創建一個滯後版本的綜合動能指標，並計算變化
    if 'comp_overall_momentum' in df.columns:
        df['comp_momentum_prev'] = df.groupby('stock_id')['comp_overall_momentum'].shift(1)
        df['comp_momentum_change'] = df['comp_overall_momentum'] - df['comp_momentum_prev']
        
        # 動能反轉指標 (當前動能與前期動能的乘積，負值表示反轉)
        df['comp_momentum_reversal'] = df['comp_overall_momentum'] * df['comp_momentum_prev']
    
    # 7. 市場趨勢相關複合特徵 - 支持舊版和新版指數特徵
    
    # 7.1 为兼容性添加舊版市場特徵 (如果需要且不存在)
    if ('comp_market_relative' not in df.columns or 'comp_market_alignment' not in df.columns) and 'p_close_change' in df.columns:
        # 計算每日所有股票的平均收盤價變化作為市場趨勢
        market_trend = df.groupby('date')['p_close_change'].mean().reset_index()
        market_trend.rename(columns={'p_close_change': 'market_avg_change'}, inplace=True)
        
        # 合併市場趨勢
        df = pd.merge(df, market_trend, on='date', how='left')
        
        # 計算個股與市場的相對表現
        df['comp_market_relative'] = df['p_close_change'] - df['market_avg_change']
        
        # 計算與市場趨勢的一致性
        df['comp_market_alignment'] = df['p_close_change'] * df['market_avg_change']
        
        # 如果不存在真正的市場同步率，模擬一個
        if 'market_sync_rate' not in df.columns and 'comp_sync_momentum' not in df.columns:
            df['comp_sync_momentum'] = df['market_avg_change'] * df['p_close_change']
        
        # 移除輔助列
        df = df.drop(columns=['market_avg_change'])
    
    # 7.2 使用新版指數特徵 (如果存在)
    
    # 相對強弱與價格變化的組合
    if 'rel_strength' in df.columns and 'p_close_change' in df.columns:
        # 相對強弱與價格變化的一致性
        df['comp_rel_strength_price'] = df['rel_strength'] * df['p_close_change']
        
        # 相對強弱過高但價格變化不大時的潛在信號
        df['comp_rel_strength_potential'] = df['rel_strength'] - df['p_close_change']
    
    # Beta與價格波動的組合
    if 'market_beta_30d' in df.columns and 'p_day_range' in df.columns:
        # Beta與日內波動的關係
        df['comp_beta_volatility'] = df['market_beta_30d'] * df['p_day_range']
        
        # 高Beta低波動股可能有爆發潛力
        df['comp_beta_potential'] = ((df['market_beta_30d'] > 1.2) & (df['p_day_range'] < 0.02)).astype(int)
    
    # 市場相關性與動能特徵
    if 'market_corr_20d' in df.columns and 'p_close_change' in df.columns:
        # 高相關性時的價格變動倍增效應
        df['comp_corr_momentum'] = df['market_corr_20d'] * df['p_close_change']
        
        # 相關性變化作為趨勢轉變的潛在信號
        if 'rel_strength_change' in df.columns:
            df['comp_corr_trend_change'] = df['market_corr_20d'] * df['rel_strength_change']
    
    # 市場同步率與價格動量
    if 'market_sync_rate' in df.columns and 'p_close_change' in df.columns:
        # 如果有真正的市場同步率，覆蓋之前模擬的版本
        df['comp_sync_momentum'] = df['market_sync_rate'] * df['p_close_change']
    
    # 如果缺少lead_lag_days，創建一個默認值版本
    if 'lead_lag_days' not in df.columns and 'p_close_change' in df.columns:
        df['lead_lag_days'] = np.sign(df.groupby('stock_id')['p_close_change'].diff())
    
    # 結合RSI特徵(如果存在)
    if 'midx_idx_rsi_14' in df.columns:
        # RSI超賣區間(低於30)中表現強勢的股票
        if 'rel_strength' in df.columns:
            df['comp_rsi_strength'] = np.where(
                df['midx_idx_rsi_14'] < 30,
                df['rel_strength'],
                0
            )
        
        # RSI超買區間(高於70)中表現弱勢的股票
        if 'rel_strength' in df.columns:
            df['comp_rsi_weakness'] = np.where(
                df['midx_idx_rsi_14'] > 70,
                -df['rel_strength'],
                0
            )
    
    # 外資/融資與相對強弱的組合
    if 'rel_strength' in df.columns:
        if 'f_net_ratio' in df.columns:
            df['comp_f_rel_strength'] = df['f_net_ratio'] * df['rel_strength']
        
        if 'm_momentum' in df.columns:
            df['comp_m_rel_strength'] = df['m_momentum'] * df['rel_strength']
    
    # 清理數據
    # 將無限值替換為 NaN，並填充缺失值
    comp_columns = [col for col in df.columns if col.startswith('comp_')]
    df[comp_columns] = df[comp_columns].replace([np.inf, -np.inf], np.nan).fillna(0)
    
    total_time = time.time() - start_time
    print(f"複合特徵計算完成，總耗時: {total_time:.2f}秒")
    print(f"新增複合特徵數量: {len(comp_columns)}個")
    
    return df

# 特徵分析與可視化函數
def analyze_feature_groups(df: pd.DataFrame, model) -> None:
    """
    分析各類特徵組的重要性
    
    Parameters:
    -----------
    df : pd.DataFrame
        包含特徵的DataFrame
    model : 訓練好的模型
        包含特徵重要性的模型
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # 獲取特徵重要性
    feature_importance = pd.DataFrame({
        'feature': df.columns[:-1],  # 排除標籤列
        'importance': [model.get_score().get(f, 0) for f in df.columns[:-1]]
    })
    
    # 正規化重要性分數
    feature_importance['importance_normalized'] = feature_importance['importance'] / feature_importance['importance'].sum() * 100
    
    # 為每個特徵分配前綴類型
    def get_feature_type(feature_name):
        if feature_name in ['date', 'stock_id', 'label', 'price_range_ratio', 'next_day_high', 
                           'next_day_open', 'profit', 'profit_volume', 'close']:
            return 'meta'
        
        prefix = feature_name.split('_')[0]
        
        mapping = {
            'p': '價格相關',
            'vol': '成交量相關',
            'm': '融資相關',
            's': '融券相關',
            'ms': '融資融券綜合',
            'f': '外資相關',
            'comp': '複合特徵'
        }
        
        return mapping.get(prefix, '其他')
    
    feature_importance['feature_type'] = feature_importance['feature'].apply(get_feature_type)
    
    # 按類型分組並計算重要性總和
    group_importance = feature_importance.groupby('feature_type')['importance_normalized'].sum().reset_index()
    group_importance = group_importance.sort_values('importance_normalized', ascending=False)
    
    # 繪製特徵組重要性柱狀圖
    plt.figure(figsize=(12, 6))
    sns.barplot(x='importance_normalized', y='feature_type', data=group_importance, palette='viridis')
    plt.title('各類特徵組重要性佔比')
    plt.xlabel('重要性百分比 (%)')
    plt.ylabel('特徵類型')
    
    # 在每個柱子上添加百分比值
    for i, row in enumerate(group_importance.itertuples()):
        plt.text(row.importance_normalized + 0.5, i, f'{row.importance_normalized:.2f}%', va='center')
    
    plt.tight_layout()
    plt.savefig('./analysis_results/feature_group_importance.png', bbox_inches='tight', dpi=300)
    plt.close()
    
    # 打印各組特徵重要性
    print("\n各類特徵組重要性:")
    for _, row in group_importance.iterrows():
        print(f"{row['feature_type']}: {row['importance_normalized']:.2f}%")
    
    # 針對每個組別，找出前5個最重要的特徵
    print("\n各組別中最重要的特徵:")
    for feature_type in group_importance['feature_type']:
        if feature_type == 'meta':
            continue
            
        top_features = feature_importance[
            feature_importance['feature_type'] == feature_type
        ].sort_values('importance_normalized', ascending=False).head(5)
        
        print(f"\n{feature_type}類型中最重要的前5個特徵:")
        for _, row in top_features.iterrows():
            print(f"  - {row['feature']}: {row['importance_normalized']:.2f}%")
def calculate_historical_features(df: pd.DataFrame, window_size: int = 5) -> pd.DataFrame:
    """
    計算歷史特徵
    """
    df = df.copy()
    
    # 對每個股票分別計算歷史特徵
    for i in range(1, window_size + 1):
        # 計算前N天的變化率
        df[f'p_open_change_d{i}'] = df.groupby('stock_id')['p_open_change'].shift(i)
        df[f'p_high_change_d{i}'] = df.groupby('stock_id')['p_high_change'].shift(i)
        df[f'p_low_change_d{i}'] = df.groupby('stock_id')['p_low_change'].shift(i)
        df[f'p_close_change_d{i}'] = df.groupby('stock_id')['p_close_change'].shift(i)
        df[f'vol_change_d{i}'] = df.groupby('stock_id')['vol_change'].shift(i)
        
        # 計算N天累積漲跌幅
        df[f'p_cumulative_return_d{i}'] = df.groupby('stock_id')['p_close_change'].rolling(window=i).sum().reset_index(0, drop=True)
        
        # 計算N天累積成交量變化
        df[f'vol_cumulative_volume_d{i}'] = df.groupby('stock_id')['vol_change'].rolling(window=i).sum().reset_index(0, drop=True)
    
    return df

# 解析命令列參數
def parse_arguments():
    """
    解析命令列參數以控制特徵使用
    
    Returns:
    --------
    argparse.Namespace
        解析後的參數
    """
    parser = argparse.ArgumentParser(description='股票當沖策略訓練與測試')
    
    # 資料時間區間參數
    parser.add_argument('--train_start_date', type=str, default='2025-01-01', help='訓練開始日期 (YYYY-MM-DD)')
    parser.add_argument('--train_end_date', type=str, default='2025-02-28', help='訓練結束日期 (YYYY-MM-DD)')
    parser.add_argument('--test_start_date', type=str, default='2025-03-01', help='測試開始日期 (YYYY-MM-DD)')
    parser.add_argument('--test_end_date', type=str, default='2025-03-14', help='測試結束日期 (YYYY-MM-DD)')
    
    # 特徵選擇參數
    parser.add_argument('--use_price', action='store_true', default=True, help='使用價格相關特徵')
    parser.add_argument('--use_volume', action='store_true', default=True, help='使用成交量相關特徵')
    parser.add_argument('--use_broker', action='store_true', help='使用籌碼分點特徵')
    parser.add_argument('--use_foreign', action='store_true', help='使用外資特徵')
    parser.add_argument('--use_margin', action='store_true', help='使用融資融券特徵')
    parser.add_argument('--use_composite', action='store_true', default=True, help='使用複合特徵')
    
    # 模型與訓練參數
    parser.add_argument('--model_type', type=str, default='xgboost', choices=['xgboost', 'lightgbm', 'tabnet'], help='模型類型')
    parser.add_argument('--lookback_days', type=int, default=30, help='回溯天數')
    parser.add_argument('--capital_per_trade', type=int, default=1000, help='每筆交易金額')
    parser.add_argument('--threshold', type=float, default=0.6, help='預測閾值')
    parser.add_argument('--feature_selection_threshold', type=float, default=0.5, help='特徵選擇閾值 (0-1之間)')
    parser.add_argument('--output_dir', type=str, default='./analysis_results', help='結果輸出目錄')
    
    return parser.parse_args()

# 產生特徵識別字串
def generate_feature_identifier(args):
    """
    根據啟用的特徵產生識別字串
    
    Parameters:
    -----------
    args : argparse.Namespace
        命令列參數
        
    Returns:
    --------
    str
        特徵識別字串
    """
    features = []
    
    if args.use_price:
        features.append("price")
    
    if args.use_volume:
        features.append("volume")
    
    if args.use_broker:
        features.append("broker")
    
    if args.use_foreign:
        features.append("foreign")
    
    if args.use_margin:
        features.append("margin")
    
    if not features:
        return "basic"
    
    return "_".join(features)

def read_index_data_cached(twse_file_path='twse_index_data.csv', tpex_file_path='tpex_index_data.csv'):
    """
    讀取上市加權指數和上櫃櫃買指數資料，加入快取機制
    
    Parameters:
    -----------
    twse_file_path : str
        加權指數CSV檔案路徑
    tpex_file_path : str
        櫃買指數CSV檔案路徑
        
    Returns:
    --------
    tuple
        (twse_df, tpex_df) - 加權指數和櫃買指數的DataFrame
    """
    import pandas as pd
    import os
    
    # 建立快取檔案名稱
    twse_cache_file = f"D:/data_cache/index_data/twse_index_processed.pkl"
    tpex_cache_file = f"D:/data_cache/index_data/tpex_index_processed.pkl"
    
    # 確保快取目錄存在
    os.makedirs("D:/data_cache/index_data", exist_ok=True)
    
    # 檢查加權指數快取
    if os.path.exists(twse_cache_file):
        print(f"從快取讀取加權指數資料: {twse_cache_file}")
        try:
            twse_df = pd.read_pickle(twse_cache_file)
        except Exception as e:
            print(f"讀取加權指數快取失敗: {e}，重新處理資料")
            twse_df = None
    else:
        twse_df = None
    
    # 如果快取不存在或讀取失敗，從原始檔案讀取並處理
    if twse_df is None:
        try:
            twse_df = pd.read_csv(twse_file_path)
            # 確保欄位名稱一致
            twse_df.columns = ['date', 'open', 'high', 'low', 'close']
            # 確保日期格式一致
            twse_df['date'] = pd.to_datetime(twse_df['date'])
            # 排序資料
            twse_df = twse_df.sort_values('date')
            # 儲存處理後的資料到快取
            twse_df.to_pickle(twse_cache_file)
            print(f"加權指數資料已處理並儲存到快取: {twse_cache_file}")
        except Exception as e:
            print(f"讀取加權指數資料失敗: {e}")
            twse_df = pd.DataFrame(columns=['date', 'open', 'high', 'low', 'close'])
    
    # 檢查櫃買指數快取
    if os.path.exists(tpex_cache_file):
        print(f"從快取讀取櫃買指數資料: {tpex_cache_file}")
        try:
            tpex_df = pd.read_pickle(tpex_cache_file)
        except Exception as e:
            print(f"讀取櫃買指數快取失敗: {e}，重新處理資料")
            tpex_df = None
    else:
        tpex_df = None
    
    # 如果快取不存在或讀取失敗，從原始檔案讀取並處理
    if tpex_df is None:
        try:
            tpex_df = pd.read_csv(tpex_file_path)
            # 確保欄位名稱一致
            tpex_df.columns = ['date', 'open', 'high', 'low', 'close']
            # 確保日期格式一致
            tpex_df['date'] = pd.to_datetime(tpex_df['date'])
            # 排序資料
            tpex_df = tpex_df.sort_values('date')
            # 儲存處理後的資料到快取
            tpex_df.to_pickle(tpex_cache_file)
            print(f"櫃買指數資料已處理並儲存到快取: {tpex_cache_file}")
        except Exception as e:
            print(f"讀取櫃買指數資料失敗: {e}")
            tpex_df = pd.DataFrame(columns=['date', 'open', 'high', 'low', 'close'])
    
    return twse_df, tpex_df
def calculate_index_features_optimized(index_df):
    """
    優化計算指數相關的基本特徵，移除快取功能
    
    Parameters:
    -----------
    index_df : pd.DataFrame
        包含指數日資料的DataFrame，需要有date, open, high, low, close欄位
        
    Returns:
    --------
    pd.DataFrame
        包含指數特徵的DataFrame
    """
    import pandas as pd
    import numpy as np
    import time
    
    start_time = time.time()
    print("計算指數基本特徵...")
    
    # 複製資料避免修改原始資料
    df = index_df.copy()
    df = df.sort_values('date')
    
    # 計算基本特徵
    # 1. 前一日收盤價
    df['prev_close'] = df['close'].shift(1)
    
    # 2. 價格變化
    df['idx_open_change'] = (df['open'] - df['prev_close']) / df['prev_close']
    df['idx_high_change'] = (df['high'] - df['prev_close']) / df['prev_close']
    df['idx_low_change'] = (df['low'] - df['prev_close']) / df['prev_close']
    df['idx_close_change'] = (df['close'] - df['prev_close']) / df['prev_close']
    
    # 3. 日內波動
    df['idx_day_range'] = (df['high'] - df['low']) / df['prev_close']
    df['idx_up_shadow'] = (df['high'] - df[['open', 'close']].max(axis=1)) / df['prev_close']
    df['idx_down_shadow'] = (df[['open', 'close']].min(axis=1) - df['low']) / df['prev_close']
    df['idx_body'] = abs(df['close'] - df['open']) / df['prev_close']
    
    # 4. 移動平均線 - 使用向量化操作
    for window in [5, 10, 20, 60]:
        df[f'idx_ma{window}'] = df['close'].rolling(window=window).mean()
        df[f'idx_close_ma{window}_ratio'] = df['close'] / df[f'idx_ma{window}']
    
    # 5. 動量特徵 - 向量化操作
    for period in [1, 3, 5, 10, 20]:
        # 過去N天的漲跌幅
        df[f'idx_return_{period}d'] = df['close'].pct_change(periods=period)
        
        # 過去N天的波動率
        df[f'idx_volatility_{period}d'] = df['idx_close_change'].rolling(window=period).std()
    
    # 6. MACD相關指標 (12, 26, 9) - 向量化操作
    df['idx_ema12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['idx_ema26'] = df['close'].ewm(span=26, adjust=False).mean()
    df['idx_macd'] = df['idx_ema12'] - df['idx_ema26']
    df['idx_macd_signal'] = df['idx_macd'].ewm(span=9, adjust=False).mean()
    df['idx_macd_histogram'] = df['idx_macd'] - df['idx_macd_signal']
    
    # 7. RSI指標 (14天) - 向量化操作
    delta = df['close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    # 避免除以零
    avg_loss = avg_loss.replace(0, 1e-10)
    rs = avg_gain / avg_loss
    df['idx_rsi_14'] = 100 - (100 / (1 + rs))
    
    # 8. 趨勢特徵
    df['idx_trend_5d'] = np.sign(df['idx_ma5'].diff())
    df['idx_trend_20d'] = np.sign(df['idx_ma20'].diff())
    
    # 9. 模式識別特徵 - 向量化操作
    # 上漲三連陽
    df['idx_three_up'] = ((df['idx_close_change'] > 0) & 
                         (df['idx_close_change'].shift(1) > 0) & 
                         (df['idx_close_change'].shift(2) > 0)).astype(int)
    
    # 下跌三連陰
    df['idx_three_down'] = ((df['idx_close_change'] < 0) & 
                           (df['idx_close_change'].shift(1) < 0) & 
                           (df['idx_close_change'].shift(2) < 0)).astype(int)
    
    # 大盤刷新高點
    df['idx_new_high_20d'] = (df['high'] > df['high'].rolling(20).max().shift(1)).astype(int)
    df['idx_new_low_20d'] = (df['low'] < df['low'].rolling(20).min().shift(1)).astype(int)
    
    # 移除無限值和NaN
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(0)
    
    print(f"指數基本特徵計算完成，耗時: {time.time() - start_time:.2f}秒")
    
    return df

def calculate_stock_index_features_optimized(stock_df, twse_index_df, tpex_index_df, stock_info_df=None):
    """
    優化計算股票相對於指數的特徵
    
    Parameters:
    -----------
    stock_df : pd.DataFrame
        股票日交易資料，需要有stock_id, date, open, high, low, close等欄位
    twse_index_df : pd.DataFrame
        加權指數資料，需要有上述calculate_index_features計算的特徵
    tpex_index_df : pd.DataFrame
        櫃買指數資料，需要有上述calculate_index_features計算的特徵
    stock_info_df : pd.DataFrame, optional
        股票基本資料，包含股票類型(上市/上櫃)資訊
        
    Returns:
    --------
    pd.DataFrame
        包含股票相對指數特徵的DataFrame
    """
    import pandas as pd
    import numpy as np
    from tqdm import tqdm
    import warnings
    warnings.filterwarnings('ignore')
    import time
    
    print("開始計算股票相對於指數的特徵...")
    start_time = time.time()
    
    # 複製資料避免修改原始資料
    df = stock_df.copy()
    df['date'] = pd.to_datetime(df['date'])
    
    # 確保所有日期都使用datetime格式
    twse_index_df['date'] = pd.to_datetime(twse_index_df['date'])
    tpex_index_df['date'] = pd.to_datetime(tpex_index_df['date'])
    
    # 對指數資料進行預處理，移除多餘欄位，只保留需要的
    twse_cols = ['date', 'idx_close_change', 'idx_trend_5d', 'idx_trend_20d', 
                 'idx_three_up', 'idx_three_down', 'idx_new_high_20d', 
                 'idx_new_low_20d', 'idx_rsi_14', 'idx_macd_histogram', 
                 'idx_volatility_20d']
    tpex_cols = twse_cols.copy()
    
    # 確保所有需要的欄位都存在
    for col in twse_cols[1:]:
        if col not in twse_index_df.columns:
            twse_index_df[col] = 0
    
    for col in tpex_cols[1:]:
        if col not in tpex_index_df.columns:
            tpex_index_df[col] = 0
    
    # 只保留需要的欄位以減少記憶體使用
    twse_index_df = twse_index_df[twse_cols]
    tpex_index_df = tpex_index_df[tpex_cols]
    
    # 如果有提供股票基本資料，判斷每支股票是上市還是上櫃
    if stock_info_df is not None and 'type' in stock_info_df.columns:
        # 合併股票類型資訊
        if 'type' not in df.columns:
            stock_info_subset = stock_info_df[['stock_id', 'type']].copy()
            stock_info_subset['stock_id'] = stock_info_subset['stock_id'].astype(str)
            df['stock_id'] = df['stock_id'].astype(str)
            df = pd.merge(df, stock_info_subset, on='stock_id', how='left')
    
    # 默認所有股票使用加權指數
    df['market_type'] = 'twse'
    
    # 如果有提供股票類型資訊，更新市場類型
    if 'type' in df.columns:
        # 將上市股票標記為 'twse'
        df.loc[df['type'] == 'twse', 'market_type'] = 'twse'
        # 將上櫃股票標記為 'tpex'
        df.loc[df['type'] == 'tpex', 'market_type'] = 'tpex'
    
    # 準備結果 DataFrame
    result_df = df[['date', 'stock_id']].copy()
    
    if 'is_analysis_period' in df.columns:
        result_df['is_analysis_period'] = df['is_analysis_period']
    
    # 將股票資料分成上市和上櫃兩組
    twse_stocks = df[df['market_type'] == 'twse'].copy()
    tpex_stocks = df[df['market_type'] == 'tpex'].copy()
    
    # 定義要處理的關鍵欄位
    stock_change_col = 'p_close_change' if 'p_close_change' in df.columns else None
    if stock_change_col is None:
        # 手動計算股票漲跌幅
        if 'close' in df.columns and 'prev_close' in df.columns:
            df['stock_change'] = (df['close'] - df['prev_close']) / (df['prev_close'] + 1e-10)
            stock_change_col = 'stock_change'
            twse_stocks['stock_change'] = (twse_stocks['close'] - twse_stocks['prev_close']) / (twse_stocks['prev_close'] + 1e-10)
            tpex_stocks['stock_change'] = (tpex_stocks['close'] - tpex_stocks['prev_close']) / (tpex_stocks['prev_close'] + 1e-10)
        else:
            # 如果無法計算，創建一個空的列
            df['stock_change'] = 0
            twse_stocks['stock_change'] = 0
            tpex_stocks['stock_change'] = 0
            stock_change_col = 'stock_change'
    
    def calculate_beta_with_duplicate_safe(df, stock_id_col, stock_change_col, market_change_col, window_size=30):
        """
        計算 beta 值的安全方法，處理重複索引問題
        """
        # 初始化結果列表
        results = []
        
        # 以股票為單位分組計算
        for stock_id, group in tqdm(df.groupby(stock_id_col), desc="計算 Beta 值"):
            # 按日期排序
            group = group.sort_values('date')
            if len(group) < window_size:
                continue
            
            # 保存原始索引、日期以便後續合併
            indices = group.index.tolist()
            dates = group['date'].tolist()
            
            # 計算 beta 值
            betas = [0] * (window_size - 1)  # 前 window_size-1 天填充為 0
            
            for i in range(len(group) - window_size + 1):
                window_data = group.iloc[i:i+window_size]
                stock_returns = window_data[stock_change_col].values
                market_returns = window_data[market_change_col].values
                
                # 計算 beta
                if np.std(market_returns) == 0:
                    betas.append(0)
                else:
                    cov = np.cov(stock_returns, market_returns)[0, 1]
                    var = np.var(market_returns)
                    betas.append(cov / var)
            
            # 創建結果 DataFrame
            for idx, date, beta in zip(indices, dates, betas):
                results.append({
                    'stock_id': stock_id,
                    'date': date,
                    'index': idx,
                    'beta_value': beta
                })
        
        # 返回結果 DataFrame
        return pd.DataFrame(results)
    
    def calculate_correlation_with_duplicate_safe(df, stock_id_col, stock_change_col, market_change_col, window_size=20):
        """
        計算相關性的安全方法，處理重複索引問題
        """
        # 初始化結果列表
        results = []
        
        # 以股票為單位分組計算
        for stock_id, group in tqdm(df.groupby(stock_id_col), desc="計算相關性"):
            # 按日期排序
            group = group.sort_values('date')
            if len(group) < window_size:
                continue
            
            # 保存原始索引、日期以便後續合併
            indices = group.index.tolist()
            dates = group['date'].tolist()
            
            # 計算相關性
            corrs = [0] * (window_size - 1)  # 前 window_size-1 天填充為 0
            
            for i in range(len(group) - window_size + 1):
                window_data = group.iloc[i:i+window_size]
                stock_returns = window_data[stock_change_col].values
                market_returns = window_data[market_change_col].values
                
                # 計算相關性
                if np.std(stock_returns) == 0 or np.std(market_returns) == 0:
                    corrs.append(0)
                else:
                    corr = np.corrcoef(stock_returns, market_returns)[0, 1]
                    corrs.append(corr)
            
            # 創建結果 DataFrame
            for idx, date, corr in zip(indices, dates, corrs):
                results.append({
                    'stock_id': stock_id,
                    'date': date,
                    'index': idx,
                    'corr_value': corr
                })
        
        # 返回結果 DataFrame
        return pd.DataFrame(results)
    
    # 處理上市股票
    if not twse_stocks.empty:
        print("處理上市股票相對於加權指數的特徵...")
        # 合併加權指數資料
        twse_merged = pd.merge(twse_stocks, twse_index_df, on='date', how='left')
        
        # 計算相對強弱指標
        twse_merged['rel_strength'] = twse_merged[stock_change_col] - twse_merged['idx_close_change']
        
        # 使用向量化操作計算移動平均
        for window in [5, 10, 20]:
            twse_merged[f'rel_strength_{window}d_ma'] = twse_merged.groupby('stock_id')['rel_strength'].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean()
            )
        
        # 強弱反轉指標
        twse_merged['rel_strength_change'] = twse_merged.groupby('stock_id')['rel_strength'].diff()
        
        # 累積相對強弱
        for period in [3, 5, 10]:
            twse_merged[f'rel_strength_cum_{period}d'] = twse_merged.groupby('stock_id')['rel_strength'].transform(
                lambda x: x.rolling(window=period, min_periods=1).sum()
            )
        
        # 初始化 Beta 欄位
        twse_merged['market_beta_30d'] = 0.0
        
        # 使用安全的 Beta 計算方法
        print("計算上市股票 Beta 值...")
        beta_start = time.time()
        beta_results_df = calculate_beta_with_duplicate_safe(
            twse_merged,
            'stock_id',
            stock_change_col,
            'idx_close_change',
            window_size=30
        )
        
        # 使用直接更新方式，避免 DataFrame.loc 索引問題
        if not beta_results_df.empty:
            # 創建索引映射，用於更新 twse_merged 中的 market_beta_30d 列
            index_to_beta = dict(zip(beta_results_df['index'], beta_results_df['beta_value']))
            
            # 使用 apply 更新，避免直接索引賦值
            twse_merged['market_beta_30d'] = twse_merged.index.map(lambda x: index_to_beta.get(x, 0))
            
            print(f"Beta 值計算完成，耗時: {time.time() - beta_start:.2f}秒")
        
        # 初始化相關性欄位
        twse_merged['market_corr_20d'] = 0.0
        
        # 使用安全的相關性計算方法
        print("計算上市股票與大盤的相關性...")
        corr_start = time.time()
        corr_results_df = calculate_correlation_with_duplicate_safe(
            twse_merged,
            'stock_id',
            stock_change_col,
            'idx_close_change',
            window_size=20
        )
        
        # 使用直接更新方式，避免 DataFrame.loc 索引問題
        if not corr_results_df.empty:
            # 創建索引映射，用於更新 twse_merged 中的 market_corr_20d 列
            index_to_corr = dict(zip(corr_results_df['index'], corr_results_df['corr_value']))
            
            # 使用 apply 更新，避免直接索引賦值
            twse_merged['market_corr_20d'] = twse_merged.index.map(lambda x: index_to_corr.get(x, 0))
            
            print(f"相關性計算完成，耗時: {time.time() - corr_start:.2f}秒")
        
        # 大盤趨勢特徵與個股表現結合
        twse_merged['rel_strength_in_uptrend'] = np.where(
            twse_merged['idx_close_change'] > 0, 
            twse_merged['rel_strength'], 
            0
        )
        
        twse_merged['rel_strength_in_downtrend'] = np.where(
            twse_merged['idx_close_change'] < 0, 
            twse_merged['rel_strength'], 
            0
        )
        
        # 複合指標
        twse_merged['trend_beta_signal'] = twse_merged['idx_trend_5d'] * twse_merged['market_beta_30d']
        
        # RSI 相關特徵
        if 'idx_rsi_14' in twse_merged.columns:
            twse_merged['rel_strength_high_rsi'] = np.where(
                twse_merged['idx_rsi_14'] > 70,
                twse_merged['rel_strength'],
                0
            )
            
            twse_merged['rel_strength_low_rsi'] = np.where(
                twse_merged['idx_rsi_14'] < 30,
                twse_merged['rel_strength'],
                0
            )
        
        # 創建一個臨時的結果 DataFrame，包含上市股票相對於大盤的所有特徵
        twse_result = twse_merged[['date', 'stock_id', 'rel_strength', 'rel_strength_5d_ma',
                                  'rel_strength_10d_ma', 'rel_strength_20d_ma', 'rel_strength_change',
                                  'rel_strength_cum_3d', 'rel_strength_cum_5d', 'rel_strength_cum_10d',
                                  'market_beta_30d', 'market_corr_20d', 'rel_strength_in_uptrend',
                                  'rel_strength_in_downtrend', 'trend_beta_signal']]
        
        # 添加 RSI 相關特徵
        if 'rel_strength_high_rsi' in twse_merged.columns:
            twse_result['rel_strength_high_rsi'] = twse_merged['rel_strength_high_rsi']
            twse_result['rel_strength_low_rsi'] = twse_merged['rel_strength_low_rsi']
        
        # 添加大盤指標
        for col in ['idx_trend_5d', 'idx_trend_20d', 'idx_three_up', 'idx_three_down',
                    'idx_new_high_20d', 'idx_new_low_20d', 'idx_rsi_14',
                    'idx_macd_histogram', 'idx_volatility_20d']:
            if col in twse_merged.columns:
                twse_result[f'midx_{col}'] = twse_merged[col]
    
    # 處理上櫃股票 - 使用並行處理提高速度
    if not tpex_stocks.empty:
        print("處理上櫃股票相對於櫃買指數的特徵...")
        # 合併櫃買指數資料
        tpex_merged = pd.merge(tpex_stocks, tpex_index_df, on='date', how='left')
        
        # 計算相對強弱指標
        tpex_merged['rel_strength'] = tpex_merged[stock_change_col] - tpex_merged['idx_close_change']
        
        # 使用向量化操作計算移動平均
        for window in [5, 10, 20]:
            tpex_merged[f'rel_strength_{window}d_ma'] = tpex_merged.groupby('stock_id')['rel_strength'].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean()
            )
        
        # 強弱反轉指標
        tpex_merged['rel_strength_change'] = tpex_merged.groupby('stock_id')['rel_strength'].diff()
        
        # 累積相對強弱
        for period in [3, 5, 10]:
            tpex_merged[f'rel_strength_cum_{period}d'] = tpex_merged.groupby('stock_id')['rel_strength'].transform(
                lambda x: x.rolling(window=period, min_periods=1).sum()
            )
        
        # 初始化 Beta 欄位
        tpex_merged['market_beta_30d'] = 0.0
        
        # 使用安全的 Beta 計算方法
        print("計算上櫃股票 Beta 值...")
        beta_start = time.time()
        beta_results_df = calculate_beta_with_duplicate_safe(
            tpex_merged,
            'stock_id',
            stock_change_col,
            'idx_close_change',
            window_size=30
        )
        
        # 使用直接更新方式，避免 DataFrame.loc 索引問題
        if not beta_results_df.empty:
            # 創建索引映射，用於更新 tpex_merged 中的 market_beta_30d 列
            index_to_beta = dict(zip(beta_results_df['index'], beta_results_df['beta_value']))
            
            # 使用 apply 更新，避免直接索引賦值
            tpex_merged['market_beta_30d'] = tpex_merged.index.map(lambda x: index_to_beta.get(x, 0))
            
            print(f"Beta 值計算完成，耗時: {time.time() - beta_start:.2f}秒")
        
        # 初始化相關性欄位
        tpex_merged['market_corr_20d'] = 0.0
        
        # 使用安全的相關性計算方法
        print("計算上櫃股票與大盤的相關性...")
        corr_start = time.time()
        corr_results_df = calculate_correlation_with_duplicate_safe(
            tpex_merged,
            'stock_id',
            stock_change_col,
            'idx_close_change',
            window_size=20
        )
        
        # 使用直接更新方式，避免 DataFrame.loc 索引問題
        if not corr_results_df.empty:
            # 創建索引映射，用於更新 tpex_merged 中的 market_corr_20d 列
            index_to_corr = dict(zip(corr_results_df['index'], corr_results_df['corr_value']))
            
            # 使用 apply 更新，避免直接索引賦值
            tpex_merged['market_corr_20d'] = tpex_merged.index.map(lambda x: index_to_corr.get(x, 0))
            
            print(f"相關性計算完成，耗時: {time.time() - corr_start:.2f}秒")
        
        # 大盤趨勢特徵與個股表現結合
        tpex_merged['rel_strength_in_uptrend'] = np.where(
            tpex_merged['idx_close_change'] > 0, 
            tpex_merged['rel_strength'], 
            0
        )
        
        tpex_merged['rel_strength_in_downtrend'] = np.where(
            tpex_merged['idx_close_change'] < 0, 
            tpex_merged['rel_strength'], 
            0
        )
        
        # 複合指標
        tpex_merged['trend_beta_signal'] = tpex_merged['idx_trend_5d'] * tpex_merged['market_beta_30d']
        
        # RSI 相關特徵
        if 'idx_rsi_14' in tpex_merged.columns:
            tpex_merged['rel_strength_high_rsi'] = np.where(
                tpex_merged['idx_rsi_14'] > 70,
                tpex_merged['rel_strength'],
                0
            )
            
            tpex_merged['rel_strength_low_rsi'] = np.where(
                tpex_merged['idx_rsi_14'] < 30,
                tpex_merged['rel_strength'],
                0
            )
        
        # 創建一個臨時的結果 DataFrame，包含上櫃股票相對於大盤的所有特徵
        tpex_result = tpex_merged[['date', 'stock_id', 'rel_strength', 'rel_strength_5d_ma',
                                  'rel_strength_10d_ma', 'rel_strength_20d_ma', 'rel_strength_change',
                                  'rel_strength_cum_3d', 'rel_strength_cum_5d', 'rel_strength_cum_10d',
                                  'market_beta_30d', 'market_corr_20d', 'rel_strength_in_uptrend',
                                  'rel_strength_in_downtrend', 'trend_beta_signal']]
        
        # 添加 RSI 相關特徵
        if 'rel_strength_high_rsi' in tpex_merged.columns:
            tpex_result['rel_strength_high_rsi'] = tpex_merged['rel_strength_high_rsi']
            tpex_result['rel_strength_low_rsi'] = tpex_merged['rel_strength_low_rsi']
        
        # 添加大盤指標
        for col in ['idx_trend_5d', 'idx_trend_20d', 'idx_three_up', 'idx_three_down',
                    'idx_new_high_20d', 'idx_new_low_20d', 'idx_rsi_14',
                    'idx_macd_histogram', 'idx_volatility_20d']:
            if col in tpex_merged.columns:
                tpex_result[f'midx_{col}'] = tpex_merged[col]
    
    # 合併上市和上櫃的結果
    if not twse_stocks.empty and not tpex_stocks.empty:
        print("合併上市和上櫃股票的特徵...")
        # 核心方法使用concat，比循環效率更高
        result_df_temp = pd.concat([twse_result, tpex_result], ignore_index=True)
    elif not twse_stocks.empty:
        result_df_temp = twse_result
    elif not tpex_stocks.empty:
        result_df_temp = tpex_result
    else:
        # 如果沒有任何股票資料，創建一個空的DataFrame
        result_df_temp = pd.DataFrame(columns=['date', 'stock_id'])
    
    # 合併回原始result_df
    print("合併指數特徵到最終結果...")
    merge_start = time.time()
    result_df = pd.merge(result_df, result_df_temp, on=['date', 'stock_id'], how='left')
    print(f"合併完成，耗時: {time.time() - merge_start:.2f}秒")
    
    # 處理NaN和無限值
    result_df = result_df.replace([np.inf, -np.inf], np.nan)
    result_df = result_df.fillna(0)
    
    # 從結果中移除type和market_type欄位，因為這些只是中間過程需要的
    if 'type' in result_df.columns:
        result_df = result_df.drop(columns=['type'])
    if 'market_type' in result_df.columns:
        result_df = result_df.drop(columns=['market_type'])
    
    print(f"股票指數特徵計算完成，總耗時: {time.time() - start_time:.2f}秒")
    
    return result_df
def generate_market_index_features_optimized(df, stock_info=None, lookback_days=30):
    """
    優化後的市場指數相關特徵生成函數，移除快取功能，確保每次計算都是最新的
    
    Parameters:
    -----------
    df : pd.DataFrame
        原始特徵DataFrame
    stock_info : pd.DataFrame, optional
        股票基本資訊
    lookback_days : int, default=30
        特徵回溯天數
        
    Returns:
    --------
    pd.DataFrame
        合併指數特徵後的DataFrame
    """
    import pandas as pd
    import numpy as np
    import time
    from datetime import datetime, timedelta
    from tqdm import tqdm
    import gc
    
    print("計算市場指數特徵...")
    start_time = time.time()
    
    # 如果沒有提供股票基本資訊，則獲取
    if stock_info is None:
        stock_info = get_stock_info_cached()
    
    # 讀取指數資料 - 使用優化後的快取函數
    twse_df, tpex_df = read_index_data_cached()
    
    # 計算回溯開始日期
    min_date = pd.to_datetime(df['date']).min()
    max_date = pd.to_datetime(df['date']).max()
    start_date = min_date - timedelta(days=lookback_days)
    start_date_str = start_date.strftime('%Y-%m-%d')
    
    # 過濾並排序指數資料
    twse_df = twse_df[twse_df['date'] >= start_date].sort_values('date')
    tpex_df = tpex_df[tpex_df['date'] >= start_date].sort_values('date')
    
    # 計算指數特徵 - 使用優化後的函數
    print("計算加權指數特徵...")
    feature_start = time.time()
    twse_features = calculate_index_features_optimized(twse_df)
    print(f"完成加權指數特徵計算，耗時: {time.time() - feature_start:.2f}秒")
    
    print("計算櫃買指數特徵...")
    feature_start = time.time()
    tpex_features = calculate_index_features_optimized(tpex_df)
    print(f"完成櫃買指數特徵計算，耗時: {time.time() - feature_start:.2f}秒")
    
    # 計算股票相對於指數的特徵 - 使用優化後的函數
    print("計算股票相對於指數的特徵...")
    feature_start = time.time()
    index_features = calculate_stock_index_features_optimized(df, twse_features, tpex_features, stock_info)
    print(f"完成股票相對指數特徵計算，耗時: {time.time() - feature_start:.2f}秒")
    
    # 合併到原始特徵
    print("合併指數特徵到主DataFrame...")
    merge_start = time.time()
    
    # 確保日期類型一致
    df['date'] = pd.to_datetime(df['date'])
    index_features['date'] = pd.to_datetime(index_features['date'])
    
    # 在合併前檢查並移除type欄位
    if 'type' in df.columns:
        print("從主DataFrame移除type欄位...")
        df = df.drop(columns=['type'])
    
    # 記錄合併前的資料筆數
    pre_merge_count = len(df)
    
    # 使用左連接保留原始DataFrame中的所有行
    result_df = pd.merge(df, index_features, on=['date', 'stock_id'], how='left')
    
    # 檢查合併後的資料筆數是否一致
    post_merge_count = len(result_df)
    if post_merge_count != pre_merge_count:
        print(f"警告: 合併前後資料筆數不一致! 合併前: {pre_merge_count}, 合併後: {post_merge_count}")
        # 嘗試修復 - 移除可能的重複項
        result_df = result_df.drop_duplicates(subset=['date', 'stock_id'], keep='first')
        print(f"修復後資料筆數: {len(result_df)}")
    
    # 再次檢查是否有type欄位
    if 'type' in result_df.columns:
        print("從結果DataFrame移除type欄位...")
        result_df = result_df.drop(columns=['type'])
    
    # 填充可能的缺失值
    result_df = result_df.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    print(f"完成合併操作，耗時: {time.time() - merge_start:.2f}秒")
    
    # 釋放記憶體
    del twse_features, tpex_features, index_features
    gc.collect()
    
    print(f"市場指數特徵計算完成，總耗時: {time.time() - start_time:.2f}秒")
    
    return result_df
def calculate_simplified_index_features(stock_df, twse_index_df, tpex_index_df, stock_info_df=None):
    """
    計算簡化版個股相對於指數的特徵
    
    只計算以下核心特徵:
    1. 相對強弱 (股票漲跌幅 - 指數漲跌幅)
    2. 相對強弱移動平均
    3. 相對強弱變化 (加速度)
    """
    import pandas as pd
    import numpy as np
    import time
    
    start_time = time.time()
    print("計算簡化版指數特徵...")
    
    # 複製資料避免修改原始資料
    df = stock_df.copy()
    df['date'] = pd.to_datetime(df['date'])
    
    # 確保指數日期正確
    twse_index_df['date'] = pd.to_datetime(twse_index_df['date'])
    tpex_index_df['date'] = pd.to_datetime(tpex_index_df['date'])
    
    # 計算指數漲跌幅
    twse_index_df['prev_close'] = twse_index_df['close'].shift(1)
    twse_index_df['idx_close_change'] = (twse_index_df['close'] - twse_index_df['prev_close']) / twse_index_df['prev_close']
    twse_index_df = twse_index_df[['date', 'idx_close_change']]
    
    tpex_index_df['prev_close'] = tpex_index_df['close'].shift(1)
    tpex_index_df['idx_close_change'] = (tpex_index_df['close'] - tpex_index_df['prev_close']) / tpex_index_df['prev_close']
    tpex_index_df = tpex_index_df[['date', 'idx_close_change']]
    
    # 判斷是上市還是上櫃
    if stock_info_df is not None and 'type' in stock_info_df.columns:
        if 'type' not in df.columns:
            stock_info_subset = stock_info_df[['stock_id', 'type']].copy()
            stock_info_subset['stock_id'] = stock_info_subset['stock_id'].astype(str)
            df['stock_id'] = df['stock_id'].astype(str)
            df = pd.merge(df, stock_info_subset, on='stock_id', how='left')
    
    # 預設使用加權指數
    df['market_type'] = 'twse'
    if 'type' in df.columns:
        df.loc[df['type'] == 'tpex', 'market_type'] = 'tpex'
    
    # 準備結果資料框
    result_df = df[['date', 'stock_id']].copy()
    if 'is_analysis_period' in df.columns:
        result_df['is_analysis_period'] = df['is_analysis_period']
    
    # 計算股票漲跌幅
    stock_change_col = 'p_close_change' if 'p_close_change' in df.columns else None
    if stock_change_col is None and 'close' in df.columns and 'prev_close' in df.columns:
        df['stock_change'] = (df['close'] - df['prev_close']) / (df['prev_close'] + 1e-10)
        stock_change_col = 'stock_change'
    elif stock_change_col is None:
        df['stock_change'] = 0
        stock_change_col = 'stock_change'
    
    # 合併指數資料
    twse_stocks = df[df['market_type'] == 'twse'].copy()
    if not twse_stocks.empty:
        twse_stocks = pd.merge(twse_stocks, twse_index_df, on='date', how='left')
        
        # 計算相對強弱
        twse_stocks['rel_strength'] = twse_stocks[stock_change_col] - twse_stocks['idx_close_change']
        
        # 計算相對強弱移動平均
        for window in [5, 10, 20]:
            twse_stocks[f'rel_strength_{window}d_ma'] = twse_stocks.groupby('stock_id')['rel_strength'].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean()
            )
        
        # 計算相對強弱變化
        twse_stocks['rel_strength_change'] = twse_stocks.groupby('stock_id')['rel_strength'].diff()
        
        # 創建相對強弱累積
        for period in [3, 5, 10]:
            twse_stocks[f'rel_strength_cum_{period}d'] = twse_stocks.groupby('stock_id')['rel_strength'].transform(
                lambda x: x.rolling(window=period, min_periods=1).sum()
            )
    
    tpex_stocks = df[df['market_type'] == 'tpex'].copy()
    if not tpex_stocks.empty:
        tpex_stocks = pd.merge(tpex_stocks, tpex_index_df, on='date', how='left')
        
        # 計算相對強弱
        tpex_stocks['rel_strength'] = tpex_stocks[stock_change_col] - tpex_stocks['idx_close_change']
        
        # 計算相對強弱移動平均
        for window in [5, 10, 20]:
            tpex_stocks[f'rel_strength_{window}d_ma'] = tpex_stocks.groupby('stock_id')['rel_strength'].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean()
            )
        
        # 計算相對強弱變化
        tpex_stocks['rel_strength_change'] = tpex_stocks.groupby('stock_id')['rel_strength'].diff()
        
        # 創建相對強弱累積
        for period in [3, 5, 10]:
            tpex_stocks[f'rel_strength_cum_{period}d'] = tpex_stocks.groupby('stock_id')['rel_strength'].transform(
                lambda x: x.rolling(window=period, min_periods=1).sum()
            )
    
    # 合併結果
    rel_columns = ['date', 'stock_id', 'rel_strength', 'rel_strength_5d_ma', 
                  'rel_strength_10d_ma', 'rel_strength_20d_ma', 'rel_strength_change', 
                  'rel_strength_cum_3d', 'rel_strength_cum_5d', 'rel_strength_cum_10d']
    
    if not twse_stocks.empty and not tpex_stocks.empty:
        rel_twse = twse_stocks[rel_columns].copy()
        rel_tpex = tpex_stocks[rel_columns].copy()
        rel_result = pd.concat([rel_twse, rel_tpex], ignore_index=True)
    elif not twse_stocks.empty:
        rel_result = twse_stocks[rel_columns].copy()
    elif not tpex_stocks.empty:
        rel_result = tpex_stocks[rel_columns].copy()
    else:
        rel_result = pd.DataFrame(columns=rel_columns)
    
    # 合併到結果
    result_df = pd.merge(result_df, rel_result, on=['date', 'stock_id'], how='left')
    
    # 清理結果
    result_df = result_df.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # 移除type欄位
    if 'type' in result_df.columns:
        result_df = result_df.drop(columns=['type'])
    if 'market_type' in result_df.columns:
        result_df = result_df.drop(columns=['market_type'])
    
    print(f"簡化版指數特徵計算完成，耗時: {time.time() - start_time:.2f}秒")
    return result_df
# 在prepare_training_data_with_flexible_features函數中添加市場指數特徵
def prepare_training_data_with_flexible_features_updated(start_date: str, end_date: str, args, lookback_days: int = 30) -> pd.DataFrame:
    """
    准备训练资料，根据指定的参数弹性选择特征，包含市场指数特征，并排除每天重复的标的
    
    Parameters:
    -----------
    start_date : str
        分析开始日期，格式为 'YYYY-MM-DD'
    end_date : str
        分析结束日期，格式为 'YYYY-MM-DD'
    args : argparse.Namespace
        命令列参数，控制使用哪些特征
    lookback_days : int, default=30
        向前回溯的天数，用于计算滚动特征
    
    Returns:
    --------
    pd.DataFrame
        准备好的特征资料，仅包含实际分析期间的数据，每天每个标的只会出现一次
    """
    import pandas as pd
    import numpy as np
    import time
    import os
    from datetime import datetime, timedelta
    
    # 产生特征识别字串用于快取檔名
    feature_id = generate_feature_identifier(args)
    
    # 使用市场指数特征的标志
    use_market_index = getattr(args, 'use_market_index', True)
    
    # 生成快取檔案名称
    feature_id_with_market = f"{feature_id}_with_market" if use_market_index else feature_id
    filename = f"final_features_{start_date}_to_{end_date}_lookback{lookback_days}_{feature_id_with_market}.pkl"
    
    # 检查快取是否存在
    cache_dir = os.path.join('D:/data_cache/features', 'combined_features')
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, filename)
    
    # 如果没有快取或载入失败，执行完整的特征计算流程
    print(f"开始准备特征 (使用特征组合: {feature_id_with_market})...")
    
    # 获取包含回溯期间的原始资料
    stock_data = get_stock_data_with_lookback_cached(start_date, end_date, lookback_days)
    
    # 确保日期格式一致
    stock_data['date'] = pd.to_datetime(stock_data['date'])
    stock_data = stock_data.sort_values(['date'])
    
    # 检查并移除每天重复的标的
    # 首先检查是否有重复
    duplicate_check = stock_data.duplicated(subset=['date', 'stock_id'], keep=False)
    if duplicate_check.any():
        print(f"\n警告: 发现{duplicate_check.sum()}个重复的日期和股票ID组合")
        # 只保留每个日期和股票ID组合的第一条记录
        stock_data = stock_data.drop_duplicates(subset=['date', 'stock_id'], keep='first')
        print(f"已删除重复项，现在原始数据集包含{len(stock_data)}行")
    else:
        print("\n检查通过: 原始数据中没有重复的日期和股票ID组合")
    
    # 获取股票基本资料
    stock_info = get_stock_info_cached()
    
    # 获取兴柜股票清单
    emerging_stocks = set(stock_info[stock_info['type'] == 'emerging']['stock_id'])
    
    # 初始化DataFrame
    df = stock_data.copy()

    # 根据参数选择性添加特征
    if args.use_price:
        print("计算价格特征...")
        df = calculate_price_features(df)
    
    if args.use_volume:
        print("计算成交量特征...")
        df = calculate_volume_features(df)  

    # 筹码分点特征
    broker_features = None
    if args.use_broker:
        print("获取筹码分点资料...")
        broker_data = get_broker_data_with_lookback_cached(start_date, end_date, lookback_days)
        broker_data['date'] = pd.to_datetime(broker_data['date'])
        
        # 检查并移除筹码分点数据中每天重复的标的
        broker_duplicate_check = broker_data.duplicated(subset=['date', 'stock_id'], keep=False)
        if broker_duplicate_check.any():
            print(f"\n警告: 发现{broker_duplicate_check.sum()}个重复的筹码分点日期和股票ID组合")
            broker_data = broker_data.drop_duplicates(subset=['date', 'stock_id'], keep='first')
            print(f"已删除筹码分点重复项，现在筹码分点数据集包含{len(broker_data)}行")
        
        print("计算筹码分点特征...")
        broker_features = generate_broker_features_optimized_cached(broker_data, stock_data)
        broker_features['date'] = pd.to_datetime(broker_features['date'])
        
        # 检查计算后的筹码特征是否有重复
        broker_features_duplicate_check = broker_features.duplicated(subset=['date', 'stock_id'], keep=False)
        if broker_features_duplicate_check.any():
            print(f"\n警告: 计算后的筹码特征中发现{broker_features_duplicate_check.sum()}个重复项")
            broker_features = broker_features.drop_duplicates(subset=['date', 'stock_id'], keep='first')
    
    # 融资融券特征
    margin_features = None
    if args.use_margin:
        print("获取融资融券资料...")
        margin_data = get_margin_data_with_lookback_cached(start_date, end_date, lookback_days)
        margin_data['date'] = pd.to_datetime(margin_data['date'])
        
        # 检查并移除融资融券数据中每天重复的标的
        margin_duplicate_check = margin_data.duplicated(subset=['date', 'stock_id'], keep=False)
        if margin_duplicate_check.any():
            print(f"\n警告: 发现{margin_duplicate_check.sum()}个重复的融资融券日期和股票ID组合")
            margin_data = margin_data.drop_duplicates(subset=['date', 'stock_id'], keep='first')
            print(f"已删除融资融券重复项，现在融资融券数据集包含{len(margin_data)}行")
        
        print("计算融资融券特征...")
        margin_features = calculate_margin_features(margin_data)
        
        # 检查计算后的融资融券特征是否有重复
        if 'date' in margin_features.columns:
            margin_features_duplicate_check = margin_features.duplicated(subset=['date', 'stock_id'], keep=False)
            if margin_features_duplicate_check.any():
                print(f"\n警告: 计算后的融资融券特征中发现{margin_features_duplicate_check.sum()}个重复项")
                margin_features = margin_features.drop_duplicates(subset=['date', 'stock_id'], keep='first')
    
    # 外资特征
    foreign_features = None
    if args.use_foreign:
        print("获取外资交易资料...")
        foreign_features = generate_foreign_features_with_lookback(start_date, end_date, lookback_days)
        foreign_features['date'] = pd.to_datetime(foreign_features['date'])
        
        # 检查并移除外资特征中每天重复的标的
        foreign_duplicate_check = foreign_features.duplicated(subset=['date', 'stock_id'], keep=False)
        if foreign_duplicate_check.any():
            print(f"\n警告: 发现{foreign_duplicate_check.sum()}个重复的外资日期和股票ID组合")
            foreign_features = foreign_features.drop_duplicates(subset=['date', 'stock_id'], keep='first')
            print(f"已删除外资重复项，现在外资数据集包含{len(foreign_features)}行")
    
    # 合并特征，根据参数选择性合并
    print("合并选择的特征...")
     
    # 融资融券特征合并
    if args.use_margin and margin_features is not None:
        if 'is_analysis_period' in margin_features.columns:
            margin_features = margin_features.drop(columns=['is_analysis_period'])
        df = pd.merge(df, margin_features, on=['date', 'stock_id'], how='left', suffixes=('', '_margin'))
        
        # 检查合并后是否有重复
        merge_duplicate_check = df.duplicated(subset=['date', 'stock_id'], keep=False)
        if merge_duplicate_check.any():
            print(f"\n警告: 融资融券合并后发现{merge_duplicate_check.sum()}个重复项")
            df = df.drop_duplicates(subset=['date', 'stock_id'], keep='first')
    
    # 外资特征合并
    if args.use_foreign and foreign_features is not None:
        df = pd.merge(df, foreign_features, on=['date', 'stock_id'], how='left', suffixes=('', '_foreign'))
        
        # 检查合并后是否有重复
        merge_duplicate_check = df.duplicated(subset=['date', 'stock_id'], keep=False)
        if merge_duplicate_check.any():
            print(f"\n警告: 外资特征合并后发现{merge_duplicate_check.sum()}个重复项")
            df = df.drop_duplicates(subset=['date', 'stock_id'], keep='first')
    
    # 筹码分点特征合并
    if args.use_broker and broker_features is not None:
        df = pd.merge(df, broker_features, on=['date', 'stock_id'], how='left', suffixes=('', '_broker'))
        
        # 检查合并后是否有重复
        merge_duplicate_check = df.duplicated(subset=['date', 'stock_id'], keep=False)
        if merge_duplicate_check.any():
            print(f"\n警告: 筹码分点特征合并后发现{merge_duplicate_check.sum()}个重复项")
            df = df.drop_duplicates(subset=['date', 'stock_id'], keep='first')
    
    # 市场指数特征添加 - 使用优化后的函数
    # 使用簡化版市場指數特徵計算函數
    if use_market_index:
        print("計算市場指數特徵 (使用優化版)...")
        
        # 暫時保存type資訊
        if 'type' not in df.columns and 'type' in stock_info.columns:
            print("暫時添加type資訊以便計算市場指數特徵...")
            stock_info_subset = stock_info[['stock_id', 'type']].copy()
            stock_info_subset['stock_id'] = stock_info_subset['stock_id'].astype(str)
            df['stock_id'] = df['stock_id'].astype(str)
            df = pd.merge(df, stock_info_subset, on='stock_id', how='left')
        
        # 讀取指數資料
        twse_df, tpex_df = read_index_data_cached()
        
        # 計算簡化版市場指數特徵
        start_time = time.time()
        index_features_df = calculate_simplified_index_features(df, twse_df, tpex_df, stock_info)
        print(f"簡化版市場指數特徵計算完成，用時: {time.time() - start_time:.2f}秒")
        
        # 合併市場指數特徵到原始DataFrame
        # 確保合併時不會產生重複列
        df = pd.merge(df, index_features_df, on=['date', 'stock_id'], how='left', suffixes=('', '_idx'))
        
        # 檢查合併後是否有重複
        merge_duplicate_check = df.duplicated(subset=['date', 'stock_id'], keep=False)
        if merge_duplicate_check.any():
            print(f"\n警告: 市場指數特徵合併後發現{merge_duplicate_check.sum()}個重複項")
            df = df.drop_duplicates(subset=['date', 'stock_id'], keep='first')
        
        # 確保type欄位被移除
        if 'type' in df.columns:
            print("移除臨時添加的type欄位...")
        df = df.drop(columns=['type'])
    
    df = df.sort_values(['date'])
    print("--------------------")
    for col in df.columns:
        print(col)
    print("--------------------") 
    
    # 建立复合特征，只有在选择复合特征时计算
    # 建立简化版复合特征
    if args.use_composite:
        print("创建复合特征...")
        
        # 1. 外资与价格的相关特征
        if 'f_net_ratio' in df.columns and 'p_close_change' in df.columns:
            df['comp_f_price_alignment'] = df['f_net_ratio'] * df['p_close_change']
            df['comp_f_price_divergence'] = df['f_net_ratio'] - df['p_close_change']
        
        # 2. 融资融券与价格的相关特征
        if 'm_momentum' in df.columns and 'p_close_change' in df.columns:
            df['comp_m_price_alignment'] = df['m_momentum'] * df['p_close_change']
            df['comp_m_price_divergence'] = df['m_momentum'] - df['p_close_change']
        
        # 3. 外资与融资融券的相关特征
        if 'f_net_ratio' in df.columns and 'm_momentum' in df.columns:
            df['comp_f_m_alignment'] = df['f_net_ratio'] * df['m_momentum']
            df['comp_f_m_divergence'] = np.abs(df['f_net_ratio'] - df['m_momentum'])
        
        # 4. 成交量与价格的相关特征
        if 'vol_change' in df.columns and 'p_close_change' in df.columns:
            df['comp_vol_price_alignment'] = df['p_close_change'] * df['vol_change']
            
            # 避免除以零或生成无限值
            temp = df['vol_change'].replace(0, np.nan)
            df['comp_vol_price_ratio'] = df['p_close_change'] / temp
            df['comp_vol_price_ratio'] = df['comp_vol_price_ratio'].replace([np.inf, -np.inf], np.nan).fillna(0)
        
        # 5. 相对强弱与价格变化的组合
        if 'rel_strength' in df.columns and 'p_close_change' in df.columns:
            df['comp_rel_strength_price'] = df['rel_strength'] * df['p_close_change']
        
        # 清理数据
        comp_columns = [col for col in df.columns if col.startswith('comp_')]
        if comp_columns:
            df[comp_columns] = df[comp_columns].replace([np.inf, -np.inf], np.nan).fillna(0)
        
        print(f"复合特征计算完成，新增特征数量: {len(comp_columns)}个")
    # 计算标签
    print("计算交易标签...")
    df['next_day_high'] = df.groupby('stock_id')['high'].shift(-1)
    df['next_day_low'] = df.groupby('stock_id')['low'].shift(-1)
    df['next_day_open'] = df.groupby('stock_id')['open'].shift(-1)
    df['next_day_close'] = df.groupby('stock_id')['close'].shift(-1)
    df['price_range_ratio'] = (df['next_day_open'] - df['next_day_close']) / df['close'] * 100
    df['label'] = (df['price_range_ratio'] >= 0.2).astype(int)
    df['profit'] = (df['next_day_open'] - df['next_day_close'])
    df['profit_volume'] = (df['next_day_open'] * 0.0015 + 
                          (df['next_day_open'] + df['next_day_close']) * 0.001425 * 0.3) #假设拿3折手续费
    
    df['stock_id'] = df['stock_id'].astype(str)
    
    # 确保不会有NaN值导致过滤问题
    if 'vol_ma5' in df.columns:
        df['vol_ma5'] = df['vol_ma5'].astype(float).fillna(0)
    if 't_money' in df.columns:
        df['t_money'] = df['t_money'].astype(float).fillna(0)
    
    df = df[~df['stock_id'].isin(emerging_stocks)].copy()
    
    # 过滤条件：4码股票、成交量和金额门槛
    stock_filter = df['stock_id'].str.len() == 4
    
    # 确保vol_ma5和t_money列存在
    volume_filter = df['vol_ma5'] > 1000000 if 'vol_ma5' in df.columns else True
    money_filter = df['t_money'] > 100000000 if 't_money' in df.columns else True
    
    # 应用过滤条件
    filtered_df = df[stock_filter & volume_filter & money_filter].copy()
    
    # 最后再检查一次重复
    final_duplicate_check = filtered_df.duplicated(subset=['date', 'stock_id'], keep=False)
    if final_duplicate_check.any():
        print(f"\n最终警告: 过滤后仍有{final_duplicate_check.sum()}个重复项")
        filtered_df = filtered_df.drop_duplicates(subset=['date', 'stock_id'], keep='first')
        print(f"已删除最终重复项，现在数据集包含{len(filtered_df)}行")
    
    filtered_df = filtered_df.sort_values(['date'])
    filtered_df = filtered_df.reset_index(drop=True)
    print(len(filtered_df))
    
    # 移除不需要的字段
    cols_to_drop = [
        'open', 'high', 'low', 't_volume', 't_money',
        'prev_close', 'prev_volume', 'prev_money',
        'next_day_low', 'next_day_close'
    ]
    
    # 确保type也在移除列表中
    if 'type' in filtered_df.columns:
        cols_to_drop.append('type')
    
    if args.use_volume:
        additional_cols = ['vol_ma5', 'vol_ma10', 'vol_ma20', 
                          'vol_money_ma5', 'vol_money_ma10', 'vol_money_ma20']
        cols_to_drop.extend(additional_cols)
    
    drop_cols = [col for col in cols_to_drop if col in filtered_df.columns]
    filtered_df = filtered_df.drop(columns=drop_cols)
    
    # 填充缺失值并处理极端值
    filtered_df = filtered_df.replace([np.inf, -np.inf], np.nan)
    filtered_df = filtered_df.fillna(0)
    
    # 确定是否存在分析期间的列
    analysis_period_columns = [col for col in filtered_df.columns if 'is_analysis_period' in col]
    
    # 使用正确的列名进行过滤
    if 'is_analysis_period' in filtered_df.columns:
        analysis_period_col = 'is_analysis_period'
    elif 'is_analysis_period_x' in filtered_df.columns:
        analysis_period_col = 'is_analysis_period_x'
    else:
        # 如果找不到任何分析期间的列，中止执行
        raise ValueError("找不到任何is_analysis_period相关的列，请检查合并的DataFrame")
    
    # 过滤出只属于分析期间的资料（排除回溯期间的资料）
    filtered_df = filtered_df[filtered_df[analysis_period_col] == True].copy()
    
    # 移除所有is_analysis_period相关的列
    filtered_df = filtered_df.drop(columns=analysis_period_columns)
    
    # 针对每一天过滤不可当冲的股票
    print("过滤不可当冲股票...")
    filtered_dfs = []
    for date in filtered_df['date'].unique():
        # 获取当天的资料
        daily_data = filtered_df[filtered_df['date'] == date].copy()
        
        # 获取当天不可当冲的股票清单
        print(f"获取 {date} 日不可当冲股票清单...")
        no_intraday_stocks = get_no_intraday_stocks(pd.to_datetime(date))
        nosell_intraday_stocks = get_nosell_intraday_stocks(pd.to_datetime(date))
        tpexno_intraday_stocks = get_tpexno_intraday_stocks(pd.to_datetime(date))
        
        # 过滤掉不可当冲的股票
        daily_data = daily_data[~daily_data['stock_id'].astype(str).isin(no_intraday_stocks)]
        daily_data = daily_data[~daily_data['stock_id'].astype(str).isin(nosell_intraday_stocks)]
        daily_data = daily_data[~daily_data['stock_id'].astype(str).isin(tpexno_intraday_stocks)]
        filtered_dfs.append(daily_data)
    
    # 合并所有过滤后的资料
    intra_filtered_df = pd.concat(filtered_dfs, ignore_index=True)
    
    # 最终重复检查
    final_check = intra_filtered_df.duplicated(subset=['date', 'stock_id'], keep=False)
    if final_check.any():
        print(f"\n最终数据集中发现{final_check.sum()}个重复项，将被删除")
        intra_filtered_df = intra_filtered_df.drop_duplicates(subset=['date', 'stock_id'], keep='first')
    
    # 资料检查
    print("\n资料集基本信息:")
    print(f"资料笔数: {len(intra_filtered_df)}")
    print(f"特征数量: {len(intra_filtered_df.columns)}")
    print(f"日期范围: {intra_filtered_df['date'].min()} to {intra_filtered_df['date'].max()}")
    print(f"股票数量: {len(intra_filtered_df['stock_id'].unique())}")
    print("\n标签分布:")
    print(intra_filtered_df['label'].value_counts(normalize=True))
    
    # 储存最终特征快取
    try:
        intra_filtered_df.to_pickle(cache_path)
        print(f"儲存最终特征快取完成: {filename}")
    except Exception as e:
        print(f"儲存快取失败: {e}")
    
    # 分析特征类型统计
    prefix_counts = {}
    for col in filtered_df.columns:
        if col in ['date', 'stock_id', 'label', 'price_range_ratio', 'next_day_high', 
                  'next_day_open', 'profit', 'profit_volume', 'close', 'type']:
            continue
        
        prefix = col.split('_')[0]
        if prefix not in prefix_counts:
            prefix_counts[prefix] = 0
        prefix_counts[prefix] += 1
    
    print("\n特征类型统计:")
    for prefix, count in prefix_counts.items():
        print(f"{prefix} 类型特征: {count}个")
    
    return intra_filtered_df
# 在prepare_training_data_with_flexible_features函數中添加市場指數特徵
import optuna
import numpy as np
import pandas as pd
import os
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import time
from datetime import datetime
import joblib

def select_features(X_train, y_train, feature_columns, threshold='median'):
    """
    使用模型進行特徵選擇
    
    Parameters:
    -----------
    X_train : pd.DataFrame
        訓練特徵
    y_train : pd.Series
        訓練標籤
    feature_columns : list
        特徵列名稱
    threshold : str or float, default='median'
        特徵選擇閾值，可以是'median'或0到1之間的浮點數
    
    Returns:
    --------
    list
        選擇的特徵列表
    """
    from sklearn.feature_selection import SelectFromModel
    
    # 使用XGBoost模型進行特徵選擇
    selector = SelectFromModel(
        estimator=xgb.XGBClassifier(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=8
        ),
        threshold=threshold
    )
    
    selector.fit(X_train, y_train)
    selected_features_mask = selector.get_support()
    selected_features = [feature_columns[i] for i in range(len(feature_columns)) 
                        if selected_features_mask[i]]
    
    print(f"\n選擇的特徵數量: {len(selected_features)}")
    
    # 按前綴分類統計特徵數量
    prefix_counts = {}
    for feature in selected_features:
        prefix = feature.split('_')[0]
        if prefix not in prefix_counts:
            prefix_counts[prefix] = 0
        prefix_counts[prefix] += 1
    
    print("\n各類特徵選擇統計:")
    for prefix, count in prefix_counts.items():
        print(f"{prefix} 類型特徵: 選擇了 {count} 個")
    
    return selected_features
def objective(trial, args, train_df, test_df, feature_columns):
    """
    Optuna的目標函數，用於優化模型參數
    
    Parameters:
    -----------
    trial : optuna.trial.Trial
        Optuna試驗對象
    args : argparse.Namespace
        命令列參數
    train_df : pd.DataFrame
        訓練資料
    test_df : pd.DataFrame
        測試資料
    feature_columns : list
        特徵名稱列表
        
    Returns:
    --------
    float
        模型評分 (越高越好)
    """
    # 定義搜索空間
    param = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'tree_method': 'gpu_hist',  # 使用GPU加速
        'gpu_id': 0,
        
        # 核心超參數
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 100, 3000),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        
        # 正則化參數
        'gamma': trial.suggest_float('gamma', 0.0, 1.0),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 5),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 5),
        
        # 其他參數
        'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1, 10),  # 處理不平衡資料
    }
    
    # 特徵選擇閾值
    feature_threshold = trial.suggest_float('feature_threshold', 0.1, 0.9)
    
    # 預測閾值
    prediction_threshold = trial.suggest_float('prediction_threshold', 0.4, 0.8)
    
    # 處理特徵
    selected_features = select_features(
        train_df[feature_columns], 
        train_df['label'], 
        feature_columns, 
        threshold=feature_threshold
    )
    
    X_train = train_df[selected_features].copy()
    y_train = train_df['label'].copy()
    X_test = test_df[selected_features].copy()
    y_test = test_df['label'].copy()
    
    # 標準化特徵
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 設定早停參數
    early_stopping_rounds = trial.suggest_int('early_stopping_rounds', 50, 300)
    
    # 轉換為DMatrix格式
    dtrain = xgb.DMatrix(X_train_scaled, label=y_train, feature_names=selected_features)
    dvalid = xgb.DMatrix(X_test_scaled, label=y_test, feature_names=selected_features)
    
    # 訓練模型
    eval_set = [(dtrain, "train"), (dvalid, "valid")]
    model = xgb.train(
        param,
        dtrain,
        num_boost_round=param['n_estimators'],
        evals=eval_set,
        early_stopping_rounds=early_stopping_rounds,
        verbose_eval=False
    )
    
    # 評估模型
    train_pred_proba = model.predict(dtrain)
    test_pred_proba = model.predict(dvalid)
    
    train_pred = (train_pred_proba >= prediction_threshold).astype(int)
    test_pred = (test_pred_proba >= prediction_threshold).astype(int)
    
    # 計算訓練和測試的指標
    train_accuracy = accuracy_score(y_train, train_pred)
    test_accuracy = accuracy_score(y_test, test_pred)
    
    # 計算F1分數
    train_f1 = f1_score(y_train, train_pred)
    test_f1 = f1_score(y_test, test_pred)
    
    # 計算ROC AUC
    train_auc = roc_auc_score(y_train, train_pred_proba)
    test_auc = roc_auc_score(y_test, test_pred_proba)
    
    # 計算權益曲線和收益
    try:
        train_equity = calculate_equity_curve(
            train_pred, 
            y_train.values, 
            train_df, 
            0, 
            len(train_df), 
            args.train_start_date,
            args.train_end_date,
            capital_per_trade=args.capital_per_trade,
            open_threshold=0.00
        )
        
        # 使用測試集計算權益曲線
        test_equity = calculate_equity_curve(
            test_pred, 
            y_test.values, 
            test_df, 
            0, 
            len(test_df),
            args.test_start_date,
            args.test_end_date,
            capital_per_trade=args.capital_per_trade,
            open_threshold=0.00
        )
        
        # 從測試集權益曲線中計算收益
        total_profit = test_equity['trade_profit'].sum()
        
        # 檢查收益是否為負
        if total_profit <= 0:
            # 如果收益為負，返回一個懲罰得分
            return -1000
    except Exception as e:
        # 若權益曲線計算出錯，返回懲罰得分
        print(f"權益曲線計算錯誤: {e}")
        return -1000
    
    # 監控過擬合: 訓練集和測試集性能差距
    overfitting_penalty = abs(train_accuracy - test_accuracy) * 10  # 懲罰過擬合
    
    # 計算綜合評分 (可以根據你的目標調整權重)
    # 我們希望測試集性能好，同時避免過擬合
    score = (
        test_auc * 2.0 +         # 重視AUC
        test_f1 * 2.0 +          # 重視F1分數
        test_accuracy * 1.0 +     # 考慮準確率
        total_profit / 10000.0 +  # 考慮總收益 (歸一化)
        -overfitting_penalty      # 懲罰過擬合
    )
    
    # 儲存每次試驗的結果
    trial.set_user_attr('train_accuracy', float(train_accuracy))
    trial.set_user_attr('test_accuracy', float(test_accuracy))
    trial.set_user_attr('train_f1', float(train_f1))
    trial.set_user_attr('test_f1', float(test_f1))
    trial.set_user_attr('train_auc', float(train_auc))
    trial.set_user_attr('test_auc', float(test_auc))
    trial.set_user_attr('total_profit', float(total_profit))
    trial.set_user_attr('overfitting_gap', float(train_accuracy - test_accuracy))
    trial.set_user_attr('selected_features_count', len(selected_features))
    trial.set_user_attr('prediction_threshold', prediction_threshold)
    
    # 返回評分 (Optuna會最大化這個值)
    return score

def run_hyperparameter_optimization(args, n_trials=50):
    """
    執行完整的超參數優化流程
    
    Parameters:
    -----------
    args : argparse.Namespace
        命令列參數
    n_trials : int, default=50
        Optuna試驗次數
        
    Returns:
    --------
    tuple
        (最佳模型, 最佳參數, 研究對象)
    """
    # 載入或準備訓練和測試資料
    print("準備訓練資料...")
    train_df = prepare_training_data_with_flexible_features_updated(
        args.train_start_date, 
        args.train_end_date, 
        args, 
        args.lookback_days
    )
    
    print("準備測試資料...")
    test_df = prepare_training_data_with_flexible_features_updated(
        args.test_start_date, 
        args.test_end_date, 
        args, 
        args.lookback_days
    )
    
    # 篩選特徵
    exclude_columns = [
        'date', 'stock_id', 'label', 'close',
        'price_range_ratio', 'next_day_high', 'next_day_low',
        'next_day_open', 'next_day_close', 'profit', 'profit_volume',
        'type'
    ]
    feature_columns = [col for col in train_df.columns if col not in exclude_columns]
    
    # 確保兩個資料集有相同的特徵
    train_features = set(train_df.columns) - set(exclude_columns)
    test_features = set(test_df.columns) - set(exclude_columns)
    common_features = sorted(list(train_features.intersection(test_features)))
    feature_columns = common_features
    
    # 創建Optuna的研究對象
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    study_name = f"trading_model_optimization_{timestamp}"
    storage_name = f"sqlite:///./hyperopt_results/{study_name}.db"
    
    # 確保目錄存在
    os.makedirs("./hyperopt_results", exist_ok=True)
    
    # 創建日誌目錄
    log_dir = f"./hyperopt_results/logs_{timestamp}"
    os.makedirs(log_dir, exist_ok=True)
    
    # 設定Optuna的Sampler
    sampler = optuna.samplers.TPESampler(seed=42)
    
    # 創建研究對象
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_name,
        direction="maximize",
        sampler=sampler,
        load_if_exists=True
    )
    
    # 建立目標函數的封裝
    def objective_wrapper(trial):
        return objective(trial, args, train_df, test_df, feature_columns)
    
    # 設定回調函數以便記錄進度
    def save_study_callback(study, trial):
        # 計算並保存當前最佳試驗的各項指標
        best_trial = study.best_trial
        
        # 儲存目前的研究進度
        joblib.dump(study, f"{log_dir}/study_progress_{study.trials[-1].number}.pkl")
        
        # 繪製參數重要性圖表 (如果超過10次試驗)
        if len(study.trials) > 10 and len(study.trials) % 10 == 0:
            try:
                # 參數重要性
                param_importance = optuna.importance.get_param_importances(study)
                param_names = list(param_importance.keys())
                importance_values = list(param_importance.values())
                
                plt.figure(figsize=(12, 8))
                plt.barh(param_names, importance_values)
                plt.xlabel('Importance')
                plt.ylabel('Parameter')
                plt.title('Parameter Importance')
                plt.tight_layout()
                plt.savefig(f"{log_dir}/param_importance_{study.trials[-1].number}.png")
                plt.close()
                
                # 試驗績效圖
                trial_numbers = [t.number for t in study.trials if t.value is not None]
                values = [t.value for t in study.trials if t.value is not None]
                
                plt.figure(figsize=(12, 6))
                plt.plot(trial_numbers, values, 'o-')
                plt.xlabel('Trial Number')
                plt.ylabel('Objective Value')
                plt.title('Objective Value per Trial')
                plt.axhline(y=study.best_value, color='r', linestyle='-', label=f'Best Value: {study.best_value:.4f}')
                plt.legend()
                plt.grid(True)
                plt.savefig(f"{log_dir}/objective_values_{study.trials[-1].number}.png")
                plt.close()
            except Exception as e:
                print(f"繪製圖表時出錯: {e}")
                
        # 顯示目前最佳試驗資訊
        print(f"\nTrial {trial.number} finished.")
        if trial.number % 5 == 0:
            print(f"Current best value: {study.best_value:.4f} (trial {study.best_trial.number})")
            print(f"Best parameters so far:")
            for param_name, param_value in study.best_params.items():
                print(f"    {param_name}: {param_value}")
            print(f"Best metrics:")
            for key, value in best_trial.user_attrs.items():
                print(f"    {key}: {value}")
    
    # 執行優化
    try:
        start_time = time.time()
        print(f"開始超參數優化，計畫執行 {n_trials} 次試驗...")
        study.optimize(objective_wrapper, n_trials=n_trials, callbacks=[save_study_callback])
        print(f"優化完成，耗時: {(time.time() - start_time) / 60:.2f} 分鐘")
    except KeyboardInterrupt:
        print("使用者中斷優化過程！")
    
    # 顯示最佳結果
    print("\n最佳試驗結果:")
    print(f"試驗編號: {study.best_trial.number}")
    print(f"目標函數值: {study.best_value:.4f}")
    print("\n最佳參數:")
    for param_name, param_value in study.best_params.items():
        print(f"    {param_name}: {param_value}")
    
    print("\n最佳模型指標:")
    for key, value in study.best_trial.user_attrs.items():
        print(f"    {key}: {value}")
    
    # 使用最佳參數訓練最終模型
    print("\n使用最佳參數訓練最終模型...")
    
    # 從最佳試驗獲取參數
    best_params = study.best_params.copy()
    
    # 處理特殊參數
    feature_threshold = best_params.pop('feature_threshold', 0.5)
    prediction_threshold = best_params.pop('prediction_threshold', 0.6)
    early_stopping_rounds = best_params.pop('early_stopping_rounds', 100)
    
    # 選擇特徵
    selected_features = select_features(
        train_df[feature_columns], 
        train_df['label'], 
        feature_columns, 
        threshold=feature_threshold
    )
    
    # 準備資料
    X_train = train_df[selected_features].copy()
    y_train = train_df['label'].copy()
    X_test = test_df[selected_features].copy()
    y_test = test_df['label'].copy()
    
    # 標準化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 確保模型參數中包含GPU加速設定
    final_params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'tree_method': 'gpu_hist',
        'gpu_id': 0,
        **best_params  # 合併最佳參數
    }
    
    # 訓練最終模型
    dtrain = xgb.DMatrix(X_train_scaled, label=y_train, feature_names=selected_features)
    dvalid = xgb.DMatrix(X_test_scaled, label=y_test, feature_names=selected_features)
    
    eval_set = [(dtrain, "train"), (dvalid, "valid")]
    best_model = xgb.train(
        final_params,
        dtrain,
        num_boost_round=final_params.get('n_estimators', 1000),
        evals=eval_set,
        early_stopping_rounds=early_stopping_rounds,
        verbose_eval=100
    )
    
    # 儲存最終模型和參數
    result_dir = f"./hyperopt_results/best_model_{timestamp}"
    os.makedirs(result_dir, exist_ok=True)
    
    # 儲存模型
    best_model.save_model(f"{result_dir}/best_model.json")
    
    # 儲存標準化器
    joblib.dump(scaler, f"{result_dir}/scaler.pkl")
    
    # 儲存特徵列表
    with open(f"{result_dir}/selected_features.txt", "w") as f:
        f.write("\n".join(selected_features))
    
    # 儲存超參數
    with open(f"{result_dir}/best_params.txt", "w") as f:
        for param_name, param_value in final_params.items():
            f.write(f"{param_name}: {param_value}\n")
        f.write(f"feature_threshold: {feature_threshold}\n")
        f.write(f"prediction_threshold: {prediction_threshold}\n")
        f.write(f"early_stopping_rounds: {early_stopping_rounds}\n")
    
    # 產生特徵重要性分析
    analyze_feature_importance(best_model, selected_features)
    
    # 儲存研究對象以備後續分析
    joblib.dump(study, f"{result_dir}/study.pkl")
    
    print(f"\n最佳模型和參數已儲存至: {result_dir}")
    
    return best_model, best_params, study

def load_and_continue_optimization(study_path, args, additional_trials=20):
    """
    載入已有的研究並繼續優化
    
    Parameters:
    -----------
    study_path : str
        已有研究的路徑
    args : argparse.Namespace
        命令列參數
    additional_trials : int, default=20
        要額外執行的試驗數量
        
    Returns:
    --------
    tuple
        (最佳模型, 最佳參數, 研究對象)
    """
    # 載入研究
    study = joblib.load(study_path)
    print(f"已載入研究，當前最佳值: {study.best_value:.4f} (試驗 {study.best_trial.number})")
    
    # 重新執行優化流程，但使用已有的研究對象
    return run_hyperparameter_optimization(args, n_trials=additional_trials)

def analyze_optimization_results(study_path):
    """
    分析優化結果並生成報告
    
    Parameters:
    -----------
    study_path : str
        已完成研究的路徑
    """
    # 載入研究
    study = joblib.load(study_path)
    
    # 創建結果目錄
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"./hyperopt_results/analysis_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    
    # 參數重要性
    try:
        param_importance = optuna.importance.get_param_importances(study)
        param_names = list(param_importance.keys())
        importance_values = list(param_importance.values())
        
        plt.figure(figsize=(12, 8))
        plt.barh(param_names, importance_values)
        plt.xlabel('Importance')
        plt.ylabel('Parameter')
        plt.title('Parameter Importance')
        plt.tight_layout()
        plt.savefig(f"{results_dir}/param_importance.png")
        plt.close()
        
        # 將參數重要性寫入文件
        with open(f"{results_dir}/param_importance.txt", "w") as f:
            for param, importance in param_importance.items():
                f.write(f"{param}: {importance:.4f}\n")
    except Exception as e:
        print(f"無法計算參數重要性: {e}")
    
    # 繪製優化歷程
    plt.figure(figsize=(12, 6))
    
    # 提取有效的試驗
    valid_trials = [t for t in study.trials if t.value is not None]
    trial_numbers = [t.number for t in valid_trials]
    values = [t.value for t in valid_trials]
    
    plt.plot(trial_numbers, values, 'o-')
    plt.xlabel('Trial Number')
    plt.ylabel('Objective Value')
    plt.title('Optimization History')
    plt.axhline(y=study.best_value, color='r', linestyle='-', label=f'Best Value: {study.best_value:.4f}')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{results_dir}/optimization_history.png")
    plt.close()
    
    # 試驗績效統計
    metrics = ['train_accuracy', 'test_accuracy', 'train_f1', 'test_f1', 
               'train_auc', 'test_auc', 'total_profit', 'overfitting_gap']
    
    # 收集每個試驗的指標
    metric_values = {metric: [] for metric in metrics}
    for trial in valid_trials:
        for metric in metrics:
            if metric in trial.user_attrs:
                metric_values[metric].append(trial.user_attrs[metric])
            else:
                metric_values[metric].append(None)
    
    # 建立績效資料框
    performance_df = pd.DataFrame({
        'trial': trial_numbers,
        'objective': values,
        **{metric: metric_values[metric] for metric in metrics}
    })
    
    # 儲存績效資料
    performance_df.to_csv(f"{results_dir}/trial_performance.csv", index=False)
    
    # 繪製各指標箱型圖
    plt.figure(figsize=(15, 10))
    performance_df[metrics].boxplot()
    plt.title('Performance Metrics Distribution')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{results_dir}/metrics_boxplot.png")
    plt.close()
    
    # 繪製參數分佈
    param_names = list(study.best_params.keys())
    n_params = len(param_names)
    n_cols = 3
    n_rows = (n_params + n_cols - 1) // n_cols
    
    plt.figure(figsize=(15, n_rows * 4))
    for i, param_name in enumerate(param_names, 1):
        plt.subplot(n_rows, n_cols, i)
        
        # 收集參數值
        param_values = [t.params.get(param_name) for t in valid_trials if param_name in t.params]
        objective_values = [t.value for t in valid_trials if param_name in t.params]
        
        # 繪製散點圖
        plt.scatter(param_values, objective_values, alpha=0.7)
        plt.xlabel(param_name)
        plt.ylabel('Objective Value')
        plt.title(f'{param_name} vs Objective')
        plt.grid(True, alpha=0.3)
        
        # 標記最佳值
        best_value = study.best_params[param_name]
        plt.axvline(x=best_value, color='r', linestyle='-', 
                   label=f'Best: {best_value}')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig(f"{results_dir}/param_distributions.png")
    plt.close()
    
    # 生成綜合報告
    with open(f"{results_dir}/optimization_report.txt", "w") as f:
        f.write("==== 超參數優化報告 ====\n\n")
        
        f.write(f"總試驗次數: {len(study.trials)}\n")
        f.write(f"有效試驗次數: {len(valid_trials)}\n\n")
        
        f.write("最佳試驗:\n")
        f.write(f"  試驗編號: {study.best_trial.number}\n")
        f.write(f"  目標函數值: {study.best_value:.4f}\n\n")
        
        f.write("最佳參數:\n")
        for param_name, param_value in study.best_params.items():
            f.write(f"  {param_name}: {param_value}\n")
        f.write("\n")
        
        f.write("最佳模型績效:\n")
        for key, value in study.best_trial.user_attrs.items():
            f.write(f"  {key}: {value}\n")
        f.write("\n")
        
        f.write("參數重要性 (前5):\n")
        counter = 0
        for param, importance in param_importance.items():
            f.write(f"  {param}: {importance:.4f}\n")
            counter += 1
            if counter >= 5:
                break
        f.write("\n")
        
        f.write("績效統計摘要:\n")
        for metric in metrics:
            values = [v for v in metric_values[metric] if v is not None]
            if values:
                f.write(f"  {metric}:\n")
                f.write(f"    平均值: {np.mean(values):.4f}\n")
                f.write(f"    中位數: {np.median(values):.4f}\n")
                f.write(f"    最小值: {np.min(values):.4f}\n")
                f.write(f"    最大值: {np.max(values):.4f}\n")
                f.write(f"    標準差: {np.std(values):.4f}\n\n")
    
    print(f"分析報告已儲存至: {results_dir}")
    
    return results_dir

if __name__ == "__main__":
    import argparse
    
    # 解析命令列參數
    parser = argparse.ArgumentParser(description='交易模型超參數優化')
    
    # 超參數優化參數
    parser.add_argument('--n_trials', type=int, default=50, help='Optuna試驗次數')
    parser.add_argument('--continue_study', type=str, default=None, help='繼續已有研究的路徑')
    parser.add_argument('--analyze_only', type=str, default=None, help='只分析已有研究的路徑')
    
    # 時間區間參數
    parser.add_argument('--train_start_date', type=str, default='2021-01-01', help='訓練開始日期 (YYYY-MM-DD)')
    parser.add_argument('--train_end_date', type=str, default='2022-12-31', help='訓練結束日期 (YYYY-MM-DD)')
    parser.add_argument('--test_start_date', type=str, default='2023-01-01', help='測試開始日期 (YYYY-MM-DD)')
    parser.add_argument('--test_end_date', type=str, default='2023-12-31', help='測試結束日期 (YYYY-MM-DD)')
    
    # 特徵選擇參數
    parser.add_argument('--use_price', action='store_true', default=True, help='使用價格相關特徵')
    parser.add_argument('--use_volume', action='store_true', default=True, help='使用成交量相關特徵')
    parser.add_argument('--use_broker', action='store_true', help='使用籌碼分點特徵')
    parser.add_argument('--use_foreign', action='store_true', default=True, help='使用外資特徵')
    parser.add_argument('--use_margin', action='store_true', default=True, help='使用融資融券特徵')
    parser.add_argument('--use_composite', action='store_true', default=True, help='使用複合特徵')
    parser.add_argument('--use_market_index', action='store_true', default=True, help='使用市場指數特徵')
    
    # 模型與訓練參數
    parser.add_argument('--model_type', type=str, default='xgboost', help='模型類型')
    parser.add_argument('--lookback_days', type=int, default=30, help='回溯天數')
    parser.add_argument('--capital_per_trade', type=int, default=1000, help='每筆交易金額(千元)')
    
    args = parser.parse_args()
    
    # 確保結果目錄存在
    os.makedirs("./hyperopt_results", exist_ok=True)
    
    # 執行選擇的操作
    if args.analyze_only:
        # 只分析研究結果
        print(f"分析研究結果: {args.analyze_only}")
        analyze_optimization_results(args.analyze_only)
    elif args.continue_study:
        # 繼續已有研究
        print(f"繼續研究: {args.continue_study}")
        best_model, best_params, study = load_and_continue_optimization(
            args.continue_study,
            args,
            additional_trials=args.n_trials
        )
    else:
        # 開始新的優化
        print("開始新的超參數優化")
        best_model, best_params, study = run_hyperparameter_optimization(args, n_trials=args.n_trials)