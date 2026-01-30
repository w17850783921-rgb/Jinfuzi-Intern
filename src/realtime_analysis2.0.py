import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.stats import percentileofscore
import os

"""
Last Edit Date: 2026-01-30
Author: Jiawen Liang
Project: Two-factor independent track real trading strategy
"""

# ===================== 0. 全局设定 =====================

# 字体设定
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 文件读取路径
FILE_PATH = './data/realtime_data/merged_index_fund_returns.csv'

# 回测时间区间
BACKTEST_START_DATE = '2023-01-01'
BACKTEST_END_DATE = '2099-12-31'

# 特定日期记录
SPECIFIC_STAT_DATE = '2025-12-24'

# RTVR 策略参数
RTVR_WINDOW = 40 # 40-day average smoothing
RTVR_LOOKBACK = 66 # Calculate the historical percentile of the current value over the past 66 days
RTVR_THRESHOLDS = {'H': 0.70, 'L': 0.30, 'FH': 0.90, 'FL': 0.10, 'MH': 0.60, 'ML': 0.40} # Threshold

# TSM 策略参数
TSM_MIN_STEP = 0.01
TSM_SENSITIVITY = 30

# 交易成本与滑点
COST = 0.0002  # 佣金/印花税等固定成本 (万二)
SLIPPAGE = 0.0003  # 滑点 (万三)：模拟大额订单偏离VWAP的冲击成本

# ===================== 1. 数据加载与预处理（此部分暂无额外预处理） =====================

# Data loading
if not os.path.exists(FILE_PATH):
    print(f"❌ 错误：找不到文件 {FILE_PATH}")
    exit()
try:
    df = pd.read_csv(FILE_PATH, parse_dates=['TradingDay'])
    print(f"✅ 成功加载数据: {len(df)} 条记录")
except Exception as e:
    print(f"❌ 无法读取文件: {e}")
    exit()

# 按照交易日排序
df = df.set_index('TradingDay').sort_index()

#预览已加载的数据
print("数据预览：")
print(df.head())

# ===================== 2. 区分【信号源数据】和【标的数据】 =====================

# 1. 信号源数据：来自指数数据，构建一个新的df，只包含指数数据
signal_df = df[[col for col in df.columns if 'idx' in col]]

# 将0值视为缺失值 (避免除以0错误)
signal_df.replace(0, np.nan, inplace=True)

# 2. 标的数据：来自基金数据，构建一个新的df，只包含基金数据
target_df = df[[col for col in df.columns if 'fund' in col]]

# 将0值视为缺失值 (避免除以0错误)
target_df.replace(0, np.nan, inplace=True)

# ===================== 3. 因子原始数据准备 =====================

# RTVR 数据准备, 中证500交易额 / 中证500交易额 + 红利交易额
df['RTVR_raw'] = signal_df['idx_000905_SH__turnover_value'] / (signal_df['idx_000905_SH__turnover_value'] + signal_df['idx_000922_SH__turnover_value'])

# 计算滑动平均值
df['RTVR_factor'] = df['RTVR_raw'].rolling(window=RTVR_WINDOW, min_periods=1).mean()

# 计算当前值在过去66天中的分位数
df['RTVR_rank'] = df['RTVR_factor'].rolling(window=RTVR_LOOKBACK, min_periods=1).apply(
    lambda x: percentileofscore(x[:-1], x.iloc[-1]) / 100 if len(x) == RTVR_LOOKBACK else np.nan, raw=False
)

# TSM 数据准备, 计算中证500指数的动量
