import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# ----------------------------------------------------------------------
# 📌 0. 全局设置
# ----------------------------------------------------------------------
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

FILE_PATH = r"C:\Users\86178\Desktop\交易情绪因子2.csv"
BACKTEST_START_DATE = '2021-01-01'
BACKTEST_END_DATE = '2099-12-31'

# ======================================================================
# 🌟 核心参数
# ======================================================================
SENTIMENT_WINDOW = 30
MID_TERM_WINDOW = 20
SHORT_TERM_WINDOW = 4
REVERSAL_WEIGHT = 0.8

# 🌟 棘轮策略参数 (Ratchet) 🌟
STRENGTH_WINDOW = 60  # 计算Z-Score的窗口
THRES_START = 0.5  # 开始加仓阈值 (在此之前强制50%)
THRES_FULL = 1.5  # 满仓阈值
THRES_RESET = 0.2  # 止盈重置阈值 (跌破此值，解除锁定回归50%)

COST = 0.0002
SLIPPAGE = 0.0003

# ----------------------------------------------------------------------
# 1. 数据准备
# ----------------------------------------------------------------------
if not os.path.exists(FILE_PATH):
    print(f"❌ 错误：找不到文件 {FILE_PATH}")
    exit()

df = pd.read_csv(FILE_PATH, parse_dates=['TradingDay']).set_index('TradingDay').sort_index()

df.rename(columns={
    'index_return1': 'Ret_Idx_500', 'turnover_value1': 'Val_500', 'negotiable_mv1': 'MV_500',
    'index_return2': 'Ret_Idx_HL', 'turnover_value2': 'Val_HL', 'negotiable_mv2': 'MV_HL'
}, inplace=True)

cols_etf = ['close_price4', 'prev_close4', 'close_price5', 'prev_close5']
df[cols_etf] = df[cols_etf].replace(0, np.nan).ffill().bfill()
df['Ret_ETF_500'] = df['close_price4'] / df['prev_close4'] - 1
df['Ret_ETF_HL'] = df['close_price5'] / df['prev_close5'] - 1


# ----------------------------------------------------------------------
# 2. 因子计算
# ----------------------------------------------------------------------
def calc_sentiment_residual(series_ret, series_val, series_mv, window):
    if series_mv.sum() == 0 or series_mv.isna().all():
        tr = np.log(series_val)
        delta_tr = tr.diff()
    else:
        tr = series_val / series_mv
        delta_tr = tr / tr.shift(1) - 1
    delta_tr = delta_tr.replace([np.inf, -np.inf], np.nan).fillna(0)

    cov = series_ret.rolling(window).cov(delta_tr)
    var = delta_tr.rolling(window).var()
    beta = cov / var
    alpha = series_ret.rolling(window).mean() - beta * delta_tr.rolling(window).mean()

    return series_ret - (alpha + beta * delta_tr)


# 1. 原始因子
df['Sent_500'] = calc_sentiment_residual(df['Ret_Idx_500'], df['Val_500'], df['MV_500'], SENTIMENT_WINDOW)
df['Sent_HL'] = calc_sentiment_residual(df['Ret_Idx_HL'], df['Val_HL'], df['MV_HL'], SENTIMENT_WINDOW)
df['Factor_Cum'] = (df['Sent_500'] - df['Sent_HL']).cumsum()

# 2. 趋势反转合成
df['Mom_Mid'] = df['Factor_Cum'].diff(MID_TERM_WINDOW)
df['Mom_Short'] = df['Factor_Cum'].diff(SHORT_TERM_WINDOW)
df['Alpha_Score'] = df['Mom_Mid'] - (REVERSAL_WEIGHT * df['Mom_Short'])

# 3. 信号平滑
df['Alpha_Score_Smooth'] = df['Alpha_Score'].rolling(3).mean()

# ----------------------------------------------------------------------
# 3. 🔥 棘轮仓位管理 (Ratchet Position Sizing) 🔥
# ----------------------------------------------------------------------
# 计算 Z-Score
roll_mean = df['Alpha_Score_Smooth'].rolling(STRENGTH_WINDOW).mean()
roll_std = df['Alpha_Score_Smooth'].rolling(STRENGTH_WINDOW).std()
df['Signal_Z'] = (df['Alpha_Score_Smooth'] - roll_mean) / roll_std


# 核心棘轮逻辑函数
def calculate_ratchet_weight(z_values, start, full, reset):
    weights = []
    current_w = 0.5  # 初始标配

    # 0.5 = 标配, 1.0 = 满仓500, 0.0 = 满仓红利

    for z in z_values:
        if pd.isna(z):
            weights.append(0.5)
            continue

        # --- 情况 A: 当前持有 500 (w > 0.5) ---
        if current_w > 0.5:
            # 1. 止盈/重置检查: 趋势是否彻底坏了?
            if z < reset:
                current_w = 0.5  # 跌破0.2，所有利润落袋，回归标配
            else:
                # 2. 棘轮逻辑: 计算理论仓位，只增不减
                # 线性映射: (z - start) / (full - start) -> [0, 1]
                # 然后映射到 [0.5, 1.0] 区间
                raw_w = 0.5 + 0.5 * (z - start) / (full - start)
                raw_w = min(raw_w, 1.0)  # 上限1.0

                # 关键: 取 max(当前, 新理论)，实现"只加不减"
                current_w = max(current_w, raw_w)

        # --- 情况 B: 当前持有 红利 (w < 0.5) ---
        elif current_w < 0.5:
            # 1. 止盈/重置检查 (对称逻辑)
            if z > -reset:
                current_w = 0.5  # 反弹回-0.2以上，空头平仓，回归标配
            else:
                # 2. 棘轮逻辑: 只减不增 (即红利仓位只增不减)
                # 计算距离: (abs(z) - start)
                raw_w = 0.5 - 0.5 * (abs(z) - start) / (full - start)
                raw_w = max(raw_w, 0.0)  # 下限0.0

                # 关键: 取 min(当前, 新理论)，实现 500权重"只降不升"
                current_w = min(current_w, raw_w)

        # --- 情况 C: 当前标配 (w == 0.5) ---
        else:
            # 等待突破 0.5 或 -0.5 才能启动
            if z > start:
                current_w = 0.5 + 0.01  # 启动做多 (给一点点增量激活状态)
            elif z < -start:
                current_w = 0.5 - 0.01  # 启动做空
            else:
                current_w = 0.5  # 继续在噪音区躺平

        weights.append(current_w)

    return np.array(weights)


# 应用棘轮逻辑
targets = calculate_ratchet_weight(df['Signal_Z'].values, THRES_START, THRES_FULL, THRES_RESET)
df['Target_Weight'] = targets
df['Exec_Weight'] = df['Target_Weight'].shift(1)  # T+1执行

# ----------------------------------------------------------------------
# 4. 回测执行
# ----------------------------------------------------------------------
df_bt = df.loc[BACKTEST_START_DATE:BACKTEST_END_DATE].copy()
df_bt = df_bt.dropna(subset=['Signal_Z', 'Exec_Weight'])

if df_bt.empty: exit()

targets = df_bt['Exec_Weight'].values
weights = np.zeros(len(df_bt))
w_curr = targets[0]

ret_500 = df_bt['Ret_ETF_500'].values
ret_hl = df_bt['Ret_ETF_HL'].values

for i in range(len(df_bt)):
    if abs(w_curr - targets[i]) > 0.001:
        w_curr = targets[i]
    weights[i] = w_curr

    r_day = w_curr * ret_500[i] + (1 - w_curr) * ret_hl[i]
    w_curr = w_curr * (1 + ret_500[i]) / (1 + r_day)
    w_curr = np.clip(w_curr, 0.0, 1.0)

df_bt['W_500'] = weights
df_bt['Turnover'] = (df_bt['W_500'] - df_bt['W_500'].shift(1).fillna(weights[0])).abs()

raw_ret = df_bt['W_500'] * ret_500 + (1 - df_bt['W_500']) * ret_hl
df_bt['Strat_Ret'] = raw_ret - (df_bt['Turnover'] * (COST + SLIPPAGE) * 2)

df_bt['Strat_Cum'] = (1 + df_bt['Strat_Ret']).cumprod()
df_bt['Bench_Cum'] = (1 + (0.5 * ret_500 + 0.5 * ret_hl)).cumprod()
df_bt['Excess_Cum'] = df_bt['Strat_Cum'] / df_bt['Bench_Cum'] - 1

# ----------------------------------------------------------------------
# 5. 结果展示
# ----------------------------------------------------------------------
ann = (df_bt['Strat_Cum'].iloc[-1] / df_bt['Strat_Cum'].iloc[0]) ** (252 / len(df_bt)) - 1
bench_ann = (df_bt['Bench_Cum'].iloc[-1] / df_bt['Bench_Cum'].iloc[0]) ** (252 / len(df_bt)) - 1
turnover_avg = df_bt['Turnover'].mean()

print("\n" + "=" * 50)
print(f"🏆 棘轮策略 (Ratchet Scaling) 🏆")
print(f"⚙️ 规则: 0~0.5标配 | 0.5~1.5加仓(只进不退) | <0.2止盈重置")
print("=" * 50)
print(f"✅ 策略年化: {ann:.2%}")
print(f"🔹 基准年化: {bench_ann:.2%}")
print(f"🔥 超额收益: {ann - bench_ann:.2%}")
print(f"💸 日均换手: {turnover_avg:.2%}")
print("-" * 50)

# ----------------------------------------------------------------------
# 🌟 画图
# ----------------------------------------------------------------------
fig, axes = plt.subplots(4, 1, figsize=(10, 15), sharex=True)

# 1. 净值
axes[0].plot(df_bt['Strat_Cum'], color='#d62728', lw=2, label='棘轮策略')
axes[0].plot(df_bt['Bench_Cum'], color='gray', ls='--', label='基准')
axes[0].set_title('净值表现')
axes[0].legend(loc='upper left')
axes[0].grid(True, alpha=0.3)

# 2. 超额
axes[1].plot(df_bt['Excess_Cum'], color='blue', lw=1.5, label='累计超额')
axes[1].axhline(0, color='black', ls='--')
axes[1].fill_between(df_bt.index, df_bt['Excess_Cum'], 0, where=(df_bt['Excess_Cum'] > 0), color='red', alpha=0.1)
axes[1].set_title('超额收益')
axes[1].grid(True, alpha=0.3)

# 3. 信号强度 Z-Score
axes[2].plot(df_bt['Signal_Z'], color='purple', lw=1, label='Z-Score')
axes[2].axhline(THRES_START, color='red', ls=':', label='加仓起点(0.5)')
axes[2].axhline(THRES_FULL, color='red', ls='--', label='满仓点(1.5)')
axes[2].axhline(THRES_RESET, color='green', ls='-', label='止盈重置点(0.2)')
axes[2].axhline(-THRES_START, color='orange', ls=':')
axes[2].set_title('信号强度与关键阈值')
axes[2].legend(loc='upper left')

# 4. 棘轮仓位展示
# 这里的仓位应该是阶梯状上升，然后垂直下落
axes[3].plot(df_bt.index, df_bt['W_500'], color='orange', lw=1.5, label='500仓位')
axes[3].fill_between(df_bt.index, df_bt['W_500'], 0, color='orange', alpha=0.3)
axes[3].axhline(0.5, color='gray', ls=':', label='标配线')
axes[3].set_title('棘轮仓位 (阶梯式加仓 -> 垂直重置)')
axes[3].set_ylim(-0.1, 1.1)

plt.tight_layout()
plt.show()