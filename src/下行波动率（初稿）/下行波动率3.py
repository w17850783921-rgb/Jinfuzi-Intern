import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import percentileofscore
import os

# ----------------------------------------------------------------------
# 📌 0. 全局设置
# ----------------------------------------------------------------------
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 🌟 参数设置 🌟
FILE_PATH = r"C:\Users\86178\Desktop\整合数据.csv"
DATA_START_DATE = '2021-01-01'

# 🌟 因子参数 🌟
RDVR_WINDOW = 20
RDVR_LOOKBACK = 60
MOM_WINDOW = 20
RANK_SMOOTH_WIN = 3

# 🌟 交易门槛 🌟
TRADE_THRESHOLD = 0.10

# 交易成本
COST = 0.0002
SLIPPAGE = 0.0003

# ----------------------------------------------------------------------
# 1. 数据加载与预处理
# ----------------------------------------------------------------------
if not os.path.exists(FILE_PATH):
    print(f"❌ 错误：找不到文件 {FILE_PATH}")
    raise SystemExit

df = pd.read_csv(FILE_PATH, parse_dates=['TradingDay']).set_index('TradingDay').sort_index()
df = df.loc[DATA_START_DATE:].copy()
print(f"✅ 数据加载完成: 从 {DATA_START_DATE} 开始 (含预热期)")

# 字段映射
df['Close_Idx_500'] = df['idx_000905_SH__close_price']
df['Close_Idx_HL'] = df['idx_000922_SH__close_price']

# ETF数据清洗
cols = ['fund_512510__close_price', 'fund_512510__prev_close', 'fund_512510__avg_price',
        'fund_515180__close_price', 'fund_515180__prev_close', 'fund_515180__avg_price']
df[cols] = df[cols].replace(0, np.nan).ffill().bfill()

df['Close_ETF_500'] = df['fund_512510__close_price']
df['Prev_ETF_500'] = df['fund_512510__prev_close']
df['Close_ETF_HL'] = df['fund_515180__close_price']
df['Prev_ETF_HL'] = df['fund_515180__prev_close']

# 计算收益率
df['Ret_Idx_500'] = df['idx_000905_SH__close_price'] / df['idx_000905_SH__prev_close'] - 1
df['Ret_Idx_HL'] = df['idx_000922_SH__close_price'] / df['idx_000922_SH__prev_close'] - 1
df['Ret_ETF_500'] = df['Close_ETF_500'] / df['Prev_ETF_500'] - 1
df['Ret_ETF_HL'] = df['Close_ETF_HL'] / df['Prev_ETF_HL'] - 1


# ----------------------------------------------------------------------
# 2. 因子构建
# ----------------------------------------------------------------------
# A. 风险因子
def calc_downside_deviation(series, window):
    downside_ret = series.copy()
    downside_ret[downside_ret > 0] = 0
    return downside_ret.rolling(window).std()


df['DD_500'] = calc_downside_deviation(df['Ret_Idx_500'], RDVR_WINDOW)
df['DD_HL'] = calc_downside_deviation(df['Ret_Idx_HL'], RDVR_WINDOW)
df['RDVR_Diff'] = df['DD_500'] - df['DD_HL']

df['RDVR_Rank'] = df['RDVR_Diff'].rolling(RDVR_LOOKBACK).apply(
    lambda x: percentileofscore(x[:-1], x.iloc[-1]) / 100 if len(x) == RDVR_LOOKBACK else np.nan, raw=False
)
df['RDVR_Rank_Smooth'] = df['RDVR_Rank'].rolling(RANK_SMOOTH_WIN).mean()

# B. 动量因子
df['Mom_500'] = df['Close_Idx_500'].pct_change(MOM_WINDOW)
df['Mom_HL'] = df['Close_Idx_HL'].pct_change(MOM_WINDOW)
df['500_Stronger'] = df['Mom_500'] > df['Mom_HL']


# ----------------------------------------------------------------------
# 3. 生成目标仓位 (🌟 核心修改：借鉴参考代码的线性调仓 + 棘轮逻辑)
# ----------------------------------------------------------------------
# 逻辑说明：
# 1. 基础状态 (Base) 为 0.5 (标配)。
# 2. Rank > 0.8 (高危): 仓位从 0.5 线性降至 0.0。
# 3. Rank < 0.2 (安全): 仓位从 0.5 线性升至 1.0。
# 4. 棘轮效应: 在高危区只降不升，在安全区只升不降，直到回到中枢 (0.4-0.6) 重置。
# 5. 动量保护: 计算出的线性仓位，如果遇到 500 弱势，直接一票否决归零。

def get_linear_target(P):
    """根据 Rank 计算理论上的线性仓位"""
    if pd.isna(P): return 0.5

    # --- 危险区 (Rank 0.8 ~ 1.0) ---
    # 目标从 0.5 降到 0.0
    if P > 0.80:
        # (P - 0.80) / 0.20 归一化到 0~1
        ratio = (P - 0.80) / 0.20
        # 结果: 0.5 减去 (0~0.5)
        return 0.5 - (ratio * 0.5)

    # --- 安全区 (Rank 0.0 ~ 0.2) ---
    # 目标从 0.5 升到 1.0
    if P < 0.20:
        # (0.20 - P) / 0.20 归一化到 0~1
        ratio = (0.20 - P) / 0.20
        # 结果: 0.5 加上 (0~0.5)
        return 0.5 + (ratio * 0.5)

    # --- 中间区 ---
    return 0.5


df['Target_500'] = 0.5
current_state_w = 0.5  # 记录上一期的状态位

for i in range(len(df)):
    rank = df['RDVR_Rank_Smooth'].iloc[i]
    is_strong = df['500_Stronger'].iloc[i]

    if pd.isna(rank):
        df.iloc[i, df.columns.get_loc('Target_500')] = 0.5
        continue

    # --- A. 线性 + 棘轮计算 (基于风险) ---
    # 1. 回归中枢 (Reset Zone): 0.4 ~ 0.6 -> 重置为 0.5
    if 0.40 <= rank <= 0.60:
        current_state_w = 0.5

    # 2. 危险区 (Rank > 0.8): 触发线性减仓，且只降不升 (棘轮)
    elif rank > 0.80:
        linear_w = get_linear_target(rank)
        # 保持之前的低仓位，或者变得更低，不轻易反弹
        current_state_w = min(current_state_w, linear_w)

    # 3. 安全区 (Rank < 0.2): 触发线性加仓，且只升不降 (棘轮)
    elif rank < 0.20:
        linear_w = get_linear_target(rank)
        # 保持之前的高仓位，或者变得更高，不轻易回撤
        current_state_w = max(current_state_w, linear_w)

    # 4. 缓冲区 (0.2~0.4 和 0.6~0.8): 保持上一期状态不变 (Hysteresis)
    else:
        # current_state_w 保持不变
        pass

    # --- B. 动量一票否决 (Momentum Veto) ---
    # 即使线性模型说要满仓 (Rank很低)，如果500走势弱于红利，也必须空仓。
    # 保护 2022 年这种 "低波动阴跌" 行情。
    if not is_strong:
        final_target = 0.0
    else:
        final_target = current_state_w

    df.iloc[i, df.columns.get_loc('Target_500')] = final_target

# ----------------------------------------------------------------------
# 4. 回测执行 (漂移逻辑 + B&H基准)
# ----------------------------------------------------------------------
df['Target_500_Exec'] = df['Target_500'].shift(1)
df_bt = df.dropna(subset=['RDVR_Rank_Smooth', 'Target_500_Exec']).copy()

print(f"🚀 实际交易区间: {df_bt.index[0].date()} 至 {df_bt.index[-1].date()}")

ret_500 = df_bt['Ret_ETF_500'].values
ret_hl = df_bt['Ret_ETF_HL'].values
targets = df_bt['Target_500_Exec'].values

w_actual_500 = np.zeros(len(df_bt))
strat_ret = np.zeros(len(df_bt))
turnover = np.zeros(len(df_bt))

# 初始化漂移变量
current_w_drifted = targets[0]
w_actual_500[0] = current_w_drifted

for i in range(len(df_bt)):
    target_today = targets[i]

    # --- 判断调仓 (Threshold) ---
    if abs(target_today - current_w_drifted) > TRADE_THRESHOLD:
        w_final = target_today  # 强制归位
        delta = abs(w_final - current_w_drifted)
    else:
        w_final = current_w_drifted  # 保持漂移
        delta = 0.0

    turnover[i] = delta
    w_actual_500[i] = w_final

    # --- 计算收益 ---
    cost_total = delta * (COST + SLIPPAGE) * 2
    gross_ret = w_final * ret_500[i] + (1 - w_final) * ret_hl[i]
    net_ret = gross_ret - cost_total
    strat_ret[i] = net_ret

    # --- 计算次日漂移权重 ---
    if i < len(df_bt) - 1:
        new_val_500 = w_final * (1 + ret_500[i])
        new_val_port = 1 + gross_ret
        current_w_drifted = new_val_500 / new_val_port

df_bt['Strat_Ret'] = strat_ret
df_bt['Strat_Cum'] = (1 + df_bt['Strat_Ret']).cumprod()
df_bt['Turnover'] = turnover
df_bt['W_Actual_500'] = w_actual_500

# 基准 (Buy & Hold)
nav_500 = (1 + df_bt['Ret_ETF_500']).cumprod()
nav_hl = (1 + df_bt['Ret_ETF_HL']).cumprod()
df_bt['Bench_Cum'] = 0.5 * nav_500 + 0.5 * nav_hl
df_bt['Bench_Cum'] = df_bt['Bench_Cum'] / df_bt['Bench_Cum'].iloc[0] * df_bt['Strat_Cum'].iloc[0]

# 超额收益
df_bt['Excess_Cum'] = df_bt['Strat_Cum'] / df_bt['Bench_Cum'] - 1

# ----------------------------------------------------------------------
# 5. 绩效与画图
# ----------------------------------------------------------------------
ann_ret = (df_bt['Strat_Cum'].iloc[-1] / df_bt['Strat_Cum'].iloc[0]) ** (252 / len(df_bt)) - 1
bench_ann = (df_bt['Bench_Cum'].iloc[-1] / df_bt['Bench_Cum'].iloc[0]) ** (252 / len(df_bt)) - 1
mdd = (df_bt['Strat_Cum'] / df_bt['Strat_Cum'].cummax() - 1).min()
sharpe = ann_ret / (df_bt['Strat_Ret'].std() * np.sqrt(252))

print("\n" + "=" * 50)
print(f"🚀 RDVR v7 (线性调仓 + 棘轮锁定 + 动量否决) 🚀")
print("=" * 50)
print(f"策略年化: {ann_ret:.2%}")
print(f"基准年化: {bench_ann:.2%}")
print(f"超额年化: {ann_ret - bench_ann:.2%}")
print(f"最大回撤: {mdd:.2%}")
print(f"日均换手: {df_bt['Turnover'].mean():.2%}")

# 画图
fig, axes = plt.subplots(4, 1, figsize=(12, 14), sharex=True)

# 1. 净值
axes[0].plot(df_bt['Strat_Cum'], label='Strategy v7', color='#d62728', lw=2)
axes[0].plot(df_bt['Bench_Cum'], label='Benchmark (B&H)', color='black', ls='--')
axes[0].set_title('Strategy Equity Curve')
axes[0].legend(loc='upper left')
axes[0].grid(True, alpha=0.3)

# 2. 真实持仓 (可以看到线性的变化)
axes[1].fill_between(df_bt.index, df_bt['W_Actual_500'], 0, color='#1f77b4', alpha=0.3)
axes[1].plot(df_bt['W_Actual_500'], color='#1f77b4', label='Actual Weight 500')
axes[1].set_ylabel('Weight 500')
axes[1].set_title('Actual Weight (Linear Adjustment)')
axes[1].grid(True, alpha=0.3)

# 3. 超额收益
axes[2].plot(df_bt['Excess_Cum'], color='green', lw=1.5, label='Excess Return')
axes[2].fill_between(df_bt.index, df_bt['Excess_Cum'], 0, where=(df_bt['Excess_Cum'] > 0), color='green', alpha=0.1)
axes[2].fill_between(df_bt.index, df_bt['Excess_Cum'], 0, where=(df_bt['Excess_Cum'] < 0), color='red', alpha=0.1)
axes[2].axhline(0, color='black', ls='--', lw=1)
axes[2].set_title('Cumulative Excess Return')

# 4. 因子监控
ax4 = axes[3]
ax4.plot(df_bt['RDVR_Rank_Smooth'], color='orange', label='Rank', lw=1)
# 画出线性调整的区间线
ax4.axhline(0.8, color='red', ls=':', label='Start Selling (>0.8)')
ax4.axhline(0.2, color='green', ls=':', label='Start Buying (<0.2)')
ax4.set_title('Rank Monitor (Linear Zones: <0.2 and >0.8)')
ax4.legend(loc='upper right')

plt.tight_layout()
plt.show()

# ----------------------------------------------------------------------
# 6. 实盘指导
# ----------------------------------------------------------------------
latest = df.iloc[-1]
curr_drift_w = w_actual_500[-1]
target = latest['Target_500']
rank = latest['RDVR_Rank_Smooth']

print("\n" + "#" * 60)
print(f"📢 实盘操作建议 (当前持仓已自然漂移至: {curr_drift_w:.2%})")
print("#" * 60)
print(f"1. 因子状态:")
print(f"   - Rank: {rank:.2%} (线性计算区间: 0.2~0.8)")
print(f"   - Mom:  {'500强' if latest['500_Stronger'] else '红利强 (一票否决)'}")
print(f"2. 信号目标: {target:.2%}")

if abs(target - curr_drift_w) > TRADE_THRESHOLD:
    print(f"3. 操作动作: ✅ 偏差 > {TRADE_THRESHOLD:.0%}，触发调仓！")
    print(f"   -> 请将中证500调整至: {target:.2%}")
    print(f"   -> 请将红利低波调整至: {1 - target:.2%}")
else:
    print(f"3. 操作动作: 🚫 偏差 {abs(target - curr_drift_w):.2%} < 阈值，继续躺平。")
    print(f"   -> 维持当前漂移仓位: 500 [{curr_drift_w:.2%}] / 红利 [{1 - curr_drift_w:.2%}]")
print("#" * 60)