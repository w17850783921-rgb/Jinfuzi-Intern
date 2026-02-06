import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.stats import percentileofscore  # 用于计算分位数
import os

# (请根据您的文件实际位置进行调整) ---
file_path = r"C:\Users\86178\Desktop\中证500和中证红利日收益率.xlsx"


# -----------------------------------------------------------------

# Helper function to segment time series by constant values
def get_segments(series):
    """
    将时间序列按连续不变的值进行分段
    """
    # Find points where the value changes and create cumulative sum groups
    change_points = series.ne(series.shift()).cumsum()
    return series.groupby(change_points)


def calculate_metrics(daily_returns, cumulative_values_plus_one):
    """
    计算最大回撤 (MDD) 和年化夏普比率 (Sharpe Ratio)。
    """
    # --- MDD Calculation ---
    peak = cumulative_values_plus_one.expanding().max()
    drawdown = cumulative_values_plus_one / peak - 1
    mdd = drawdown.min()

    # --- Sharpe Ratio Calculation (Rf=0, 252 trading days) ---
    avg_daily_return = daily_returns.mean()
    std_daily_return = daily_returns.std()
    sharpe = (avg_daily_return / std_daily_return) * np.sqrt(252) if std_daily_return > 0 else 0

    return mdd, sharpe


# --- 配置参数 (基于新策略逻辑修改) ---
START_DATE = '2022-01-01'
ROLLING_WINDOW = 40  # RTVR因子平滑窗口
LOOKBACK = 66  # 滚动计算阈值的周期 (用于分位数计算的历史窗口)

# **** 阈值和边界 ****
P_HIGH_THRESHOLD = 0.70  # 线性调仓起始阈值 (高)
P_LOW_THRESHOLD = 0.30  # 线性调仓起始阈值 (低)
P_FULL_HIGH = 0.90  # 满仓边界 (高)
P_FULL_LOW = 0.10  # 满仓边界 (低)

P_MID_HIGH = 0.60  # 中位重置区间的上限
P_MID_LOW = 0.40  # 中位重置区间的下限

COMMISSION = 0.0001  # 单边交易费用 0.01%
# -----------------------------------------------------------------

# --- 1. 数据加载与准备 ---
try:
    if file_path.lower().endswith('.xlsx'):
        df = pd.read_excel(file_path, parse_dates=['TradingDay'])
    else:
        df = pd.read_csv(file_path, parse_dates=['TradingDay'])
except Exception as e:
    print(f"无法读取文件，请检查文件路径和格式: {e}")
    exit()

df = df.set_index('TradingDay').sort_index()
df_test = df[df.index >= START_DATE].copy()

# 统一列名
df_test.rename(columns={
    'turnover_value1': 'TV_500',
    'turnover_value2': 'TV_HL',
    'index_return1': 'Ret_500',
    'index_return2': 'Ret_HL'
}, inplace=True)

# ----------------------------------------------------
# 2. 因子计算 (RTVR Factor) & 滚动分位数计算
# ----------------------------------------------------
df_test['RTVR'] = df_test['TV_500'] / (df_test['TV_500'] + df_test['TV_HL'])
df_test['RTVR_Factor'] = df_test['RTVR'].rolling(window=ROLLING_WINDOW).mean()


def calculate_percentile_rank(series, lookback):
    ranks = series.rolling(window=lookback).apply(
        lambda x: np.nan if len(x) < lookback else percentileofscore(x[:-1], x.iloc[-1]) / 100,
        raw=False
    )
    return ranks


df_test['Percentile_Rank'] = calculate_percentile_rank(df_test['RTVR_Factor'], LOOKBACK)

df_test.dropna(subset=['RTVR_Factor', 'Percentile_Rank'], inplace=True)
print(f"数据准备完成。回测起始日: {df_test.index.min().strftime('%Y-%m-%d')}")

# ----------------------------------------------------
# 3. 信号生成与回测模拟 (非反向线性调仓 + 动态趋势确认)
# ----------------------------------------------------

# 仓位初始化
df_test['Weight_500'] = np.nan
df_test['Weight_HL'] = np.nan
df_test['Trade_Flag'] = 0  # 1: 500增加, -1: 500减少, 2: 重置50/50
df_test['Strategy_Return'] = 0.0
df_test['Benchmark_Return'] = 0.5 * df_test['Ret_500'] + 0.5 * df_test['Ret_HL'] # 50/50 基准

first_valid_index = df_test.index[0]
df_test.loc[first_valid_index, 'Weight_500'] = 0.5
df_test.loc[first_valid_index, 'Weight_HL'] = 0.5

trades_log = []
last_trade_log_start_date = first_valid_index


def get_target_weight(P, p_low, p_high, p_full_low, p_full_high):
    """
    根据新的百分位数区间和线性/满仓规则计算目标仓位 W_500_target。
    注意: RTVR值越高，红利越强势，W_500 应该越低。
    """
    w_target = np.nan

    # 极高满仓区 (P > 0.90): W_HL = 1.0 -> W_500 = 0.0
    if P > p_full_high:
        w_target = 0.0

    # 极高线性区 (0.75 < P <= 0.90): W_500 从 0.5 线性降至 0.0
    elif P > p_high and P <= p_full_high:
        # Exceedance ratio: (P - 0.75) / (0.90 - 0.75)
        exceed_ratio = (P - p_high) / (p_full_high - p_high)
        # W_500 = 50% - (Exceedance Ratio * 50%)
        w_target = 0.50 - exceed_ratio * 0.50

    # 极低满仓区 (P < 0.10): W_500 = 1.0
    elif P < p_full_low:
        w_target = 1.0

    # 极低线性区 (0.10 <= P < 0.25): W_500 从 0.5 线性升至 1.0
    elif P >= p_full_low and P < p_low:
        # Exceedance ratio: (0.25 - P) / (0.25 - 0.10)
        exceed_ratio = (p_low - P) / (p_low - p_full_low)
        # W_500 = 50% + (Exceedance Ratio * 50%)
        w_target = 0.50 + exceed_ratio * 0.50

    # 中性区 (0.25 <= P <= 0.75)
    else:
        w_target = np.nan

    # 确保目标仓位在 [0, 1] 范围内
    return np.clip(w_target, 0.0, 1.0) if not np.isnan(w_target) else np.nan


for i in range(1, len(df_test)):
    current_date = df_test.index[i]

    # 前一日的仓位
    w_500_prev = df_test['Weight_500'].iloc[i - 1]

    # 以前一日的分位数决定今日的信号
    P = df_test['Percentile_Rank'].iloc[i - 1]

    w_500_target = w_500_prev
    is_traded = False
    trade_flag = 0

    # 1. 中位重置逻辑 (优先级最高)
    if P_MID_LOW <= P <= P_MID_HIGH:
        if abs(w_500_prev - 0.5) > 0.0001:
            w_500_target = 0.5
            trade_flag = 2

    # 2. 极值区动态调仓逻辑 (P > 0.75 或 P < 0.25)
    elif P > P_HIGH_THRESHOLD or P < P_LOW_THRESHOLD:

        is_trend_confirmed = False
        if i >= 3:
            p_curr_signal = df_test['Percentile_Rank'].iloc[i - 1]  # P(t-1)
            p_yday_signal = df_test['Percentile_Rank'].iloc[i - 2]  # P(t-2)
            p_2day_signal = df_test['Percentile_Rank'].iloc[i - 3]  # P(t-3)

            if P > P_HIGH_THRESHOLD:
                # 极高区 (看多红利, P值越大越好): 趋势必须是 P值连续三天大于前一天 (上升趋势)
                if p_curr_signal > p_yday_signal and p_yday_signal > p_2day_signal:
                    is_trend_confirmed = True

            elif P < P_LOW_THRESHOLD:
                # 极低区 (看多500, P值越小越好): 趋势必须是 P值连续三天小于前一天 (下降趋势)
                if p_curr_signal < p_yday_signal and p_yday_signal < p_2day_signal:
                    is_trend_confirmed = True

        # B. 趋势确认后，进行目标仓位计算
        if is_trend_confirmed:
            w_500_extreme_target = get_target_weight(
                P, P_LOW_THRESHOLD, P_HIGH_THRESHOLD, P_FULL_LOW, P_FULL_HIGH
            )

            if not np.isnan(w_500_extreme_target):

                # C. 非反向调整 (只增强，不减弱倾向)
                if P < P_LOW_THRESHOLD:
                    # 低极值区 (看多500, W_500 只能增加或保持)
                    # 目标是向 1.0 移动，所以只接受更大的 W_500
                    w_500_target = max(w_500_prev, w_500_extreme_target)
                    if w_500_target > w_500_prev:
                        trade_flag = 1 # 500增加

                elif P > P_HIGH_THRESHOLD:
                    # 高极值区 (看空500, 看多红利, W_500 只能减少或保持)
                    # 目标是向 0.0 移动，所以只接受更小的 W_500
                    w_500_target = min(w_500_prev, w_500_extreme_target)
                    if w_500_target < w_500_prev:
                        trade_flag = -1 # 500减少

        # 如果趋势确认失败，则 w_500_target 保持 w_500_prev (即 w_500_target = w_500_prev)

    # 3. 确定最终仓位和交易量
    w_500_curr = w_500_target
    w_hl_curr = 1.0 - w_500_curr

    # 交易量是仓位变化绝对值的两倍 (一买一卖)
    trade_amount = abs(w_500_curr - w_500_prev) * 2

    if trade_amount > 0.00001:
        is_traded = True

    df_test.loc[current_date, 'Trade_Flag'] = trade_flag

    # 4. 更新仓位
    df_test.loc[current_date, 'Weight_500'] = w_500_curr
    df_test.loc[current_date, 'Weight_HL'] = w_hl_curr

    # 5. 计算当日收益 (包含交易成本)
    daily_return = (w_500_curr * df_test['Ret_500'].iloc[i] +
                    w_hl_curr * df_test['Ret_HL'].iloc[i])
    cost = trade_amount * COMMISSION

    df_test.loc[current_date, 'Strategy_Return'] = daily_return - cost

    # 6. 记录交易胜率 (逻辑与之前保持一致)
    if is_traded:
        # 从上一次交易的开始日到当前日的前一日
        holding_period = df_test.loc[last_trade_log_start_date:df_test.index[i - 1]]
        if not holding_period.empty:
            trade_return_strategy = (1 + holding_period['Strategy_Return']).prod() - 1
            trade_return_benchmark = (1 + holding_period['Benchmark_Return']).prod() - 1
            # 策略收益高于基准收益（考虑到微小误差）即为胜利
            trades_log.append(trade_return_strategy >= trade_return_benchmark - 0.0002)
        last_trade_log_start_date = df_test.index[i]

# 计算最后一笔交易的胜率
last_hold = df_test.loc[last_trade_log_start_date:]
if not last_hold.empty and len(last_hold) > 1:
    trade_return_strategy = (1 + last_hold['Strategy_Return']).prod() - 1
    trade_return_benchmark = (1 + last_hold['Benchmark_Return']).prod() - 1
    trades_log.append(trade_return_strategy > trade_return_benchmark)

# ----------------------------------------------------
# 4. 绩效指标计算与绘图 (新增两基准)
# ----------------------------------------------------

# 1. 累计收益率计算
df_test['Strategy_Cumulative'] = (1 + df_test['Strategy_Return']).cumprod()
df_test['Benchmark_Cumulative'] = (1 + df_test['Benchmark_Return']).cumprod() # 50/50 基准
# 新增两条基准
df_test['Full_500_Cumulative'] = (1 + df_test['Ret_500']).cumprod()
df_test['Full_HL_Cumulative'] = (1 + df_test['Ret_HL']).cumprod()


# 2. 超额收益率和累计超额收益
df_test['Excess_Return'] = df_test['Strategy_Return'] - df_test['Benchmark_Return']
df_test['Excess_Cumulative'] = (1 + df_test['Excess_Return']).cumprod() - 1

# 3. 交易指标 (基于超额收益的胜率)
trade_count = len(trades_log)
winning_trades = sum(trades_log)
win_rate = winning_trades / trade_count if trade_count > 0 else 0

# 4. 最终收益计算 & 年化收益率
days = (df_test.index.max() - df_test.index.min()).days
T = 365.25 / days # 年化因子

cumulative_return_strategy = df_test['Strategy_Cumulative'].iloc[-1] - 1
cumulative_return_benchmark = df_test['Benchmark_Cumulative'].iloc[-1] - 1
cumulative_return_500 = df_test['Full_500_Cumulative'].iloc[-1] - 1
cumulative_return_HL = df_test['Full_HL_Cumulative'].iloc[-1] - 1
cumulative_excess_return = df_test['Excess_Cumulative'].iloc[-1]

ann_return_strategy = ((1 + cumulative_return_strategy) ** T) - 1
ann_return_benchmark = ((1 + cumulative_return_benchmark) ** T) - 1
ann_return_500 = ((1 + cumulative_return_500) ** T) - 1
ann_return_HL = ((1 + cumulative_return_HL) ** T) - 1


# 5. 绩效指标 (MDD 和 Sharpe)
max_drawdown_strategy, annualized_sharpe_strategy = calculate_metrics(
    df_test['Strategy_Return'], df_test['Strategy_Cumulative']
)
max_drawdown_benchmark, annualized_sharpe_benchmark = calculate_metrics(
    df_test['Benchmark_Return'], df_test['Benchmark_Cumulative']
)
max_drawdown_excess, annualized_sharpe_excess = calculate_metrics(
    df_test['Excess_Return'], df_test['Excess_Cumulative'] + 1
)
# 新增两条基准的指标
max_drawdown_500, annualized_sharpe_500 = calculate_metrics(
    df_test['Ret_500'], df_test['Full_500_Cumulative']
)
max_drawdown_HL, annualized_sharpe_HL = calculate_metrics(
    df_test['Ret_HL'], df_test['Full_HL_Cumulative']
)

# 6. 绘图 (三图并列 - 绘图逻辑与之前保持一致)
fig = plt.figure(figsize=(14, 15))

# --- Subplot 1: Strategy Cumulative Return with Background Fill (新增两条线)---
ax1 = plt.subplot(3, 1, 1)

weight_fill_map = {
    1.0: ('red', 0.15), 0.9: ('red', 0.13), 0.8: ('red', 0.10), 0.7: ('red', 0.07), 0.6: ('red', 0.04),
    0.5: ('lightgray', 0.1),
    0.4: ('green', 0.04), 0.3: ('green', 0.07), 0.2: ('green', 0.10), 0.1: ('green', 0.13), 0.0: ('green', 0.15)
}


def get_weight_for_fill(weight):
    if weight == 0.5: return 0.5
    if 0 <= weight <= 1: return round(weight * 10) / 10
    return 0.5


df_test['Weight_Fill_Key'] = df_test['Weight_500'].apply(get_weight_for_fill)
weight_labels = {k: f'CSI 500 Weight: {k * 100:.0f}%' for k in weight_fill_map}

weights_plotted = set()
legend_handles = []
legend_labels = []
key_weights_to_plot = {1.0, 0.5, 0.0}

for group_id, segment in get_segments(df_test['Weight_Fill_Key']):
    weight = segment.iloc[0]

    if weight in weight_fill_map:
        color, alpha = weight_fill_map[weight]
        label_text = weight_labels[weight]

        start_date_num = mdates.date2num(segment.index.min())
        end_date_num = mdates.date2num(segment.index.max())

        if weight in key_weights_to_plot and weight not in weights_plotted:
            patch = plt.Rectangle((0, 0), 1, 1, fc=color, alpha=alpha)
            legend_handles.append(patch)
            legend_labels.append(label_text)
            weights_plotted.add(weight)

        ax1.axvspan(xmin=start_date_num, xmax=end_date_num + 0.5, color=color, alpha=alpha, zorder=1)

df_test['Strategy_Cumulative'].plot(ax=ax1, label='Strategy Cumulative Return', color='tab:blue', linewidth=2.5,
                                    zorder=3)
legend_handles.append(ax1.lines[-1])
legend_labels.append('Strategy Cumulative Return')

df_test['Benchmark_Cumulative'].plot(ax=ax1, label='Benchmark (50/50 Fixed)', color='black', linestyle='--',
                                     linewidth=1.5, zorder=2)
legend_handles.append(ax1.lines[-1])
legend_labels.append('Benchmark (50/50 Fixed)')

# 新增两条基准曲线
df_test['Full_500_Cumulative'].plot(ax=ax1, label='Benchmark (Full CSI 500)', color='purple', linestyle=':', linewidth=1.5, zorder=2)
legend_handles.append(ax1.lines[-1])
legend_labels.append('Benchmark (Full CSI 500)')

df_test['Full_HL_Cumulative'].plot(ax=ax1, label='Benchmark (Full Dividend)', color='orange', linestyle=':', linewidth=1.5, zorder=2)
legend_handles.append(ax1.lines[-1])
legend_labels.append('Benchmark (Full Dividend)')


ax1.set_title('Strategy Cumulative Return & CSI 500 Weight (Background Fill)', fontsize=16)
ax1.set_ylabel('Cumulative Value (Start=1)', fontsize=12)
ax1.set_xlabel('Date', fontsize=12)
ax1.legend(handles=legend_handles, labels=legend_labels, fontsize=10, loc='upper left')
ax1.grid(True, linestyle='--', alpha=0.7)

# --- Subplot 2: Cumulative Excess Return ---
ax2 = plt.subplot(3, 1, 2)
df_test['Excess_Cumulative'].plot(label='Excess Return (Strategy - 50/50 Benchmark)', color='tab:green', ax=ax2)
ax2.axhline(0, color='red', linestyle=':', linewidth=1)
ax2.set_title('Cumulative Excess Return (vs 50/50 Benchmark)', fontsize=16)
ax2.set_xlabel('Date', fontsize=12)
ax2.set_ylabel('Cumulative Excess Return (Start=0)', fontsize=12)
ax2.legend(fontsize=10)
ax2.grid(True, linestyle='--', alpha=0.7)

# --- Subplot 3: RTVR Factor & Historical Percentile ---
ax3 = plt.subplot(3, 1, 3)
df_test['RTVR_Factor'].plot(ax=ax3, label='RTVR Factor (38-Day MA)', color='tab:blue', linewidth=1.5)

ax4 = ax3.twinx()
df_test['Percentile_Rank'].plot(ax=ax4, label='Historical Percentile Rank (75-Day Lookback)', color='tab:red',
                                linestyle=':', linewidth=1.5)


h3, l3 = ax3.get_legend_handles_labels()
h4, l4 = ax4.get_legend_handles_labels()
ax4.legend(h3 + h4, l3 + l4, loc='upper left', fontsize=10)

ax3.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()

# 7. 打印最终报告 (更新报告表格)
print("\n--- RTVR 风格轮动策略回测报告 (动态趋势确认 + 非反向调整) ---")
print(f"回测区间: {df_test.index.min().strftime('%Y-%m-%d')} 至 {df_test.index.max().strftime('%Y-%m-%d')}")
print(f"线性调仓区间: 70%-90% (减 500) 和 10%-30% (加 500). 满仓边界: 90% (红利) 和 10% (500).")
print(f"极值调仓策略: **P>0.70需连续3天上升**，**P<0.30需连续3天下降**，且仅进行**非反向调整**（增强倾向）。")
print(f"中位调仓策略: 落在 {P_MID_LOW * 100:.0f}%-{P_MID_HIGH * 100:.0f}% 区间时，**即时重置** 50/50")
print(f"总交易次数: {trade_count}")
print("-" * 90)
print("             | 策略收益 | 基准(50/50)| 全仓500 | 全仓红利 | 超额收益")
print("-" * 90)
print(
    f"累计收益率  | {cumulative_return_strategy * 100:8.2f}% | {cumulative_return_benchmark * 100:8.2f}% | {cumulative_return_500 * 100:8.2f}% | {cumulative_return_HL * 100:8.2f}% | {cumulative_excess_return * 100:8.2f}%")
print(f"年化收益率  | {ann_return_strategy * 100:8.2f}% | {ann_return_benchmark * 100:8.2f}% | {ann_return_500 * 100:8.2f}% | {ann_return_HL * 100:8.2f}% | {'-':8s}")
print(
    f"最大回撤率  | {max_drawdown_strategy * 100:8.2f}% | {max_drawdown_benchmark * 100:8.2f}% | {max_drawdown_500 * 100:8.2f}% | {max_drawdown_HL * 100:8.2f}% | {max_drawdown_excess * 100:8.2f}%")
print(
    f"年化夏普比率| {annualized_sharpe_strategy:8.2f} | {annualized_sharpe_benchmark:8.2f} | {annualized_sharpe_500:8.2f} | {annualized_sharpe_HL:8.2f} | {annualized_sharpe_excess:8.2f}")
print("-" * 90)
print(f"交易胜率 (基于超额): {win_rate * 100:.2f}%")