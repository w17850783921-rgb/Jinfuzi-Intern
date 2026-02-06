import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# ----------------------------------------------------------------------
# 📌 0. 全局设置
# ----------------------------------------------------------------------
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 🌟🌟🌟 用户核心参数修改区 🌟🌟🌟
# 1. 调仓速度控制 (自适应)
MIN_DAILY_STEP = 0.01  # 基础步长：每天只调 1% (极慢，防止震荡磨损)
SLOPE_SENSITIVITY = 30  # 斜率敏感度：建议 20~30。数值越大，爆发期追得越快。
# 计算公式：今日步长 = MIN + (斜率绝对值 * SENSITIVITY)

# 2. 过滤器与成本
VOL_THRESHOLD = 1  # 量能门槛 (VR)：VR > 1 才允许进攻
COST = 0.0001  # 单边交易成本 (1bp)

# ----------------------------------------------------------------------
# --- 1. 数据加载 (省略，假设数据已成功加载到 df) ---
# ----------------------------------------------------------------------
file_path = r"C:\Users\86178\Desktop\交易情绪因子.csv"

try:
    df = pd.read_csv(file_path)
    print(f"✅ 成功加载数据: {file_path}")
except FileNotFoundError:
    print(f"❌ 错误：未能找到文件，请检查路径: {file_path}")
    exit()

# 数据预处理
df['TradingDay'] = pd.to_datetime(df['TradingDay'])
df = df.set_index('TradingDay').sort_index()

# ----------------------------------------------------------------------
# 🆕 2. 成交量 (VR) 因子计算 (省略，与原代码相同)
# ----------------------------------------------------------------------
try:
    df['MA20_Vol_500'] = df['turnover_value1'].rolling(window=20).mean()
    df['MA20_Vol_HL'] = df['turnover_value2'].rolling(window=20).mean()
    df['VR_500'] = df['turnover_value1'] / df['MA20_Vol_500']
    df['VR_HL'] = df['turnover_value2'] / df['MA20_Vol_HL']
    df['VR_500'] = df['VR_500'].fillna(1.0)
    df['VR_HL'] = df['VR_HL'].fillna(1.0)
except KeyError:
    df['VR_500'] = 100.0
    df['VR_HL'] = 100.0


# ----------------------------------------------------------------------
# --- 3. TSM 因子与斜率计算 (省略，与原代码相同) ---
# ----------------------------------------------------------------------
def calculate_tsm(data, prefix, otsm_window=69, dtsm_window=3):
    prev_close = data[f'prev_close{prefix}']
    open_price = data[f'open_price{prefix}']
    high_price = data[f'high_price{prefix}']
    low_price = data[f'low_price{prefix}']
    range_hl = high_price - low_price
    ot_sm = np.where(range_hl != 0, (high_price - prev_close) / range_hl, 0)
    tsm_otsm = pd.Series(ot_sm, index=data.index).rolling(window=otsm_window, min_periods=otsm_window).mean()
    dt_sm = np.where(range_hl != 0, (high_price - open_price) / range_hl, 0)
    tsm_dtsm = pd.Series(dt_sm, index=data.index).rolling(window=dtsm_window, min_periods=dtsm_window).mean()
    return 0.5 * tsm_otsm + 0.5 * tsm_dtsm


df['Return_500'] = df['index_return1']
df['Return_HL'] = df['index_return2']
df['TSM_500'] = calculate_tsm(df, '1')
df['TSM_HL'] = calculate_tsm(df, '2')
df['TSM_Relative'] = df['TSM_500'] - df['TSM_HL']
EWMA_SPAN = 25
df['TSM_Relative_Smooth'] = df['TSM_Relative'].ewm(span=EWMA_SPAN, adjust=False).mean()
df['Factor_Slope'] = df['TSM_Relative_Smooth'].diff()
df['Slope_Sign'] = np.sign(df['Factor_Slope']).fillna(0)
df['Slope_Abs'] = df['Factor_Slope'].abs().fillna(0)


# ----------------------------------------------------------------------
# --- 4. 信号生成 (省略，与原代码相同) ---
# ----------------------------------------------------------------------
def generate_signal(data, window=3, upper_thres=0.04, lower_thres=-0.04, vol_thres=1):
    signals = pd.Series(np.nan, index=data.index)
    signals.iloc[0] = 3
    slope_signs = data['Slope_Sign'].values
    factor_values = data['TSM_Relative_Smooth'].values
    vr_500 = data['VR_500'].values
    vr_hl = data['VR_HL'].values
    signal_arr = signals.values

    for i in range(len(data)):
        current = signal_arr[i - 1] if i > 0 and not np.isnan(signal_arr[i - 1]) else 3
        signal_arr[i] = current

        if i < window - 1: continue

        recent_slopes = slope_signs[i - window + 1: i + 1]
        is_pos = np.all(recent_slopes == 1)
        is_neg = np.all(recent_slopes == -1)
        val = factor_values[i]

        # 1. 进攻 500 (信号 1)
        if val > upper_thres and is_pos and vr_500[i] > vol_thres:
            signal_arr[i] = 1
        # 2. 进攻 红利 (信号 2)
        elif val < lower_thres and is_neg and vr_hl[i] > vol_thres:
            signal_arr[i] = 2
        # 3. 防守/中性 (信号 3)
        elif (val > upper_thres and is_neg) or (val < lower_thres and is_pos):
            signal_arr[i] = 3

    return pd.Series(signal_arr, index=data.index).shift(1).astype('float64')


df['Signal'] = generate_signal(df, window=3, upper_thres=0.04, lower_thres=-0.04, vol_thres=VOL_THRESHOLD)

# ----------------------------------------------------------------------
# 🌟 5. 自适应斜率调仓逻辑 (Adaptive Slope Rebalancing) 【已引入漂移】
# ----------------------------------------------------------------------
# 设定目标仓位
df['Target_W_500'] = np.where(df['Signal'] == 1, 1.0,
                              np.where(df['Signal'] == 2, 0.0, 0.5))

w_500_actual = pd.Series(np.nan, index=df.index)
target_values = df['Target_W_500'].values
slope_abs_values = df['Slope_Abs'].values
return_500_values = df['Return_500'].values
return_hl_values = df['Return_HL'].values

# 初始化：T-1 日收盘仓位 (净值平均法)
# 我们需要一个变量来存储每天**收盘后**的仓位，即考虑了当日收益漂移后的仓位。
w_500_close = 0.5  # 假设回测起始日 T-1 收盘是 0.5

for i in range(1, len(df)):
    target_w = target_values[i]  # T 日目标仓位
    current_slope = slope_abs_values[i]  # T 日斜率

    # 1. T 日**开盘仓位** = T-1 日**收盘仓位** (考虑了漂移)
    # 这一步是关键，它将昨天的收益效应带入今天的起始仓位
    w_500_start_of_day = w_500_close

    # --- 计算今日调仓步长 (Step) ---
    if target_w == 0.5:
        # 场景 A: 目标是中性 (0.5) -> 此时为平仓避险，必须一步到位 (调仓步长设为最大)
        step_size = 1.0
    else:
        # 场景 B: 目标是进攻 (1.0 或 0.0) -> 根据斜率自适应速度
        dynamic_boost = current_slope * SLOPE_SENSITIVITY
        step_size = MIN_DAILY_STEP + dynamic_boost
        step_size = min(step_size, 1.0)

    # --- 2. T 日执行仓位调整 (基于 T 日开盘仓位和步长) ---
    w_500_trade = w_500_start_of_day
    if w_500_trade < target_w:
        w_500_trade = min(w_500_trade + step_size, target_w)
    elif w_500_trade > target_w:
        w_500_trade = max(w_500_trade - step_size, target_w)

    w_500_actual.iloc[i] = w_500_trade  # 记录 T 日的交易仓位

    # --- 3. T 日收盘仓位 (计算漂移) ---
    # 策略 T 日收益（毛收益）
    R_strategy = w_500_trade * return_500_values[i] + (1.0 - w_500_trade) * return_hl_values[i]

    # 计算**仓位漂移**：考虑当日收益后，新的仓位比例
    # W_t+1, Close = W_t, Trade * (1 + R_500) / (1 + R_strategy)
    w_500_close = w_500_trade * (1 + return_500_values[i]) / (1 + R_strategy)

    # 防止因极端收益导致仓位超过 1 或低于 0 (理论上不会发生，作为保护)
    w_500_close = np.clip(w_500_close, 0.0, 1.0)

# 对齐仓位 (T日持仓是 T 日交易后的仓位)
df['W_500'] = w_500_actual.fillna(0.5)  # T 日的实际持仓（交易后的仓位）
df['W_HL'] = 1.0 - df['W_500']

# 收益与成本计算
# **注意：成本计算必须使用 T 日交易仓位与 T-1 日交易仓位/初始仓位之间的差额**
# T 日换手率 = |W_t, trade - W_t-1, close| * 2 (W_500 & W_HL)
# 但由于我们只记录 W_t, trade，我们只能近似计算：
# W_t-1 trade 是昨天的 W_500.shift(1).
df['Turnover'] = (np.abs(df['W_500'] - df['W_500'].shift(1)) +
                  np.abs(df['W_HL'] - df['W_HL'].shift(1)))
df['Transaction_Cost'] = (df['Turnover'] * COST).fillna(0)
df['Strategy_Return_Gross'] = df['W_500'] * df['Return_500'] + df['W_HL'] * df['Return_HL']
df['Strategy_Return_Net'] = df['Strategy_Return_Gross'] - df['Transaction_Cost']

# ----------------------------------------------------------------------
# ⚡ 基准改进：净值平均法 (Buy and Hold)
# ----------------------------------------------------------------------
# 假设 T=0 时，投资 50% 在 500 上，50% 在 HL 上，之后不再调整
nav_500 = (1 + df['Return_500']).cumprod()
nav_hl = (1 + df['Return_HL']).cumprod()
# 基准净值 = 0.5 * 500 净值 + 0.5 * 红利净值
benchmark_nav = 0.5 * nav_500 + 0.5 * nav_hl
df['Benchmark_Return'] = benchmark_nav.pct_change().fillna(0.0)

# ----------------------------------------------------------------------
# --- 6. 绩效统计 (包含所有您要求的指标) (省略，与原代码相同，但计算基于新基准) ---
# ----------------------------------------------------------------------
min_valid_index = max(90, EWMA_SPAN)
df_backtest = df.iloc[min_valid_index:].dropna(subset=['Signal', 'Strategy_Return_Net']).copy()

if df_backtest.empty:
    print("❌ 错误：数据长度不足")
    exit()

days_in_backtest = len(df_backtest)
trading_days_per_year = 252

# 基础收益指标
df_backtest['Strategy_Cumulative_Return'] = (1 + df_backtest['Strategy_Return_Net']).cumprod()
df_backtest['Benchmark_Cumulative_Return'] = (1 + df_backtest['Benchmark_Return']).cumprod()
df_backtest['Excess_Cumulative_Return'] = df_backtest['Strategy_Cumulative_Return'] / df_backtest[
    'Benchmark_Cumulative_Return']

strategy_total_return = df_backtest['Strategy_Cumulative_Return'].iloc[-1] - 1
strategy_annualized_return = ((1 + strategy_total_return) ** (trading_days_per_year / days_in_backtest) - 1)
benchmark_total_return = df_backtest['Benchmark_Cumulative_Return'].iloc[-1] - 1
benchmark_annualized_return = ((1 + benchmark_total_return) ** (trading_days_per_year / days_in_backtest) - 1)
excess_return = strategy_annualized_return - benchmark_annualized_return

cumulative_max = df_backtest['Strategy_Cumulative_Return'].cummax()
max_drawdown = ((cumulative_max - df_backtest['Strategy_Cumulative_Return']) / cumulative_max).max()
sharpe_ratio = strategy_annualized_return / (df_backtest['Strategy_Return_Net'].std() * np.sqrt(trading_days_per_year))

# ==============================================================================
# 📊 统计指标 A: 交易基础数据 (修正版：按调仓区间计算胜率)
# ==============================================================================
# 1. 实际有调仓动作的天数
# 找出所有发生交易（Turnover > 0）的行索引
trade_indices = np.where(df_backtest['Turnover'] > 0.000001)[0]
trades_days_count = len(trade_indices)

# 2. 调仓区间胜率 (Trade Interval Win Rate)
# 定义：从第 T 次调仓日(含)开始，持有直到第 T+1 次调仓日(前一日)结束。
# 逻辑：衡量这一次动作确定的仓位，在下一次变动前是否跑赢了基准。

interval_wins = 0
total_intervals = 0

if trades_days_count > 0:
    # 遍历所有调仓日（除了最后一个）
    for k in range(len(trade_indices) - 1):
        start_idx = trade_indices[k]
        end_idx = trade_indices[k + 1]  # 下一次调仓的索引

        # 截取区间：包含 start_idx，不包含 end_idx
        # 意味着：评估从这次调仓生效开始，直到下次调仓改变仓位之前的所有日子
        interval_df = df_backtest.iloc[start_idx: end_idx]

        if not interval_df.empty:
            # 计算区间累计收益
            strat_cum = (1 + interval_df['Strategy_Return_Net']).prod() - 1
            bench_cum = (1 + interval_df['Benchmark_Return']).prod() - 1

            # 判断胜负 (引入微小阈值防止浮点误差)
            if strat_cum >= bench_cum - 0.0002:
                interval_wins += 1
            total_intervals += 1

    # 处理最后一次调仓：从最后一次调仓持有到回测结束
    last_start_idx = trade_indices[-1]
    last_interval_df = df_backtest.iloc[last_start_idx:]

    if not last_interval_df.empty:
        strat_cum = (1 + last_interval_df['Strategy_Return_Net']).prod() - 1
        bench_cum = (1 + last_interval_df['Benchmark_Return']).prod() - 1
        if strat_cum >= bench_cum - 0.0002:
            interval_wins += 1
        total_intervals += 1

# 计算最终胜率
trade_interval_win_rate = interval_wins / total_intervals if total_intervals > 0 else 0.0

# ==============================================================================
# 📊 统计指标 B/C: 交易次数及胜率 (逻辑不变，基于新的收益数据)
# ==============================================================================
# B. 总调仓次数 (含中性)
trades_log_all = []
last_trade_start_idx_all = 0
target_signals = df_backtest['Target_W_500'].values

for i in range(1, len(df_backtest)):
    if target_signals[i] != target_signals[i - 1]:
        holding_slice = df_backtest.iloc[last_trade_start_idx_all: i]
        if len(holding_slice) > 0:
            s_ret = (1 + holding_slice['Strategy_Return_Net']).prod() - 1
            b_ret = (1 + holding_slice['Benchmark_Return']).prod() - 1
            trades_log_all.append(s_ret >= b_ret - 0.0002)
        last_trade_start_idx_all = i

if last_trade_start_idx_all < len(df_backtest):
    holding_slice = df_backtest.iloc[last_trade_start_idx_all:]
    if len(holding_slice) > 0:
        s_ret = (1 + holding_slice['Strategy_Return_Net']).prod() - 1
        b_ret = (1 + holding_slice['Benchmark_Return']).prod() - 1
        trades_log_all.append(s_ret >= b_ret - 0.0002)

total_trades_all = len(trades_log_all)
win_rate_all = sum(trades_log_all) / total_trades_all if total_trades_all > 0 else 0.0

# C. 主动波段交易胜率 (剔除中性)
trades_log_active = []
last_trade_start_idx_active = 0
raw_signals = df_backtest['Signal'].values

for i in range(1, len(df_backtest)):
    if target_signals[i] != target_signals[i - 1]:
        prev_signal = raw_signals[i - 1]
        if prev_signal in [1, 2]:
            holding_slice = df_backtest.iloc[last_trade_start_idx_active: i]
            if len(holding_slice) > 0:
                s_ret = (1 + holding_slice['Strategy_Return_Net']).prod() - 1
                b_ret = (1 + holding_slice['Benchmark_Return']).prod() - 1
                trades_log_active.append(s_ret >= b_ret - 0.0002)
        last_trade_start_idx_active = i

if last_trade_start_idx_active < len(df_backtest):
    last_signal = raw_signals[-1]
    if last_signal in [1, 2]:
        holding_slice = df_backtest.iloc[last_trade_start_idx_active:]
        if len(holding_slice) > 0:
            s_ret = (1 + holding_slice['Strategy_Return_Net']).prod() - 1
            b_ret = (1 + holding_slice['Benchmark_Return']).prod() - 1
            trades_log_active.append(s_ret >= b_ret - 0.0002)

total_active_trades = len(trades_log_active)
win_rate_active = sum(trades_log_active) / total_active_trades if total_active_trades > 0 else 0.0

# ==============================================================================
# 📊 统计指标 D: IC (符号IC) (逻辑不变)
# ==============================================================================
factor_diff = df_backtest['TSM_Relative']
return_diff = df_backtest['Return_500'] - df_backtest['Return_HL']
factor_sign = np.sign(factor_diff).shift(1).fillna(0)
ic_df = pd.DataFrame({'Factor_Sign': factor_sign, 'Return_Diff_Sign': np.sign(return_diff)},
                     index=df_backtest.index).dropna()
ic_df['IC_Daily'] = ic_df['Factor_Sign'] * ic_df['Return_Diff_Sign']
ic_mean = ic_df['IC_Daily'].mean()
icir = ic_mean / ic_df['IC_Daily'].std() * np.sqrt(trading_days_per_year)

# ----------------------------------------------------------------------
# --- 7. 结果展示 (全指标输出) (基于新的计算结果) ---
# ----------------------------------------------------------------------
print("\n" + "=" * 40)
print(f" 🚀 TSM 策略报告 (修正版: 净值平均基准 + 仓位漂移) 🚀")
print("=" * 40)
print(f"**基准模式**: 50/50 买入持有 (Buy and Hold)")
print(f"**策略模式**: 引入仓位漂移 (Drift)")
print("-" * 40)
print("📈 收益指标")
print(f"策略累计收益率:      {strategy_total_return:.2%}")
print(f"基准累计收益率:      {benchmark_total_return:.2%}")
print(f"超额累计收益率:      {df_backtest['Excess_Cumulative_Return'].iloc[-1] - 1:.2%}")
print(f"策略年化收益率:        {strategy_annualized_return:.2%}")
print(f"超额收益率（年化）:    {excess_return:.2%}")
print("-" * 40)
print("🛡️ 风险/风控指标")
print(f"最大回撤率:          {max_drawdown:.2%}")
print(f"夏普比率 (Rf=0):      {sharpe_ratio:.2f}")
print("-" * 40)
print("💡 因子表现 (符号IC)")
print(f"因子日IC均值:        {ic_mean:.4f}")
print(f"因子ICIR:            {icir:.2f}")
print("-" * 40)
print("🔄 交易指标 (ALL STATISTICS)")
print(f"1. 实际调仓动作天数:    {int(trades_days_count)} 天")
print(f"2. 调仓区间胜率:        {trade_interval_win_rate:.2%} (持有至下次调仓)") # 修改了这里
print(f"3. 日均换手率:          {df_backtest['Turnover'].mean():.2%}")
print(f"---")
print(f"4. 总调仓次数 (含中性):  {total_trades_all} 次")
print(f"5. 总调仓胜率 (含中性):  {win_rate_all:.2%}")
print(f"---")
print(f"6. 主动波段次数 (仅进攻): {total_active_trades} 次")
print(f"7. 主动波段胜率 (仅进攻): {win_rate_active:.2%}")
print("=" * 40)

# ----------------------------------------------------------------------
# --- 8. 可视化 (4 个子图) (与原代码相同，但基于新的计算结果) ---
# ----------------------------------------------------------------------
fig, axes = plt.subplots(4, 1, figsize=(12, 16), sharex=True)

# 1. 收益
axes[0].plot(df_backtest['Strategy_Cumulative_Return'], label='TSM 自适应策略 (Drift)', linewidth=2)
axes[0].plot(df_backtest['Benchmark_Cumulative_Return'], label='50/50 买入持有基准', linestyle='--', alpha=0.7)
axes[0].set_title(f'策略 vs 基准 (Sensitivity={SLOPE_SENSITIVITY})', fontsize=14)
axes[0].grid(True, linestyle='--', alpha=0.6)
axes[0].legend()

# 2. 超额
axes[1].plot(df_backtest['Excess_Cumulative_Return'], label='超额累计收益率', color='blue', linewidth=2)
axes[1].axhline(1.0, color='red', linestyle='--', linewidth=1.0)
axes[1].set_title('超额累计收益率', fontsize=14)
axes[1].grid(True, linestyle='--', alpha=0.6)

# 3. 仓位与因子
ax3 = axes[2]
ax3.plot(df_backtest['TSM_Relative_Smooth'], label='TSM 因子', color='green', linewidth=1.0, alpha=0.6)
ax3.axhline(0.05, color='orange', linestyle='--')
ax3.axhline(-0.05, color='purple', linestyle='--')
ax3.set_ylabel('因子值')
ax3_right = ax3.twinx()
ax3_right.plot(df_backtest.index, df_backtest['W_500'], label='实际仓位', color='blue', linewidth=2)
ax3_right.plot(df_backtest.index, df_backtest['Target_W_500'], label='目标信号', color='blue', linestyle='--',
               alpha=0.3)
ax3_right.set_ylabel('仓位')
ax3.legend(loc='upper left')
ax3_right.legend(loc='upper right')
ax3.set_title('因子 vs 实际仓位 (观察自适应速度)', fontsize=14)
ax3.grid(True, linestyle='--', alpha=0.6)

# 4. 斜率监控
axes[3].bar(df_backtest.index, df_backtest['Slope_Abs'], label='因子斜率 (绝对值)', color='gray', alpha=0.5)
axes[3].axhline(df_backtest['Slope_Abs'].mean() * 2, color='red', linestyle=':', label='2倍均值线')
axes[3].set_title('因子斜率监控 (斜率大=调仓快)', fontsize=14)
axes[3].legend()
axes[3].grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show()