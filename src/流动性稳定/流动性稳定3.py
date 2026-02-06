import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


# ----------------------------------------------------------------------
# 📌 1. 配置模块 (Configuration) - 最终定稿版
# ----------------------------------------------------------------------
class Config:
    # 🌟 文件路径
    FILE_PATH = r"C:\Users\86178\Desktop\交易情绪因子1.csv"

    # 🌟 回测时间
    START_DATE = '2023-01-01'
    END_DATE = '2099-12-31'

    # 🌟 因子参数 (Step 1 验证的最佳组合)
    STD_WINDOW = 126  # 核心周期：半年趋势
    RANK_WINDOW = 60  # 灵敏度：一季度自适应
    SMOOTH_WINDOW = 5  # 平滑窗口

    # 🌟 线性调仓阈值 (Step 2 验证的最佳组合 G组)
    # 逻辑：因子稍微偏离中枢，立即介入并锁定趋势
    LINEAR_HIGH = 0.70  # Rank > 0.6 开始减仓
    LINEAR_LOW = 0.30  # Rank < 0.4 开始加仓

    # 标配区 (无缝衔接)
    NEUTRAL_L = 0.40
    NEUTRAL_H = 0.60

    # 🌟 单边棘轮 (核心的大功臣)
    ENABLE_RATCHET = True

    # 🌟 交易成本
    COST = 0.0002
    SLIPPAGE = 0.0003


# 全局绘图设置
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


# ----------------------------------------------------------------------
# 📌 2. 数据加载与清洗
# ----------------------------------------------------------------------
def load_and_clean_data(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"❌ 错误：找不到文件 {file_path}")

    try:
        df = pd.read_csv(file_path, parse_dates=['TradingDay'])
        df = df.set_index('TradingDay').sort_index()
        print(f"✅ 数据加载成功: {len(df)} 条记录")
    except Exception as e:
        raise ValueError(f"❌ 数据读取失败: {e}")

    req_cols = ['turnover_value1', 'turnover_value2']
    if not all(col in df.columns for col in req_cols):
        raise ValueError(f"❌ 缺少必要列: {req_cols}")

    cols_price = ['close_price4', 'prev_close4', 'avg_price4',
                  'close_price5', 'prev_close5', 'avg_price5']
    for col in cols_price:
        if col in df.columns:
            df[col] = df[col].replace(0, np.nan)
    df[cols_price] = df[cols_price].ffill().bfill()

    return df


# ----------------------------------------------------------------------
# 📌 3. 因子计算引擎 (归一化修正版)
# ----------------------------------------------------------------------
def calculate_factor(df, cfg):
    print(f"🛠️ 计算因子 (Std={cfg.STD_WINDOW}, Rank={cfg.RANK_WINDOW})...")

    tv_500 = df['turnover_value1'].replace(0, np.nan).ffill()
    tv_hl = df['turnover_value2'].replace(0, np.nan).ffill()

    ln_tv_500 = np.log(tv_500)
    ln_tv_hl = np.log(tv_hl)

    std_500 = ln_tv_500.rolling(window=cfg.STD_WINDOW).std()
    std_hl = ln_tv_hl.rolling(window=cfg.STD_WINDOW).std()

    factor_raw = std_500 - std_hl
    factor_smooth = factor_raw.rolling(cfg.SMOOTH_WINDOW).mean()

    # 归一化 Rank [0, 1]
    raw_rank = factor_smooth.rolling(cfg.RANK_WINDOW).rank(pct=False)
    factor_rank = (raw_rank - 1) / (cfg.RANK_WINDOW - 1)

    res = df.copy()
    res['Factor_Rank'] = factor_rank
    return res


# ----------------------------------------------------------------------
# 📌 4. 信号生成器 (动态分母 + 棘轮)
# ----------------------------------------------------------------------
def generate_signals(df, cfg):
    print(f"🚦 生成信号 (High={cfg.LINEAR_HIGH}, Low={cfg.LINEAR_LOW}, Ratchet={cfg.ENABLE_RATCHET})")
    targets = []

    prev_w_base = 0.5
    prev_w_final = 0.5

    rank_values = df['Factor_Rank'].values

    # 动态分母计算
    denom_high = 1.0 - cfg.LINEAR_HIGH
    denom_low = cfg.LINEAR_LOW

    for rank in rank_values:
        if np.isnan(rank):
            curr_w = 0.5

        # 1. 标配区 [0.4, 0.6]
        elif cfg.NEUTRAL_L <= rank <= cfg.NEUTRAL_H:
            curr_w = 0.5

        # 2. 减仓区 (Rank > 0.6)
        elif rank >= cfg.LINEAR_HIGH:
            progress = (rank - cfg.LINEAR_HIGH) / denom_high
            curr_w = 0.5 - (progress * 0.5)
            curr_w = max(0.0, curr_w)

            # 3. 加仓区 (Rank < 0.4)
        elif rank <= cfg.LINEAR_LOW:
            progress = (cfg.LINEAR_LOW - rank) / denom_low
            curr_w = 0.5 + (progress * 0.5)
            curr_w = min(1.0, curr_w)

        else:
            curr_w = prev_w_base

        prev_w_base = curr_w

        # === 棘轮逻辑 ===
        final_w = curr_w
        if cfg.ENABLE_RATCHET:
            if curr_w > 0.5:
                # 多头只增不减
                if prev_w_final > 0.5:
                    final_w = max(curr_w, prev_w_final)
                else:
                    final_w = curr_w
            elif curr_w < 0.5:
                # 空头只减不增
                if prev_w_final < 0.5:
                    final_w = min(curr_w, prev_w_final)
                else:
                    final_w = curr_w
            else:
                final_w = 0.5

        prev_w_final = final_w
        targets.append(final_w)

    df['Target_W'] = targets
    df['Target_Exec'] = df['Target_W'].shift(1)
    return df


# ----------------------------------------------------------------------
# 📌 5. 回测执行引擎 (无再平衡)
# ----------------------------------------------------------------------
def run_backtest(df, cfg):
    try:
        start_idx = max(cfg.STD_WINDOW + cfg.RANK_WINDOW, 100)
        df_valid = df.iloc[start_idx:].copy()
        df_bt = df_valid.loc[cfg.START_DATE:cfg.END_DATE].copy()
        if df_bt.empty: raise ValueError("无数据")
        print(f"📅 回测区间: {df_bt.index[0].date()} 至 {df_bt.index[-1].date()}")
    except Exception as e:
        print(f"⚠️ 使用全部数据: {e}")
        df_bt = df_valid.copy()

    ret_500 = (df_bt['close_price4'] / df_bt['prev_close4'].fillna(df_bt['open_price4']) - 1).values
    ret_hl = (df_bt['close_price5'] / df_bt['prev_close5'].fillna(df_bt['open_price5']) - 1).values
    target_exec = df_bt['Target_Exec'].fillna(0.5).values

    # 核心循环
    w_actual = np.zeros(len(df_bt))
    signal_changes = 0

    last_signal_target = target_exec[0]
    w_curr = target_exec[0]

    for i in range(len(df_bt)):
        current_signal_target = target_exec[i]

        # 严谨判断：只有目标变了才动
        if abs(current_signal_target - last_signal_target) > 1e-6:
            w_curr = current_signal_target
            last_signal_target = current_signal_target
            signal_changes += 1

        w_actual[i] = w_curr

        # 漂移
        r_day = w_curr * ret_500[i] + (1 - w_curr) * ret_hl[i]
        w_curr = w_curr * (1 + ret_500[i]) / (1 + r_day)
        w_curr = np.clip(w_curr, 0.0, 1.0)

    df_bt['W_Final'] = w_actual
    df_bt['W_HL'] = 1.0 - w_actual

    print(f"📊 实际调仓天数: {signal_changes} / {len(df_bt)}")

    init_w = df_bt['W_Final'].iloc[0]
    df_bt['Turnover'] = (df_bt['W_Final'] - df_bt['W_Final'].shift(1).fillna(init_w)).abs()

    cost_impact = df_bt['Turnover'] * (cfg.COST + cfg.SLIPPAGE) * 2
    strat_gross = df_bt['W_Final'] * ret_500 + df_bt['W_HL'] * ret_hl
    df_bt['Strat_Ret'] = strat_gross - cost_impact
    df_bt['Strat_Cum'] = (1 + df_bt['Strat_Ret']).cumprod()

    nav_500 = (1 + pd.Series(ret_500, index=df_bt.index)).cumprod()
    nav_hl = (1 + pd.Series(ret_hl, index=df_bt.index)).cumprod()
    df_bt['Bench_Cum'] = 0.5 * nav_500 + 0.5 * nav_hl
    df_bt['Bench_Cum'] = df_bt['Bench_Cum'] / df_bt['Bench_Cum'].iloc[0] * df_bt['Strat_Cum'].iloc[0]

    return df_bt


# ----------------------------------------------------------------------
# 📌 6. 绩效统计
# ----------------------------------------------------------------------
def analyze_performance(df_bt):
    days = len(df_bt)
    ann_ret = (df_bt['Strat_Cum'].iloc[-1] / df_bt['Strat_Cum'].iloc[0]) ** (252 / days) - 1
    bench_ann = (df_bt['Bench_Cum'].iloc[-1] / df_bt['Bench_Cum'].iloc[0]) ** (252 / days) - 1
    mdd = (df_bt['Strat_Cum'] / df_bt['Strat_Cum'].cummax() - 1).min()
    excess = ann_ret - bench_ann
    avg_turnover = df_bt['Turnover'].mean()

    print("\n" + "=" * 50)
    print(f"🏆 最终策略绩效报告 (G组参数)")
    print("=" * 50)
    print(f"策略年化: {ann_ret:.2%}")
    print(f"基准年化: {bench_ann:.2%}")
    print(f"超额收益: {excess:.2%}")
    print(f"最大回撤: {mdd:.2%}")
    print(f"日均换手: {avg_turnover:.2%}")
    print(f"Calmar  : {ann_ret / abs(mdd):.2f}")

    fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

    axes[0].plot(df_bt['Strat_Cum'], label='策略', color='#d62728', lw=2)
    axes[0].plot(df_bt['Bench_Cum'], label='基准', color='gray', ls='--')
    axes[0].set_title('净值曲线')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(df_bt['W_Final'], color='green', alpha=0.9, lw=1.5, label='500持仓')
    axes[1].plot(df_bt['Target_Exec'], color='black', alpha=0.3, ls=':', label='目标')
    axes[1].set_ylabel('Weight 500')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(df_bt['Factor_Rank'], color='#1f77b4', alpha=0.5, label='Rank')
    axes[2].axhline(Config.LINEAR_HIGH, color='red', ls='--')
    axes[2].axhline(Config.LINEAR_LOW, color='green', ls='--')
    axes[2].set_title('因子分位数')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    cfg = Config()
    try:
        df_raw = load_and_clean_data(cfg.FILE_PATH)
        df_factor = calculate_factor(df_raw, cfg)
        df_signal = generate_signals(df_factor, cfg)
        df_result = run_backtest(df_signal, cfg)
        analyze_performance(df_result)

        last = df_result.iloc[-1]
        print(f"\n📊 最新状态 [{df_result.index[-1].date()}]")
        print(f"   Rank:   {last['Factor_Rank']:.2%}")
        print(f"   Target: {last['Target_W']:.2%}")

    except Exception as e:
        print(f"\n❌ 错误: {e}")