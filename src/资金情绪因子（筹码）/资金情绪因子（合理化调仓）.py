import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
import warnings

warnings.filterwarnings('ignore')


def run_final_constrained_backtest(file_path):
    print(f"读取数据: {file_path}")
    try:
        df = pd.read_excel(file_path)
    except Exception as e:
        print(f"读取失败: {e}")
        return

    df['TradingDay'] = pd.to_datetime(df['TradingDay'])
    df = df.sort_values('TradingDay').reset_index(drop=True)

    # ==========================================
    # 1. 因子与指标计算
    # ==========================================

    # A. 核心博弈因子: (主力净买 - 散户净买) / 流通市值
    def get_factor(sub_df, suffix, window=10):
        # 净流入 = (XL+L) - (XL_out+L_out) - (S_in - S_out)
        net_flow = (sub_df[f'buy_value_xl{suffix}'] + sub_df[f'buy_value_l{suffix}']) - \
                   (sub_df[f'sell_value_xl{suffix}'] + sub_df[f'sell_value_l{suffix}']) - \
                   (sub_df[f'buy_value_s{suffix}'] - sub_df[f'sell_value_s{suffix}'])
        mkt_val = sub_df[f'free_float_mktval{suffix}']
        ratio = (net_flow / mkt_val).rolling(window).sum()
        # 60日滚动标准化
        z = (ratio - ratio.rolling(32).mean()) / ratio.rolling(32).std()
        return z

    z1 = get_factor(df, '1')
    z2 = get_factor(df, '2')
    spread = z1 - z2

    # B. 绝对大单净流入 (用于新增约束: XL+L 是否大于0)
    # 不减去散户，纯看主力是不是在买
    def get_raw_main_flow(sub_df, suffix):
        main_in = sub_df[f'buy_value_xl{suffix}'] + sub_df[f'buy_value_l{suffix}']
        main_out = sub_df[f'sell_value_xl{suffix}'] + sub_df[f'sell_value_l{suffix}']
        return main_in - main_out

    raw_flow1 = get_raw_main_flow(df, '1')
    raw_flow2 = get_raw_main_flow(df, '2')

    # C. 均线 (用于趋势约束)
    ma20_1 = df['close_price1'].rolling(20).mean()
    ma20_2 = df['close_price2'].rolling(20).mean()

    # ==========================================
    # 2. IC 指标分析 (Information Coefficient)
    # ==========================================
    # 逻辑：Factor(T) 应该预测 Relative_Return(T+1)
    # Relative_Return = Index1_Ret - Index2_Ret

    next_rel_ret = (df['index_return1'] - df['index_return2']).shift(-1)
    valid_idx = spread.notna() & next_rel_ret.notna()

    # 1. Normal IC (Pearson)
    ic_val, _ = pearsonr(spread[valid_idx], next_rel_ret[valid_idx])
    # 2. Rank IC (Spearman)
    rank_ic, _ = spearmanr(spread[valid_idx], next_rel_ret[valid_idx])

    # 3. ICIR (IC / Std(IC)) - 这里简单用滚动IC来计算
    rolling_ic = spread.rolling(60).corr(next_rel_ret)
    icir = rolling_ic.mean() / rolling_ic.std()

    print("-" * 30)
    print("【IC 表现分析】(因子预测能力)")
    print(f"Rank IC (整个周期): {rank_ic:.4f} (值越高预测越准)")
    print(f"ICIR (滚动60日):   {icir:.4f} (值越高越稳定)")
    print("-" * 30)

    # ==========================================
    # 3. 策略逻辑 (含所有约束)
    # ==========================================
    target_weights = []

    NEUTRAL_TH = 0.3
    MAX_TH = 2.0
    RANGE_WIDTH = MAX_TH - NEUTRAL_TH
    REQUIRED_DAYS = 3

    last_locked_w = 0.5
    consecutive_bull = 0
    consecutive_bear = 0

    for i in range(len(spread)):
        s = spread.iloc[i]

        # 提取当天的约束条件数据
        p1, m1 = df['close_price1'].iloc[i], ma20_1.iloc[i]
        p2, m2 = df['close_price2'].iloc[i], ma20_2.iloc[i]
        rf1 = raw_flow1.iloc[i]  # 500的主力净买入金额
        rf2 = raw_flow2.iloc[i]  # 红利的主力净买入金额

        if pd.isna(s) or pd.isna(m1):
            target_weights.append(0.5)
            continue

        # 计数器
        if s > NEUTRAL_TH:
            consecutive_bull += 1
            consecutive_bear = 0
        elif s < -NEUTRAL_TH:
            consecutive_bear += 1
            consecutive_bull = 0
        else:
            consecutive_bull = 0
            consecutive_bear = 0

        # --- 决策 ---

        # 1. 中性区
        if abs(s) <= NEUTRAL_TH:
            current_target = 0.5
            last_locked_w = 0.5

        # 2. 倾向做多 500
        elif s > NEUTRAL_TH:
            # 严格约束链:
            # A. Spread > 0.4 且 持续3天
            # B. 500 价格 > 20日均线 (趋势)
            # C. 500 主力大单净流入 > 0 (真金白银)
            is_valid = (consecutive_bull >= REQUIRED_DAYS) and \
                       (p1 > m1) and \
                       (rf1 > 0)

            if is_valid:
                pct = (s - NEUTRAL_TH) / RANGE_WIDTH
                raw_w = 0.5 + 0.5 * pct
                raw_w = min(raw_w, 1.0)
                # 棘轮锁定
                if last_locked_w < 0.5:
                    current_target = raw_w
                else:
                    current_target = max(last_locked_w, raw_w)
                last_locked_w = current_target
            else:
                # 只要有一条不满足，就不加仓，维持原状
                current_target = last_locked_w

        # 3. 倾向做多 红利
        else:  # s < -NEUTRAL_TH
            # 严格约束链:
            # A. Spread < -0.4 且 持续3天
            # B. 红利 价格 > 20日均线
            # C. 红利 主力大单净流入 > 0
            is_valid = (consecutive_bear >= REQUIRED_DAYS) and \
                       (p2 > m2) and \
                       (rf2 > 0)

            if is_valid:
                pct = (abs(s) - NEUTRAL_TH) / RANGE_WIDTH
                raw_w = 0.5 - 0.5 * pct
                raw_w = max(raw_w, 0.0)
                # 棘轮锁定
                if last_locked_w > 0.5:
                    current_target = raw_w
                else:
                    current_target = min(last_locked_w, raw_w)
                last_locked_w = current_target
            else:
                current_target = last_locked_w

        target_weights.append(current_target)

    target_series = pd.Series(target_weights).shift(1).fillna(0.5)

    # ==========================================
    # 4. 回测执行 (计算胜率)
    # ==========================================
    print("执行严格回测...")

    strat_vals = []
    bench_vals = []

    holdings_1 = 0.5
    holdings_2 = 0.5
    bench_val1 = 0.5
    bench_val2 = 0.5

    cost_rate = 0.0001
    prev_target_w = 0.5

    # 记录
    real_w1_history = []

    # 交易区间胜率统计
    interval_wins = 0
    interval_total = 0
    # 记录上次调仓时的净值
    last_rebal_strat_nav = 1.0
    last_rebal_bench_nav = 1.0

    for i in range(len(df)):
        r1 = df.loc[i, 'index_return1']
        r2 = df.loc[i, 'index_return2']

        curr_target_w = target_series.iloc[i]
        curr_total = holdings_1 + holdings_2
        curr_bench_total = bench_val1 + bench_val2

        # 只有目标仓位发生实质变化才调仓
        if abs(curr_target_w - prev_target_w) > 0.001:

            # --- 结算上一个区间的胜负 ---
            # 区间收益率
            strat_ret = (curr_total / last_rebal_strat_nav) - 1
            bench_ret = (curr_bench_total / last_rebal_bench_nav) - 1

            interval_total += 1
            # 胜率判定标准：跑赢基准 -0.02% 就算赢
            if (strat_ret - bench_ret) > -0.0002:
                interval_wins += 1

            # 重置区间起点
            last_rebal_strat_nav = curr_total
            last_rebal_bench_nav = curr_bench_total

            # --- 执行调仓 ---
            tgt_val1 = curr_total * curr_target_w
            tgt_val2 = curr_total * (1 - curr_target_w)

            turnover = abs(tgt_val1 - holdings_1) + abs(tgt_val2 - holdings_2)
            cost = turnover * cost_rate

            new_total = curr_total - cost
            holdings_1 = new_total * curr_target_w
            holdings_2 = new_total * (1 - curr_target_w)

            prev_target_w = curr_target_w

        # 记录
        real_total = holdings_1 + holdings_2
        real_w1_history.append(holdings_1 / real_total if real_total > 0 else 0)

        # 净值更新
        holdings_1 *= (1 + r1)
        holdings_2 *= (1 + r2)
        strat_vals.append(holdings_1 + holdings_2)

        bench_val1 *= (1 + r1)
        bench_val2 *= (1 + r2)
        bench_vals.append(bench_val1 + bench_val2)

    # ==========================================
    # 5. 绩效指标统计
    # ==========================================
    df['Strategy_Nav'] = strat_vals
    df['Benchmark_Nav'] = bench_vals
    df['Excess_Nav'] = df['Strategy_Nav'] / df['Benchmark_Nav']

    # A. 收益率
    total_ret_strat = df['Strategy_Nav'].iloc[-1] - 1
    total_ret_bench = df['Benchmark_Nav'].iloc[-1] - 1

    # B. 夏普比率 (年化)
    strat_pct = df['Strategy_Nav'].pct_change().fillna(0)
    rf = 0.02 / 250  # 假设年化无风险利率2%
    sharpe = (strat_pct.mean() - rf) / strat_pct.std() * np.sqrt(250)

    # C. 最大回撤
    roll_max = df['Strategy_Nav'].cummax()
    dd = df['Strategy_Nav'] / roll_max - 1
    max_dd = dd.min()

    # D. 胜率
    win_rate = (interval_wins / interval_total) if interval_total > 0 else 0

    print("=" * 50)
    print("【终极策略报告】(含净流入约束 & 胜率统计)")
    print("=" * 50)
    print(f"累计收益:     {total_ret_strat * 100:.2f}% (基准: {total_ret_bench * 100:.2f}%)")
    print(f"超额收益:     {(total_ret_strat - total_ret_bench) * 100:.2f}%")
    print(f"夏普比率:     {sharpe:.2f}")
    print(f"最大回撤:     {max_dd * 100:.2f}%")
    print("-" * 30)
    print(f"总调仓次数:   {interval_total} 次")
    print(f"区间胜率:     {win_rate * 100:.2f}%")
    print("-" * 30)
    print("约束条件:")
    print("1. Spread > 0.4 (3天确认)")
    print("2. 价格 > MA20")
    print("3. 大单净流入(XL+L) > 0")
    print("=" * 50)

    # ==========================================
    # 6. 绘图 (4张图)
    # ==========================================
    fig = plt.figure(figsize=(14, 16))
    gs = fig.add_gridspec(4, 1, height_ratios=[2, 1, 1, 1])

    # 图1：净值对比
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(df['TradingDay'], df['Benchmark_Nav'], color='gray', linestyle='--', label='基准(50/50漂移)')
    ax1.plot(df['TradingDay'], df['Strategy_Nav'], color='#d62728', linewidth=2, label='严格策略净值')
    ax1.set_ylabel('净值')
    ax1.set_title('策略 vs 基准 累计收益')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)

    # 图2：超额收益图
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    # 画 (策略/基准 - 1)
    excess_pct = df['Excess_Nav'] - 1
    ax2.plot(df['TradingDay'], excess_pct, color='#800080', linewidth=1.5, label='相对强弱(Strategy/Benchmark - 1)')
    ax2.fill_between(df['TradingDay'], excess_pct, 0, where=(excess_pct > 0), color='red', alpha=0.1)
    ax2.fill_between(df['TradingDay'], excess_pct, 0, where=(excess_pct < 0), color='green', alpha=0.1)
    ax2.set_ylabel('累计超额')
    ax2.legend(loc='upper left')
    ax2.set_title('超额收益曲线 (向上代表持续跑赢)')

    # 图3：持仓权重
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    ax3.fill_between(df['TradingDay'], 0, real_w1_history, color='#d62728', alpha=0.6, label='中证500仓位')
    ax3.fill_between(df['TradingDay'], real_w1_history, 1, color='#2ca02c', alpha=0.6, label='红利仓位')
    ax3.set_ylabel('权重')
    ax3.legend(loc='upper left')
    ax3.set_title('持仓分布 (严格约束后的调仓)')

    # 图4：因子 Spread
    ax4 = fig.add_subplot(gs[3], sharex=ax1)
    ax4.plot(df['TradingDay'], spread, color='#1f77b4', linewidth=1, label='Spread')
    ax4.axhline(0.4, color='orange', linestyle='--')
    ax4.axhline(-0.4, color='orange', linestyle='--')
    ax4.set_ylabel('Spread')
    ax4.set_title('资金流 Spread')

    plt.tight_layout()
    plt.show()


# 运行
path = r"C:\Users\86178\Desktop\资金大中小因子.xlsx"
run_final_constrained_backtest(path)