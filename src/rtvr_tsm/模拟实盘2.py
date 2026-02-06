import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings

# ----------------------------------------------------------------------
# 📌 0. 全局配置 (混合配置)
# ----------------------------------------------------------------------
warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


class Config:
    # 🌟 1. 数据路径
    FILE_PATH = r"C:\Users\86178\Desktop\整合数据.csv"

    # 🌟 2. 回测区间
    START_DATE = '2021-01-01'
    END_DATE = '2099-12-31'
    REAL_START_DATE = '2026-01-26'

    # 🌟 3. 费率设置
    COST = 0.0002  # 佣金 万2
    SLIPPAGE = 0.0003  # 冲击成本 万3 (模拟 VWAP 偏差)

    # 🌟 4. 策略参数
    F2_FLOW_WINDOW = 10
    F2_Z_WINDOW = 32

    # 逻辑阈值
    NEUTRAL_TH = 0.3  # 中性阈值
    MAX_TH = 2.0  # 满仓阈值 (用于计算仓位比例)
    REQ_DAYS = 3  # 连续确认天数


# ----------------------------------------------------------------------
# 📌 1. 数据加载与清洗
# ----------------------------------------------------------------------
def load_and_clean_data(file_path, cfg):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"❌ 错误：找不到文件 {file_path}")

    print("⏳ 正在加载数据 ...")
    try:
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        else:
            df = pd.read_excel(file_path)
    except Exception as e:
        raise ValueError(f"❌ 读取失败: {e}")

    df['TradingDay'] = pd.to_datetime(df['TradingDay'])
    df = df.set_index('TradingDay').sort_index()

    # 强制截断
    df = df[df.index >= cfg.START_DATE].copy()
    if df.empty:
        raise ValueError(f"❌ 错误：截断后无数据，请检查日期或文件。")

    print(f"✂️ 已执行冷启动截断，数据范围: {df.index[0].date()} 至 {df.index[-1].date()}")

    # --- 映射列名  ---
    # 中证500
    df['MktVal_500'] = df['idx_000905_SH__free_float_mktval']
    main_net_500 = (df['idx_000905_SH__buy_value_xl'] + df['idx_000905_SH__buy_value_l']) - \
                   (df['idx_000905_SH__sell_value_xl'] + df['idx_000905_SH__sell_value_l'])
    retail_net_500 = df['idx_000905_SH__buy_value_s'] - df['idx_000905_SH__sell_value_s']
    df['Flow_Net_500'] = main_net_500 - retail_net_500
    df['Flow_Main_Raw_500'] = main_net_500  # 纯主力净买

    # 红利低波
    df['MktVal_HL'] = df['idx_000922_SH__free_float_mktval']
    main_net_hl = (df['idx_000922_SH__buy_value_xl'] + df['idx_000922_SH__buy_value_l']) - \
                  (df['idx_000922_SH__sell_value_xl'] + df['idx_000922_SH__sell_value_l'])
    retail_net_hl = df['idx_000922_SH__buy_value_s'] - df['idx_000922_SH__sell_value_s']
    df['Flow_Net_HL'] = main_net_hl - retail_net_hl
    df['Flow_Main_Raw_HL'] = main_net_hl  # 纯主力净买

    # 价格数据
    df['Open_500'] = df['fund_512510__open_price']
    df['Close_500'] = df['fund_512510__close_price']
    df['Prev_500'] = df['fund_512510__prev_close']
    df['VWAP_500'] = df['fund_512510__avg_price']

    df['Open_HL'] = df['fund_515180__open_price']
    df['Close_HL'] = df['fund_515180__close_price']
    df['Prev_HL'] = df['fund_515180__prev_close']
    df['VWAP_HL'] = df['fund_515180__avg_price']

    # 缺失值填充
    price_cols = ['Close_500', 'Prev_500', 'VWAP_500', 'Close_HL', 'Prev_HL', 'VWAP_HL']
    for col in price_cols:
        if col in df.columns:
            df[col] = df[col].replace(0, np.nan)
    df[price_cols] = df[price_cols].ffill().bfill()
    df['VWAP_500'] = df['VWAP_500'].fillna(df['Close_500'])
    df['VWAP_HL'] = df['VWAP_HL'].fillna(df['Close_HL'])

    print(f"✅ 数据加载并清洗完成: {len(df)} 条记录")
    return df


# ----------------------------------------------------------------------
# 📌 2. 因子计算
# ----------------------------------------------------------------------
def calc_factors(df, cfg):
    # 1. 计算 MA20
    df['MA20_500'] = df['Close_500'].rolling(20).mean()
    df['MA20_HL'] = df['Close_HL'].rolling(20).mean()

    # 2. 计算 Spread (Z-Score)
    def get_z_score(flow, mkt_val):
        ratio = (flow / mkt_val).rolling(cfg.F2_FLOW_WINDOW).sum()
        z = (ratio - ratio.rolling(cfg.F2_Z_WINDOW).mean()) / ratio.rolling(cfg.F2_Z_WINDOW).std()
        return z

    z1 = get_z_score(df['Flow_Net_500'], df['MktVal_500'])
    z2 = get_z_score(df['Flow_Net_HL'], df['MktVal_HL'])

    df['F2_Spread'] = z1 - z2
    return df


# ----------------------------------------------------------------------
# 📌 3. 信号生成
# ----------------------------------------------------------------------
def generate_signals(df, cfg):
    print("🚦 正在生成信号 ...")

    target_weights = []

    neutral_th = cfg.NEUTRAL_TH
    range_width = cfg.MAX_TH - neutral_th
    req_days = cfg.REQ_DAYS

    last_locked_w = 0.5
    consecutive_bull = 0
    consecutive_bear = 0

    for i in range(len(df)):
        s = df['F2_Spread'].iloc[i]

        # 提取当前数据
        p1, m1 = df['Close_500'].iloc[i], df['MA20_500'].iloc[i]
        p2, m2 = df['Close_HL'].iloc[i], df['MA20_HL'].iloc[i]
        rf1 = df['Flow_Main_Raw_500'].iloc[i]
        rf2 = df['Flow_Main_Raw_HL'].iloc[i]

        curr_w = last_locked_w

        if pd.isna(s) or pd.isna(m1):
            curr_w = 0.5
        else:
            # --- 计数器逻辑  ---
            if s > neutral_th:
                consecutive_bull += 1
                consecutive_bear = 0
            elif s < -neutral_th:
                consecutive_bear += 1
                consecutive_bull = 0
            else:
                consecutive_bull = 0
                consecutive_bear = 0

            # --- 决策逻辑  ---

            # 1. 中性区
            if abs(s) <= neutral_th:
                curr_w = 0.5
                last_locked_w = 0.5  # 中性区重置锁定

            # 2. 倾向 500
            elif s > neutral_th:
                # 约束: 持续天数 + 价格趋势 + 主力净买入
                is_valid = (consecutive_bull >= req_days) and (p1 > m1) and (rf1 > 0)

                if is_valid:
                    pct = (s - neutral_th) / range_width
                    raw_w = min(1.0, 0.5 + 0.5 * pct)
                    # 棘轮: 只能加不能减 (相对于上次锁定值)
                    curr_w = max(last_locked_w, raw_w) if last_locked_w > 0.5 else raw_w
                else:
                    curr_w = last_locked_w  # 不满足约束，不动

            # 3. 倾向 HL
            else:  # s < -neutral_th
                # 约束: 持续天数 + 价格趋势 + 主力净买入
                is_valid = (consecutive_bear >= req_days) and (p2 > m2) and (rf2 > 0)

                if is_valid:
                    pct = (abs(s) - neutral_th) / range_width
                    raw_w = max(0.0, 0.5 - 0.5 * pct)
                    # 棘轮: 只能减不能加 (增加HL仓位)
                    curr_w = min(last_locked_w, raw_w) if last_locked_w < 0.5 else raw_w
                else:
                    curr_w = last_locked_w

        last_locked_w = curr_w
        target_weights.append(curr_w)

    df['Target_W_500'] = target_weights
    return df


# ----------------------------------------------------------------------
# 📌 4. 回测执行引擎
# ----------------------------------------------------------------------
def run_backtest(df, cfg):
    print("🏃 开始回测 (引擎: Code 1 | 撮合: VWAP | 费率: 高)...")
    df_bt = df.copy()

    # 信号滞后1天 (T日信号，T+1日执行)
    df_bt['Target_W_Exec'] = df_bt['Target_W_500'].shift(1).fillna(0.5)

    close_500 = df_bt['Close_500'].values
    prev_500 = df_bt['Prev_500'].values
    close_hl = df_bt['Close_HL'].values
    prev_hl = df_bt['Prev_HL'].values

    # 1. 计算自然漂移后的实际权重
    def calc_actual_weights(targets, ret_a, ret_b):
        w_actual = np.zeros(len(targets))
        w_curr = targets[0]
        for i in range(len(targets)):
            tgt = targets[i]
            # 只有目标变动超过阈值才触发调仓，否则自然漂移
            if abs(tgt - w_curr) > 1e-4:
                w_curr = tgt
            w_actual[i] = w_curr
            # 次日漂移
            r_day = w_curr * ret_a[i] + (1 - w_curr) * ret_b[i]
            w_curr = w_curr * (1 + ret_a[i]) / (1 + r_day)
            w_curr = np.clip(w_curr, 0.0, 1.0)
        return w_actual

    ret_500_full = close_500 / prev_500 - 1
    ret_hl_full = close_hl / prev_hl - 1

    # 计算每日实际持仓 (含漂移)
    w_real_500 = calc_actual_weights(df_bt['Target_W_Exec'].values, ret_500_full, ret_hl_full)

    df_bt['W_500_Final'] = w_real_500
    df_bt['W_HL_Final'] = 1.0 - w_real_500

    # 2. VWAP 收益贡献分解
    # 逻辑: 只有持有部分享受 (Close/Prev)，买入部分享受 (Close/VWAP)，卖出部分承受 (VWAP/Prev)
    def calc_vwap_contrib(w_curr, w_prev, close, prev, vwap):
        delta = w_curr - w_prev
        # 持有部分: 全天涨跌
        ret_hold = np.minimum(w_curr, w_prev) * (close / prev - 1)
        # 买入部分: 从 VWAP 到 Close
        ret_buy = delta.clip(lower=0) * (close / vwap - 1)
        # 卖出部分: 从 Prev 到 VWAP (踏空后续涨跌) -> 这里实际上是计算亏损/收益
        # 卖出的钱只享受了 (VWAP/Prev - 1) 的收益，然后变成现金(假设不计息)
        # 但为了计算净值，通常视作卖出变现。这里采用简化的贡献度加总。
        ret_sell = delta.clip(upper=0).abs() * (vwap / prev - 1)
        return ret_hold + ret_buy + ret_sell

    init_w = df_bt['W_500_Final'].iloc[0]
    w_prev = df_bt['W_500_Final'].shift(1).fillna(init_w)

    # 换手率计算
    df_bt['Turnover'] = (df_bt['W_500_Final'] - w_prev).abs()

    # 贡献度计算
    contrib_500 = calc_vwap_contrib(df_bt['W_500_Final'], w_prev,
                                    df_bt['Close_500'], df_bt['Prev_500'], df_bt['VWAP_500'])

    # 红利仓位变化 (注意：500买入等于红利卖出)
    w_hl_curr = df_bt['W_HL_Final']
    w_hl_prev = 1.0 - w_prev
    contrib_hl = calc_vwap_contrib(w_hl_curr, w_hl_prev,
                                   df_bt['Close_HL'], df_bt['Prev_HL'], df_bt['VWAP_HL'])

    # 3. 费率扣除
    total_fee = df_bt['Turnover'] * (cfg.COST + cfg.SLIPPAGE) * 2

    # 4. 汇总净值
    df_bt['Strat_Ret'] = contrib_500 + contrib_hl - total_fee
    df_bt['Strat_Cum'] = (1 + df_bt['Strat_Ret']).cumprod()

    # 基准 (50/50 简单复利)
    b_500 = (1 + ret_500_full).cumprod()
    b_hl = (1 + ret_hl_full).cumprod()
    df_bt['Bench_Cum'] = 0.5 * b_500 + 0.5 * b_hl
    # 归一化基准
    df_bt['Bench_Cum'] = df_bt['Bench_Cum'] / df_bt['Bench_Cum'].iloc[0] * df_bt['Strat_Cum'].iloc[0]

    return df_bt


# ----------------------------------------------------------------------
# 📌 5. 绩效展示
# ----------------------------------------------------------------------
def analyze_performance(df_bt, cfg):
    # 预热期 (Z_WINDOW + FLOW_WINDOW)
    warmup_days = cfg.F2_Z_WINDOW + cfg.F2_FLOW_WINDOW

    if len(df_bt) > warmup_days:
        df_plot = df_bt.iloc[warmup_days:].copy()
        # 重新归一化
        df_plot['Strat_Cum'] /= df_plot['Strat_Cum'].iloc[0]
        df_plot['Bench_Cum'] /= df_plot['Bench_Cum'].iloc[0]
        print(f"✂️ 报告展示区间 (剔除预热): {df_plot.index[0].date()} 至 {df_plot.index[-1].date()}")
    else:
        df_plot = df_bt.copy()

    days = len(df_plot)
    if days > 0:
        ann_ret = (df_plot['Strat_Cum'].iloc[-1] / df_plot['Strat_Cum'].iloc[0]) ** (252 / days) - 1
        bench_ret = (df_plot['Bench_Cum'].iloc[-1] / df_plot['Bench_Cum'].iloc[0]) ** (252 / days) - 1
        mdd = (df_plot['Strat_Cum'] / df_plot['Strat_Cum'].cummax() - 1).min()
        sharpe = (df_plot['Strat_Ret'].mean() / df_plot['Strat_Ret'].std()) * np.sqrt(252)
        turnover = df_plot['Turnover'].mean()
    else:
        ann_ret = bench_ret = mdd = sharpe = turnover = 0

    print("\n" + "=" * 50)
    print("🏆 全局回测报告 (逻辑:Code2 | 引擎:Code1)")
    print("=" * 50)
    print(f"区间策略年化: {ann_ret:.2%}")
    print(f"区间基准年化: {bench_ret:.2%}")
    print(f"区间超额年化: {ann_ret - bench_ret:.2%}")
    print(f"区间最大回撤: {mdd:.2%}")
    print(f"区间夏普比率: {sharpe:.2f}")
    print(f"区间日均换手: {turnover:.2%}")
    print("=" * 50)

    fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

    axes[0].plot(df_plot['Strat_Cum'], color='#d62728', lw=2, label='策略净值 (VWAP撮合)')
    axes[0].plot(df_plot['Bench_Cum'], color='gray', ls='--', label='基准(50/50)')
    axes[0].set_title('累计净值曲线')
    axes[0].legend(loc='upper left')
    axes[0].grid(True, alpha=0.3)

    axes[1].fill_between(df_plot.index, 0, df_plot['W_500_Final'], color='#d62728', alpha=0.5, label='500仓位')
    axes[1].fill_between(df_plot.index, df_plot['W_500_Final'], 1, color='#2ca02c', alpha=0.5, label='红利仓位')
    axes[1].set_ylabel('权重')
    axes[1].legend(loc='upper left')
    axes[1].grid(True, alpha=0.3)

    excess = df_plot['Strat_Cum'] / df_plot['Bench_Cum'] - 1
    axes[2].plot(excess, color='purple', lw=1.5)
    axes[2].fill_between(excess.index, excess, 0, where=(excess > 0), color='red', alpha=0.1)
    axes[2].fill_between(excess.index, excess, 0, where=(excess < 0), color='green', alpha=0.1)
    axes[2].set_title('相对基准超额')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# ----------------------------------------------------------------------
# 📌 6. 实盘跟踪与诊断
# ----------------------------------------------------------------------
def print_real_record_and_advice(df_bt, df_raw, cfg):
    # --- Part 1: 实盘记录 ---
    record_date = pd.to_datetime(cfg.REAL_START_DATE)
    df_rec = df_bt[df_bt.index >= record_date].copy()

    if not df_rec.empty:
        base_s = df_rec['Strat_Cum'].iloc[0]
        base_b = df_rec['Bench_Cum'].iloc[0]
        ret_s = df_rec['Strat_Cum'].iloc[-1] / base_s - 1
        ret_b = df_rec['Bench_Cum'].iloc[-1] / base_b - 1

        print("\n" + "#" * 60)
        print(f"📈 实盘跟踪 (基准日: {cfg.REAL_START_DATE})")
        print("#" * 60)
        print(f"   💰 策略累计: {ret_s:+.2%}")
        print(f"   📊 基准累计: {ret_b:+.2%}")
        print(f"   🔥 超额收益: {ret_s - ret_b:+.2%}")

    # --- Part 2: 决策诊断  ---
    last = df_raw.iloc[-1]
    tgt_w = df_bt['Target_W_500'].iloc[-1]  # 这是根据今天收盘数据算出来的，明天执行的目标

    dt = df_raw.index[-1].date()
    spread = last['F2_Spread']

    p1, m1 = last['Close_500'], last['MA20_500']
    rf1 = last['Flow_Main_Raw_500']

    p2, m2 = last['Close_HL'], last['MA20_HL']
    rf2 = last['Flow_Main_Raw_HL']

    status_str = "⚪ 震荡区"
    if spread > cfg.NEUTRAL_TH: status_str = "🔴 信号区: 倾向500"
    if spread < -cfg.NEUTRAL_TH: status_str = "🟢 信号区: 倾向HL"

    print("\n" + "#" * 60)
    print(f"📝 策略诊断书 (数据截止: {dt})")
    print("#" * 60)
    print(f"   🔹 Spread: {spread:.4f} (阈值 {cfg.NEUTRAL_TH})")
    print(f"   🔹 状态: {status_str}")
    print("-" * 30)
    print(f"   🔎 约束检查 (500):")
    print(f"      - 价格 > MA20?  {'✅' if p1 > m1 else '❌'} ({p1:.3f} vs {m1:.3f})")
    print(f"      - 主力净买 > 0?  {'✅' if rf1 > 0 else '❌'} ({rf1 / 1e8:+.2f}亿)")
    print(f"   🔎 约束检查 (HL):")
    print(f"      - 价格 > MA20?  {'✅' if p2 > m2 else '❌'} ({p2:.3f} vs {m2:.3f})")
    print(f"      - 主力净买 > 0?  {'✅' if rf2 > 0 else '❌'} ({rf2 / 1e8:+.2f}亿)")
    print("-" * 30)
    print(f"   🚀 明日建议仓位:")
    print(f"      🔴 中证500: {tgt_w:.2%}")
    print(f"      🔵 红利低波: {1 - tgt_w:.2%}")
    print("#" * 60 + "\n")


# ----------------------------------------------------------------------
# 📌 主程序
# ----------------------------------------------------------------------
if __name__ == "__main__":
    try:
        cfg = Config()
        df_raw = load_and_clean_data(cfg.FILE_PATH, cfg)
        df_fac = calc_factors(df_raw, cfg)
        df_sig = generate_signals(df_fac, cfg)
        df_res = run_backtest(df_sig, cfg)

        analyze_performance(df_res, cfg)
        print_real_record_and_advice(df_res, df_sig, cfg)

    except Exception as e:
        print(f"\n❌ 出错: {e}")
        import traceback

        traceback.print_exc()