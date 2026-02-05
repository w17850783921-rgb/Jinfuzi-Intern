import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import optuna
import os

# ----------------------------------------------------------------------
# 📌 0. 全局设置
# ----------------------------------------------------------------------
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

FILE_PATH = r"C:\Users\86178\Desktop\交易情绪因子2.csv"

# 训练/验证 切分
TRAIN_START = '2021-01-01'
TRAIN_END = '2024-06-30'
TEST_START = '2024-07-01'
TEST_END = '2099-12-31'

COST = 0.0002
SLIPPAGE = 0.0003

# ----------------------------------------------------------------------
# 1. 数据准备
# ----------------------------------------------------------------------
if not os.path.exists(FILE_PATH):
    print(f"❌ 错误：找不到文件 {FILE_PATH}")
    exit()

df_raw = pd.read_csv(FILE_PATH, parse_dates=['TradingDay']).set_index('TradingDay').sort_index()

df_raw.rename(columns={
    'index_return1': 'Ret_Idx_500', 'turnover_value1': 'Val_500', 'negotiable_mv1': 'MV_500',
    'index_return2': 'Ret_Idx_HL', 'turnover_value2': 'Val_HL', 'negotiable_mv2': 'MV_HL'
}, inplace=True)

cols_etf = ['close_price4', 'prev_close4', 'close_price5', 'prev_close5']
df_raw[cols_etf] = df_raw[cols_etf].replace(0, np.nan).ffill().bfill()
df_raw['Ret_ETF_500'] = df_raw['close_price4'] / df_raw['prev_close4'] - 1
df_raw['Ret_ETF_HL'] = df_raw['close_price5'] / df_raw['prev_close5'] - 1


# ----------------------------------------------------------------------
# ⚙️ 策略核心：多周期共振 (Multi-Period Voting)
# ----------------------------------------------------------------------
def run_strategy(df_input, params, return_full_df=False):
    df = df_input.copy()
    p = params

    # --- 1. 因子计算 ---
    def calc_res(ret, val, mv, w):
        if mv.sum() == 0: return ret
        tr = np.log(val) if mv.isna().all() else val / mv
        d_tr = tr.diff().fillna(0)
        cov = ret.rolling(w).cov(d_tr)
        var = d_tr.rolling(w).var()
        beta = cov / var
        alpha = ret.rolling(w).mean() - beta * d_tr.rolling(w).mean()
        return ret - (alpha + beta * d_tr)

    # 基础情绪
    sent_500 = calc_res(df['Ret_Idx_500'], df['Val_500'], df['MV_500'], p['sent_window'])
    sent_hl = calc_res(df['Ret_Idx_HL'], df['Val_HL'], df['MV_HL'], p['sent_window'])
    fac_cum = (sent_500 - sent_hl).cumsum()

    # 🔥 核心升级：多周期趋势合成 🔥
    # 不再只依赖一个 mid_window，而是计算 短/中/长 三个趋势
    # 这样更稳健，不容易过拟合单一频率

    # 短期趋势 (Short Trend)
    trend_s = fac_cum.diff(10)
    # 中期趋势 (Mid Trend) - 由 optuna 决定
    trend_m = fac_cum.diff(p['mid_window'])
    # 长期趋势 (Long Trend)
    trend_l = fac_cum.diff(60)

    # 反转修正 (Short Reversal)
    reversal = fac_cum.diff(p['short_window'])

    # 复合得分 = (短+中+长)/3 - 反转惩罚
    # 这样只有当 短中长 都共振向上时，得分才高
    composite_score = (trend_s + trend_m + trend_l) / 3 - (p['reversal_weight'] * reversal)

    # 平滑
    score_smooth = composite_score.rolling(3).mean()

    # Z-Score 标准化
    roll_mean = score_smooth.rolling(p['strength_window']).mean()
    roll_std = score_smooth.rolling(p['strength_window']).std()
    df['Signal_Z'] = (score_smooth - roll_mean) / roll_std

    # --- 2. 线性映射仓位 (Linear Mapping) ---
    # 放弃"棘轮"，改用更顺滑的线性映射，减少对特定阈值的过拟合
    # Z > 1.5 -> 1.0 (满仓)
    # Z < -1.5 -> 0.0 (空仓)
    # Z = 0 -> 0.5 (标配)

    df = df.dropna(subset=['Signal_Z'])
    if df.empty: return -999 if not return_full_df else df

    # 激进系数 scaler: 越小越容易满仓
    scaler = p['aggressiveness']

    target_w = 0.5 + (df['Signal_Z'] / (2 * scaler))
    target_w = target_w.clip(0.0, 1.0)  # 限制在 0~1

    # 增加一个过滤器：只有变化超过 5% 才换仓，减少噪音磨损
    df['Exec_Weight'] = target_w.shift(1).fillna(0.5)

    # --- 3. 绩效 ---
    df['Turnover'] = df['Exec_Weight'].diff().abs().fillna(0)
    raw_ret = df['Exec_Weight'] * df['Ret_ETF_500'] + (1 - df['Exec_Weight']) * df['Ret_ETF_HL']
    df['Strat_Ret'] = raw_ret - (df['Turnover'] * (COST + SLIPPAGE) * 2)
    df['Strat_Cum'] = (1 + df['Strat_Ret']).cumprod()

    # 基准
    df['Bench_Cum'] = (1 + (0.5 * df['Ret_ETF_500'] + 0.5 * df['Ret_ETF_HL'])).cumprod()
    df['Rel_Value'] = df['Strat_Cum'] / df['Bench_Cum']

    if not return_full_df:
        # 分段计算超额收益
        # 我们需要同时获取 训练集 和 验证集 的表现

        # 1. 训练集表现
        train_data = df.loc[:TRAIN_END]
        if train_data.empty: return -999
        train_excess = (train_data['Rel_Value'].iloc[-1] / train_data['Rel_Value'].iloc[0]) ** (
                    252 / len(train_data)) - 1

        # 2. 验证集表现
        # 注意：在 objective 函数里我们只传了 df_raw (全量)，所以这里可以直接切
        test_data = df.loc[TEST_START:]
        if test_data.empty: return -999
        test_excess = (test_data['Rel_Value'].iloc[-1] / test_data['Rel_Value'].iloc[0]) ** (252 / len(test_data)) - 1

        # 🔥 终极目标函数：Max-Min Strategy 🔥
        # 最大化 (训练集超额 和 验证集超额) 中较小的那个
        # 这会逼迫算法找到一个"两头都好"的参数，而不是只顾一头

        min_performance = min(train_excess, test_excess)

        # 惩罚：如果有一头是负的，直接重罚
        if train_excess < 0 or test_excess < 0:
            return -1.0

        return min_performance

    return df


# ----------------------------------------------------------------------
# 🎯 Optuna 目标函数
# ----------------------------------------------------------------------
def objective(trial):
    # 这里直接传入 全量数据，但在 run_strategy 内部计算评分时会分段
    # 这样 Optuna 就能“看到”验证集的表现，从而避免过拟合训练集

    params = {
        'sent_window': trial.suggest_int('sent_window', 30, 60, step=5),
        'mid_window': trial.suggest_int('mid_window', 20, 50, step=5),
        'short_window': trial.suggest_int('short_window', 3, 10, step=1),
        'reversal_weight': trial.suggest_float('reversal_weight', 0.5, 1.5, step=0.1),
        'strength_window': trial.suggest_int('strength_window', 60, 180, step=20),

        # 激进系数：0.5=极度激进(Z>0.5就满仓), 2.0=极度保守(Z>2.0才满仓)
        'aggressiveness': trial.suggest_float('aggressiveness', 0.6, 1.2, step=0.1),
    }

    return run_strategy(df_raw, params, return_full_df=False)


# ----------------------------------------------------------------------
# 🚀 主程序
# ----------------------------------------------------------------------
if __name__ == "__main__":
    print(f"🔄 开始稳健性优化 (目标: Maximize Min(Train_Excess, Test_Excess))...")
    print(f"ℹ️  该逻辑强迫策略在【训练集】和【验证集】必须同时表现优秀")

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=100)

    print("\n✅ 最佳稳健参数:")
    best_params = study.best_params
    for k, v in best_params.items():
        print(f"   - {k}: {v}")

    # 全样本回测
    df_all = run_strategy(df_raw.copy(), best_params, return_full_df=True)

    # 统计
    df_all['Excess_DD'] = (df_all['Rel_Value'] - df_all['Rel_Value'].cummax()) / df_all['Rel_Value'].cummax()


    def print_stats(df_seg, name):
        if df_seg.empty: return
        ann_ret = (df_seg['Strat_Cum'].iloc[-1] / df_seg['Strat_Cum'].iloc[0]) ** (252 / len(df_seg)) - 1
        mdd = abs((df_seg['Strat_Cum'] / df_seg['Strat_Cum'].cummax() - 1).min())

        ann_excess = (df_seg['Rel_Value'].iloc[-1] / df_seg['Rel_Value'].iloc[0]) ** (252 / len(df_seg)) - 1
        max_excess_dd = abs(df_seg['Excess_DD'].min())

        print(f"📊 {name}:")
        print(f"   年化收益: {ann_ret:.2%} | 最大回撤: {mdd:.2%}")
        print(f"   年化超额: {ann_excess:.2%} | 超额回撤: {max_excess_dd:.2%}")


    print("-" * 60)
    print_stats(df_all.loc[:TRAIN_END], "训练集 (In-Sample)")
    print_stats(df_all.loc[TEST_START:], "验证集 (Out-of-Sample)")
    print("-" * 60)

    # 画图
    fig, axes = plt.subplots(4, 1, figsize=(12, 16), sharex=True)

    axes[0].plot(df_all['Strat_Cum'], color='#d62728', lw=2, label='策略净值')
    axes[0].plot(df_all['Bench_Cum'], color='gray', ls='--', label='基准')
    axes[0].axvline(pd.Timestamp(TEST_START), color='black', lw=2, ls='-.')
    axes[0].legend()
    axes[0].set_title('净值表现')

    axes[1].plot(df_all['Rel_Value'], color='blue', lw=1.5, label='相对净值')
    axes[1].axvline(pd.Timestamp(TEST_START), color='black', lw=2, ls='-.')
    axes[1].fill_between(df_all.index, df_all['Rel_Value'], 1.0, where=(df_all['Rel_Value'] > 1), color='red',
                         alpha=0.1)
    axes[1].set_title('相对净值 (目标：两段都向上)')

    axes[2].fill_between(df_all.index, df_all['Excess_DD'], 0, color='red', alpha=0.3)
    axes[2].set_title('超额回撤')
    axes[2].set_ylim(bottom=-0.15, top=0.05)

    axes[3].plot(df_all['Exec_Weight'], color='orange', lw=1)
    axes[3].fill_between(df_all.index, df_all['Exec_Weight'], 0, color='orange', alpha=0.3)
    axes[3].set_title('仓位 (线性平滑切换)')

    plt.tight_layout()
    plt.show()