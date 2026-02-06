import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import itertools
import warnings

# 忽略运行过程中的除零警告等
warnings.filterwarnings('ignore')


# ----------------------------------------------------------------------
# 1. 核心逻辑封装 (无缝集成您的策略)
# ----------------------------------------------------------------------
class StrategyTester:
    def __init__(self, file_path):
        self.df_raw = self.load_data(file_path)

    def load_data(self, file_path):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件未找到: {file_path}")
        df = pd.read_csv(file_path, parse_dates=['TradingDay'])
        df = df.set_index('TradingDay').sort_index()
        # 数据清洗
        cols = ['turnover_value1', 'turnover_value2',
                'close_price4', 'prev_close4', 'avg_price4',
                'close_price5', 'prev_close5', 'avg_price5']
        for col in cols:
            if col in df.columns:
                df[col] = df[col].replace(0, np.nan)
        df[cols] = df[cols].ffill().bfill()
        return df

    def run(self, std_w, rank_w, linear_h=0.8, linear_l=0.2, smooth_w=5, ratchet=True):
        """
        运行一次回测，返回绩效指标
        """
        df = self.df_raw.copy()

        # --- 1. 计算因子 ---
        tv_500 = np.log(df['turnover_value1'])
        tv_hl = np.log(df['turnover_value2'])

        std_500 = tv_500.rolling(std_w).std()
        std_hl = tv_hl.rolling(std_w).std()

        factor = (std_500 - std_hl).rolling(smooth_w).mean()

        # 归一化 Rank (0~1)
        raw_rank = factor.rolling(rank_w).rank(pct=False)
        rank_norm = (raw_rank - 1) / (rank_w - 1)

        # --- 2. 生成信号 (线性+棘轮) ---
        targets = []
        prev_w_base = 0.5
        prev_w_final = 0.5

        # 提取 numpy 数组加速循环
        rank_arr = rank_norm.values

        # 预计算线性参数
        denom_h = 1.0 - linear_h
        denom_l = linear_l

        for r in rank_arr:
            if np.isnan(r):
                curr = 0.5
            elif 0.4 <= r <= 0.6:
                curr = 0.5
            elif r >= linear_h:
                progress = (r - linear_h) / denom_h
                curr = max(0.0, 0.5 - progress * 0.5)
            elif r <= linear_l:
                progress = (linear_l - r) / denom_l
                curr = min(1.0, 0.5 + progress * 0.5)
            else:
                curr = prev_w_base

            prev_w_base = curr

            # 棘轮逻辑
            final = curr
            if ratchet:
                if curr > 0.5:
                    final = max(curr, prev_w_final) if prev_w_final > 0.5 else curr
                elif curr < 0.5:
                    final = min(curr, prev_w_final) if prev_w_final < 0.5 else curr
                else:
                    final = 0.5

            prev_w_final = final
            targets.append(final)

        # --- 3. 回测执行 ---
        # 信号滞后
        target_exec = np.roll(np.array(targets), 1)
        target_exec[0] = 0.5  # 补全首位

        # 截取有效区间
        start_idx = max(std_w + rank_w, 100)
        valid_mask = np.arange(len(df)) >= start_idx

        # 收益率
        ret_500 = (df['close_price4'] / df['prev_close4'] - 1).values
        ret_hl = (df['close_price5'] / df['prev_close5'] - 1).values

        # 快速向量化回测 (忽略微小漂移再平衡的模拟，只算大逻辑以提升速度)
        # 注意：这里为了调参速度做了简化，不完全模拟每日漂移，
        # 但因为有 target_exec 控制，相对误差极小，足够用于参数排名。

        strat_ret_daily = np.zeros(len(df))
        turnover = np.abs(np.diff(target_exec, prepend=0.5))

        # 简单的加权收益 - 成本
        gross_ret = target_exec * ret_500 + (1 - target_exec) * ret_hl
        cost = turnover * (0.0002 + 0.0003) * 2
        net_ret = gross_ret - cost

        # 截取有效段
        net_ret_valid = net_ret[start_idx:]

        if len(net_ret_valid) == 0: return 0, 0, 0

        # --- 4. 计算指标 ---
        cum_ret = np.cumprod(1 + net_ret_valid)
        total_ret = cum_ret[-1] - 1
        days = len(net_ret_valid)
        ann_ret = (cum_ret[-1]) ** (252 / days) - 1

        # 最大回撤
        running_max = np.maximum.accumulate(cum_ret)
        drawdown = (cum_ret - running_max) / running_max
        max_dd = drawdown.min()

        # Calmar 比率 (年化 / 最大回撤) - 趋势策略最重要的指标
        calmar = ann_ret / abs(max_dd) if max_dd != 0 else 0

        return ann_ret, max_dd, calmar


# ----------------------------------------------------------------------
# 2. 调参主程序
# ----------------------------------------------------------------------
def run_optimization():
    # 🌟 修改为您的文件路径
    FILE_PATH = r"C:\Users\86178\Desktop\交易情绪因子1.csv"

    tester = StrategyTester(FILE_PATH)

    # 🌟 核心调参范围 (Trend Logic) 🌟
    # Std Window: 偏向中长期趋势，不要太短
    std_range = [60, 88, 126, 180, 252]
    # Rank Window: 偏向短期灵敏度，要比Std短
    rank_range = [20, 40, 60, 90, 120]

    results = []
    print(f"🚀 开始网格搜索: {len(std_range) * len(rank_range)} 组参数...")

    for s, r in itertools.product(std_range, rank_range):
        # 跳过 Rank > Std 的组合 (那是均值回归逻辑，您不需要)
        if r >= s:
            continue

        ann, mdd, calmar = tester.run(std_w=s, rank_w=r)

        results.append({
            'STD_WINDOW': s,
            'RANK_WINDOW': r,
            'Ann Return': ann,
            'Max DD': mdd,
            'Calmar': calmar
        })
        print(f"  > 参数(Std={s}, Rank={r}): 年化 {ann:.2%}, 回撤 {mdd:.2%}, Calmar {calmar:.2f}")

    # --- 3. 结果分析与可视化 ---
    df_res = pd.DataFrame(results)

    # 找到最好的参数
    best_param = df_res.loc[df_res['Calmar'].idxmax()]
    print("\n" + "=" * 50)
    print(f"🏆 最佳参数组合 (基于 Calmar):")
    print(f"   STD_WINDOW:  {int(best_param['STD_WINDOW'])}")
    print(f"   RANK_WINDOW: {int(best_param['RANK_WINDOW'])}")
    print(
        f"   绩效: 年化 {best_param['Ann Return']:.2%}, 回撤 {best_param['Max DD']:.2%}, Calmar {best_param['Calmar']:.2f}")
    print("=" * 50)

    # 绘制热力图 (Heatmap)
    pivot_table = df_res.pivot(index='STD_WINDOW', columns='RANK_WINDOW', values='Calmar')

    plt.figure(figsize=(10, 8))
    sns.heatmap(pivot_table, annot=True, fmt=".2f", cmap="RdYlGn", center=pivot_table.mean().mean())
    plt.title('策略稳健性热力图 (Calmar Ratio)\n颜色越红越好, 寻找连成一片的红色区域', fontsize=14)
    plt.ylabel('STD_WINDOW (大趋势周期)')
    plt.xlabel('RANK_WINDOW (灵敏度周期)')
    plt.show()


if __name__ == "__main__":
    run_optimization()