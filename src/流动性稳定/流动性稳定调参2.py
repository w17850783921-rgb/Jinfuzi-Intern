import pandas as pd
import numpy as np
import os
import warnings

# 忽略计算过程中的无关警告
warnings.filterwarnings('ignore')


# ----------------------------------------------------------------------
# 核心调参类：专门用于测试不同的 High/Low 阈值
# ----------------------------------------------------------------------
class ThresholdTuner:
    def __init__(self, file_path):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"❌ 文件未找到: {file_path}")

        print("⏳ 正在加载数据并预计算因子 (Std=126, Rank=60)...")
        self.df = pd.read_csv(file_path, parse_dates=['TradingDay']).set_index('TradingDay').sort_index()

        # 数据清洗
        cols = ['turnover_value1', 'turnover_value2',
                'close_price4', 'prev_close4', 'close_price5', 'prev_close5']
        for col in cols:
            if col in self.df.columns:
                self.df[col] = self.df[col].replace(0, np.nan)
        self.df[cols] = self.df[cols].ffill().bfill()

        # 🌟 1. 预先计算好最佳因子 (固定 Step 1 的结果) 🌟
        # 这样在循环测试阈值时，不用重复算因子，速度飞快
        self.STD_W = 126
        self.RANK_W = 60
        self.SMOOTH_W = 5
        self._precalc_factor()

        print("✅ 因子预计算完成，开始循环测试阈值...\n")

    def _precalc_factor(self):
        """一次性计算好 Factor Rank，后续只调仓位逻辑"""
        tv_500 = np.log(self.df['turnover_value1'])
        tv_hl = np.log(self.df['turnover_value2'])

        std_500 = tv_500.rolling(self.STD_W).std()
        std_hl = tv_hl.rolling(self.STD_W).std()

        factor = (std_500 - std_hl).rolling(self.SMOOTH_W).mean()

        # 🌟 归一化 Rank (0~1) - 确保能满仓
        raw_rank = factor.rolling(self.RANK_W).rank(pct=False)
        # 将 Series 转为 numpy array 加速循环
        self.rank_values = ((raw_rank - 1) / (self.RANK_W - 1)).values

        # 预计算收益率向量
        self.ret_500 = (self.df['close_price4'] / self.df['prev_close4'] - 1).values
        self.ret_hl = (self.df['close_price5'] / self.df['prev_close5'] - 1).values

    def run_test(self, high, low, label):
        """
        输入一组 High/Low，返回回测结果
        """
        targets = []
        prev_w_base = 0.5
        prev_w_final = 0.5

        # 🌟 动态分母计算 (关键点) 🌟
        # 无论 high 是 0.8 还是 0.9，这里都能自动适配区间长度
        denom_high = 1.0 - high
        denom_low = low  # 即 low - 0.0

        # 遍历生成信号
        for r in self.rank_values:
            if np.isnan(r):
                curr = 0.5

            # 1. 标配区 (在 Low 和 High 之间)
            elif low < r < high:
                curr = 0.5

            # 2. 减仓区 (Rank >= High)
            elif r >= high:
                # 动态线性公式：(当前 - 阈值) / (1 - 阈值)
                progress = (r - high) / denom_high
                curr = 0.5 - (progress * 0.5)
                curr = max(0.0, curr)

            # 3. 加仓区 (Rank <= Low)
            elif r <= low:
                # 动态线性公式：(阈值 - 当前) / 阈值
                progress = (low - r) / denom_low
                curr = 0.5 + (progress * 0.5)
                curr = min(1.0, curr)

            else:
                curr = prev_w_base

            prev_w_base = curr

            # === 单边棘轮逻辑 (Ratchet) ===
            final = curr
            # 开启棘轮
            if curr > 0.5:
                final = max(curr, prev_w_final) if prev_w_final > 0.5 else curr
            elif curr < 0.5:
                final = min(curr, prev_w_final) if prev_w_final < 0.5 else curr
            else:
                final = 0.5

            prev_w_final = final
            targets.append(final)

        # === 快速回测统计 ===
        # T+1 执行
        target_exec = np.roll(np.array(targets), 1)
        target_exec[0] = 0.5

        # 换手率 (简单估算)
        turnover = np.abs(np.diff(target_exec, prepend=0.5))

        # 扣费收益 (Cost=万2, Slip=万3 -> 双边万10 = 0.001)
        total_cost_rate = (0.0002 + 0.0003) * 2
        costs = turnover * total_cost_rate

        # 组合收益
        strat_ret = target_exec * self.ret_500 + (1 - target_exec) * self.ret_hl
        net_ret = strat_ret - costs

        # 截取有效回测区间 (跳过因子预热期)
        # Std(126) + Rank(60) ≈ 186天
        valid_idx = 200
        net_ret_valid = net_ret[valid_idx:]
        turnover_valid = turnover[valid_idx:]

        if len(net_ret_valid) == 0: return 0, 0, 0, 0

        # 计算指标
        cum = np.cumprod(1 + net_ret_valid)
        ann_ret = cum[-1] ** (252 / len(cum)) - 1

        running_max = np.maximum.accumulate(cum)
        dd = (cum - running_max) / running_max
        max_dd = dd.min()

        calmar = ann_ret / abs(max_dd) if max_dd != 0 else 0
        avg_turnover = turnover_valid.mean()

        return ann_ret, max_dd, calmar, avg_turnover


# ----------------------------------------------------------------------
# 主程序：执行 Step 2 调参
# ----------------------------------------------------------------------
def run_step2_tuning():
    # 🌟 修改为您的文件路径
    FILE_PATH = r"C:\Users\86178\Desktop\交易情绪因子1.csv"

    try:
        tuner = ThresholdTuner(FILE_PATH)
    except Exception as e:
        print(e)
        return

    # 🌟 定义要测试的动态阈值组合 🌟
    # 格式: (High, Low, 描述)
    test_params = [
        (0.95, 0.05, "A. 极度保守 (0.95/0.05)"),
        (0.90, 0.10, "B. 狙击模式 (0.90/0.10)"),
        (0.85, 0.15, "C. 适度稳健 (0.85/0.15)"),
        (0.80, 0.20, "D. 当前基准 (0.80/0.20)"),  # 您现在的参数
        (0.75, 0.25, "E. 适度积极 (0.75/0.25)"),
        (0.70, 0.30, "F. 活跃模式 (0.70/0.30)"),
        (0.60, 0.40, "G. 极度激进 (0.60/0.40)"),
        (0.55, 0.45, "H. 疯狂模式 (0.55/0.45)")
    ]

    print("-" * 100)
    print(f"{'Label':<25} | {'Ann Return':<12} | {'Max DD':<10} | {'Calmar':<8} | {'Turnover':<10} | {'Score'}")
    print("-" * 100)

    best_score = -999
    best_cfg = None

    for high, low, label in test_params:
        ann, mdd, calmar, to = tuner.run_test(high, low, label)

        # 评分逻辑：Calmar最重要，但如果换手率太高(>3%)要扣分
        # 这是一个简单的综合打分，供参考
        penalty = 0
        if to > 0.03: penalty = (to - 0.03) * 100  # 换手惩罚
        score = calmar - penalty

        print(f"{label:<25} | {ann:<12.2%} | {mdd:<10.2%} | {calmar:<8.2f} | {to:<10.2%} | {score:.2f}")

        if score > best_score:
            best_score = score
            best_cfg = (high, low, label)

    print("-" * 100)
    print(f"🏆 推荐最佳参数: {best_cfg[2]}")
    print(f"   High (减仓阈值): {best_cfg[0]}")
    print(f"   Low  (加仓阈值): {best_cfg[1]}")
    print("   💡 理由: 在风险收益比(Calmar)和交易成本(Turnover)之间达到了最佳平衡。")


if __name__ == "__main__":
    run_step2_tuning()