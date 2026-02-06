import pandas as pd
import numpy as np
import os
import warnings

warnings.filterwarnings('ignore')


# ----------------------------------------------------------------------
# 缓冲带逻辑专用测试器
# ----------------------------------------------------------------------
class BufferTuner:
    def __init__(self, file_path):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件未找到: {file_path}")
        self.df = pd.read_csv(file_path, parse_dates=['TradingDay']).set_index('TradingDay').sort_index()

        # 数据清洗
        cols = ['turnover_value1', 'turnover_value2', 'close_price4', 'prev_close4', 'close_price5', 'prev_close5']
        for col in cols:
            if col in self.df.columns: self.df[col] = self.df[col].replace(0, np.nan).ffill().bfill()

        # 🌟 1. 预计算因子 (固定最佳参数 126/60)
        self.STD_W = 126
        self.RANK_W = 60
        self.SMOOTH_W = 5
        self._precalc_factor()

    def _precalc_factor(self):
        tv_500 = np.log(self.df['turnover_value1'])
        tv_hl = np.log(self.df['turnover_value2'])

        std_500 = tv_500.rolling(self.STD_W).std()
        std_hl = tv_hl.rolling(self.STD_W).std()

        factor = (std_500 - std_hl).rolling(self.SMOOTH_W).mean()
        raw_rank = factor.rolling(self.RANK_W).rank(pct=False)
        self.rank_values = ((raw_rank - 1) / (self.RANK_W - 1)).values

        self.ret_500 = (self.df['close_price4'] / self.df['prev_close4'] - 1).values
        self.ret_hl = (self.df['close_price5'] / self.df['prev_close5'] - 1).values

    def run(self, neutral_radius, label):
        # 🌟 固定参数
        LINEAR_HIGH = 0.70
        LINEAR_LOW = 0.30

        # 🌟 变量：中性区边界
        # radius=0.10 -> Neutral=[0.4, 0.6] -> Buffer=0.1 (0.3~0.4)
        NEUTRAL_L = 0.5 - neutral_radius
        NEUTRAL_H = 0.5 + neutral_radius

        targets = []
        prev_w_base = 0.5
        prev_w_final = 0.5

        denom_h = 1.0 - LINEAR_HIGH
        denom_l = LINEAR_LOW

        for r in self.rank_values:
            curr_w = 0.5
            if np.isnan(r):
                curr_w = 0.5

            # 1. 优先判断中性区 (强制重置区)
            elif NEUTRAL_L <= r <= NEUTRAL_H:
                curr_w = 0.5

            # 2. 激进区 (线性计算)
            elif r >= LINEAR_HIGH:
                progress = (r - LINEAR_HIGH) / denom_h
                curr_w = 0.5 - (progress * 0.5)
                curr_w = max(0.0, curr_w)

            elif r <= LINEAR_LOW:
                progress = (LINEAR_LOW - r) / denom_l
                curr_w = 0.5 + (progress * 0.5)
                curr_w = min(1.0, curr_w)

            # 3. 缓冲带 (Buffer Zone)
            # 既不在激进区，也不在中性区 -> 保持上一次的基础状态
            else:
                curr_w = prev_w_base

            prev_w_base = curr_w

            # === 棘轮逻辑 ===
            final_w = curr_w
            if True:  # 开启棘轮
                if curr_w > 0.5:
                    final_w = max(curr_w, prev_w_final) if prev_w_final > 0.5 else curr_w
                elif curr_w < 0.5:
                    final_w = min(curr_w, prev_w_final) if prev_w_final < 0.5 else curr_w
                else:
                    final_w = 0.5  # 只有进入中性区，这里才会变成0.5，棘轮才重置

            prev_w_final = final_w
            targets.append(final_w)

        # 回测统计
        target_exec = np.roll(np.array(targets), 1);
        target_exec[0] = 0.5
        turnover = np.abs(np.diff(target_exec, prepend=0.5))
        cost = turnover * (0.0002 + 0.0003) * 2
        net_ret = (target_exec * self.ret_500 + (1 - target_exec) * self.ret_hl) - cost

        # 截取有效段
        valid_idx = 130
        net_ret = net_ret[valid_idx:]

        cum = np.cumprod(1 + net_ret)
        ann = cum[-1] ** (252 / len(cum)) - 1
        dd = (cum - np.maximum.accumulate(cum)) / np.maximum.accumulate(cum)
        mdd = dd.min()
        return ann, mdd, turnover[valid_idx:].mean()


# ----------------------------------------------------------------------
# 运行 Step 3
# ----------------------------------------------------------------------
def run_step3_buffer():
    FILE_PATH = r"C:\Users\86178\Desktop\交易情绪因子1.csv"
    tester = BufferTuner(FILE_PATH)

    print(f"🚀 Step 3: 缓冲带/中性区宽度测试")
    print(f"📌 固定条件: Linear High=0.70, Linear Low=0.30")
    print("-" * 100)
    print(
        f"{'Label (Neutral Zone)':<30} | {'Buffer':<8} | {'Ann Return':<12} | {'Max DD':<10} | {'Calmar':<8} | {'Turnover':<10}")
    print("-" * 100)

    # 测试不同的中性区半径
    # 0.20 -> Neutral [0.3, 0.7] -> Buffer = 0 (无缓冲，即上一版定稿代码)
    # 0.10 -> Neutral [0.4, 0.6] -> Buffer = 0.1 (您偏好的老逻辑)
    # 0.00 -> Neutral [0.5, 0.5] -> Buffer = 0.2 (极粘，必须回到0.5才重置)

    params = [
        (0.20, "A. 无缓冲 [0.30, 0.70]"),
        (0.15, "B. 窄缓冲 [0.35, 0.65]"),
        (0.10, "C. 中缓冲 [0.40, 0.60]"),  # 您偏好的
        (0.05, "D. 宽缓冲 [0.45, 0.55]"),
        (0.00, "E. 极粘滞 [0.50, 0.50]")
    ]

    for r, label in params:
        ann, mdd, to = tester.run(neutral_radius=r, label=label)
        calmar = ann / abs(mdd) if mdd != 0 else 0
        buffer_size = 0.20 - r  # 计算缓冲带宽度

        print(f"{label:<30} | {buffer_size:<8.2f} | {ann:<12.2%} | {mdd:<10.2%} | {calmar:<8.2f} | {to:<10.2%}")


if __name__ == "__main__":
    run_step3_buffer()