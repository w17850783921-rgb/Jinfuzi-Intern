import pandas as pd
import numpy as np
import itertools
import time
import os
import csv
from typing import Dict, List, Any, Tuple

# 尝试导入进度条库，没有也不影响运行
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterator, **kwargs):
        return iterator

import warnings

warnings.filterwarnings('ignore')

# ----------------------------------------------------------------------
# 📌 1. 配置区域：目标公式与参数范围
# ----------------------------------------------------------------------
FILE_PATH = r"C:\Users\86178\Desktop\资金大中小因子.xlsx"
RESULT_FILE = "grid_search_final_result.csv"  # 结果保存文件


# === 目标函数定义 (不变) ===
def calculate_score(excess: float, sharpe: float, max_dd: float) -> float:
    """
    目标：(超额 * 夏普) / |回撤|
    """
    if excess <= 0: return -999.0
    abs_dd = abs(max_dd)
    if abs_dd < 0.001: abs_dd = 0.001
    score = (excess * sharpe) / abs_dd
    return score


# === 参数池 (不变) ===
PARAM_GRID = {
    'z_window': np.arange(20, 85, 1).tolist(),
    'factor_smooth': [5, 10, 15, 20],
    'neutral_th': np.round(np.arange(0.30, 0.60, 0.1), 2).tolist(),
    'max_th': np.round(np.arange(2.0, 4.1, 0.5), 2).tolist(),
    'req_days': [1, 2, 3, 5],
    'ma_window': [10, 20, 30]
}


# ----------------------------------------------------------------------
# 📌 2. 数据预处理与加速缓存 (已修改：新增数据划分)
# ----------------------------------------------------------------------

# Helper function to process the DataFrame slice
def process_data_slice(df_slice, param_grid):
    """
    处理 DataFrame 切片，生成 data_dict, ma_cache, factor_cache。
    """
    # 基础数据转 Numpy
    data_dict = {
        'p1': df_slice['close_price1'].values,
        'p2': df_slice['close_price2'].values,
        'r1': df_slice['index_return1'].values,
        'r2': df_slice['index_return2'].values,
        'rf1': ((df_slice['buy_value_xl1'] + df_slice['buy_value_l1']) - (
                    df_slice['sell_value_xl1'] + df_slice['sell_value_l1'])).values,
        'rf2': ((df_slice['buy_value_xl2'] + df_slice['buy_value_l2']) - (
                    df_slice['sell_value_xl2'] + df_slice['sell_value_l2'])).values
    }

    # 1. 预计算所有均线 (MA)
    ma_cache = {}
    for w in param_grid['ma_window']:
        # 使用 min_periods=1 避免大量 NaN
        ma_cache[f'ma1_{w}'] = df_slice['close_price1'].rolling(w, min_periods=1).mean().values
        ma_cache[f'ma2_{w}'] = df_slice['close_price2'].rolling(w, min_periods=1).mean().values

    # 2. 预计算所有因子组合 (Z-Score)
    factor_cache = {}

    def calc_raw_ratio(suffix):
        net = (df_slice[f'buy_value_xl{suffix}'] + df_slice[f'buy_value_l{suffix}']) - \
              (df_slice[f'sell_value_xl{suffix}'] + df_slice[f'sell_value_l{suffix}']) - \
              (df_slice[f'buy_value_s{suffix}'] - df_slice[f'sell_value_s{suffix}'])
        mkt = df_slice[f'free_float_mktval{suffix}']
        return net / mkt

    ratio1 = calc_raw_ratio('1')
    ratio2 = calc_raw_ratio('2')

    for fs, zw in itertools.product(param_grid['factor_smooth'], param_grid['z_window']):
        r1_smooth = ratio1.rolling(fs, min_periods=1).sum()
        r2_smooth = ratio2.rolling(fs, min_periods=1).sum()

        # Z-Score
        # 注意：这里使用 min_periods=1 来避免初始的大量 NaN 影响后续数据
        z1 = (r1_smooth - r1_smooth.rolling(zw, min_periods=1).mean()) / r1_smooth.rolling(zw, min_periods=1).std()
        z2 = (r2_smooth - r2_smooth.rolling(zw, min_periods=1).mean()) / r2_smooth.rolling(zw, min_periods=1).std()

        factor_cache[f'{fs}_{zw}'] = (z1 - z2).values

    return data_dict, ma_cache, factor_cache


def prepare_data_and_cache_split(file_path: str, param_grid: Dict[str, List[Any]], split_ratio: float = 0.7) -> Tuple[
    Tuple, Tuple]:
    """
    读取数据，按时间轴 7:3 划分训练集和测试集，并对两者分别进行预计算。
    """
    if not os.path.exists(file_path):
        print(f"🚨 错误：找不到文件路径 {file_path}")
        return (None, None, None), (None, None, None)

    print(f"正在读取数据并进行 7:3 划分加速: {file_path}")
    df = pd.read_excel(file_path)
    df['TradingDay'] = pd.to_datetime(df['TradingDay'])
    df = df.sort_values('TradingDay').reset_index(drop=True)

    # 划分索引
    n = len(df)
    split_index = int(n * split_ratio)

    df_train = df.iloc[:split_index].copy().reset_index(drop=True)
    df_test = df.iloc[split_index:].copy().reset_index(drop=True)

    print(f"原始数据总长度: {n} 天")
    print(
        f"🔑 训练集长度 ({split_ratio * 100:.0f}%): {len(df_train)} 天 ({df_train['TradingDay'].iloc[0].date()} 至 {df_train['TradingDay'].iloc[-1].date()})")
    print(
        f"🔒 测试集长度 ({(1 - split_ratio) * 100:.0f}%): {len(df_test)} 天 ({df_test['TradingDay'].iloc[0].date()} 至 {df_test['TradingDay'].iloc[-1].date()})")

    # 对训练集进行预计算
    train_data, train_ma, train_factor = process_data_slice(df_train, param_grid)
    print("--- 训练集预计算完成 ---")

    # 对测试集进行预计算
    test_data, test_ma, test_factor = process_data_slice(df_test, param_grid)
    print("--- 测试集预计算完成 ---")

    return (train_data, train_ma, train_factor), (test_data, test_ma, test_factor)


# ----------------------------------------------------------------------
# 📌 3. 极速回测内核 (不变)
# ----------------------------------------------------------------------
# 保持 fast_backtest 函数不变，因为它处理的是传入的 data, ma_cache, factor_cache 结构

def fast_backtest(params: Dict[str, float], data: Dict[str, np.ndarray], ma_cache: Dict[str, np.ndarray],
                  factor_cache: Dict[str, np.ndarray]) -> Tuple[float, float, float]:
    # 解包参数
    zw = int(params['z_window'])
    fs = int(params['factor_smooth'])
    nt = params['neutral_th']
    mt = params['max_th']
    rd = int(params['req_days'])
    mw = int(params['ma_window'])

    # 从缓存获取数组
    spread = factor_cache[f'{fs}_{zw}']
    ma1 = ma_cache[f'ma1_{mw}']
    ma2 = ma_cache[f'ma2_{mw}']

    p1 = data['p1']
    p2 = data['p2']
    rf1 = data['rf1']
    rf2 = data['rf2']
    r1 = data['r1']
    r2 = data['r2']

    n = len(spread)
    target_weights = np.full(n, 0.5)

    # 状态变量（需要串行迭代）
    last_locked = 0.5
    cb = 0  # consecutive bull
    cbr = 0  # consecutive bear
    range_width = mt - nt

    # --- 核心循环 ---
    for i in range(n):
        s = spread[i]

        if np.isnan(s) or np.isnan(ma1[i]):
            target_weights[i] = last_locked
            continue

        # 计数器
        if s > nt:
            cb += 1;
            cbr = 0
        elif s < -nt:
            cbr += 1;
            cb = 0
        else:
            cb = 0;
            cbr = 0

        curr = last_locked

        # 决策
        if abs(s) <= nt:
            curr = 0.5
            last_locked = 0.5
        elif s > nt:  # Index1 强势
            if (cb >= rd) and (p1[i] > ma1[i]) and (rf1[i] > 0):
                raw = 0.5 + 0.5 * ((s - nt) / range_width)
                if raw > 1.0: raw = 1.0
                if last_locked < 0.5:
                    curr = raw
                else:
                    curr = raw if raw > last_locked else last_locked
                last_locked = curr
        else:  # s < -nt，Index2 强势
            if (cbr >= rd) and (p2[i] > ma2[i]) and (rf2[i] > 0):
                raw = 0.5 - 0.5 * ((abs(s) - nt) / range_width)
                if raw < 0.0: raw = 0.0
                if last_locked > 0.5:
                    curr = raw
                else:
                    curr = raw if raw < last_locked else last_locked
                last_locked = curr

        target_weights[i] = curr

    # 信号滞后一天
    targets = np.roll(target_weights, 1);
    targets[0] = 0.5

    # --- 快速算净值 (考虑交易成本) ---
    nav_s = np.zeros(n)
    nav_b = np.ones(n)

    # 基准净值
    b1, b2 = 0.5, 0.5
    for i in range(n):
        b1 *= (1 + r1[i])
        b2 *= (1 + r2[i])
        nav_b[i] = b1 + b2

    # 策略净值
    h1, h2 = 0.5, 0.5
    prev_w = 0.5
    cost_rate = 0.0001

    for i in range(n):
        w = targets[i]
        rr1, rr2 = r1[i], r2[i]

        # 调仓
        if abs(w - prev_w) > 0.001:
            tot = h1 + h2
            t1 = tot * w
            t2 = tot * (1 - w)
            c = (abs(t1 - h1) + abs(t2 - h2)) * cost_rate
            ntot = tot - c
            h1 = ntot * w
            h2 = ntot * (1 - w)
            prev_w = w

        # 每日涨跌幅更新
        h1 *= (1 + rr1)
        h2 *= (1 + rr2)
        nav_s[i] = h1 + h2

    # --- 指标计算 ---
    total_ret = nav_s[-1] - 1
    bench_ret = nav_b[-1] - 1
    excess = total_ret - bench_ret

    # 夏普比率
    pct = np.diff(nav_s) / nav_s[:-1]
    annual_rf = 0.02
    daily_rf = annual_rf / 250
    if len(pct) < 2 or np.std(pct) < 1e-6:
        sharpe = 0
    else:
        sharpe = (np.mean(pct) - daily_rf) / np.std(pct) * np.sqrt(250)

    # 最大回撤
    cummax = np.maximum.accumulate(nav_s)
    dd = nav_s / cummax - 1
    max_dd = np.min(dd)

    return excess, sharpe, max_dd


# ----------------------------------------------------------------------
# 📌 4. 主程序：在训练集上搜索，在测试集上验证 (已修改)
# ----------------------------------------------------------------------
def main():
    # 1. 准备数据：获取 70% 训练集和 30% 测试集
    (train_data, train_ma, train_factor), (test_data, test_ma, test_factor) = prepare_data_and_cache_split(FILE_PATH,
                                                                                                           PARAM_GRID)

    if train_data is None: return

    # 2. 生成所有参数组合
    keys = list(PARAM_GRID.keys())
    valid_combinations = []
    param_values = [PARAM_GRID[k] for k in keys]

    # 过滤掉 max_th <= neutral_th 的无效组合
    for values in itertools.product(*param_values):
        params = dict(zip(keys, values))
        if params['max_th'] > params['neutral_th']:
            valid_combinations.append(params)

    total_combs = len(valid_combinations)
    results_to_write: List[List[Any]] = []

    print(f"\n🚀 开始在【训练集】上进行网格搜索优化...")
    print(f"总计有效组合数: {total_combs}")
    print("-" * 50)

    # 3. 在【训练集】上循环回测和优化
    start_time = time.time()
    best_train_score = -999.0
    best_params = None

    for i, params in tqdm(enumerate(valid_combinations), total=total_combs, unit="comb"):

        processed_params = {k: float(v) for k, v in params.items()}

        # ⚠️ 仅在【训练集】上执行回测
        excess, sharpe, max_dd = fast_backtest(processed_params, train_data, train_ma, train_factor)

        # 计算得分
        score = calculate_score(excess, sharpe, max_dd)

        # 记录最优参数
        if score > best_train_score:
            best_train_score = score
            best_params = params

        # 收集结果行
        row_values = list(params.values())
        row = row_values + [excess, sharpe, max_dd, score]
        results_to_write.append(row)

    end_time = time.time()
    duration = end_time - start_time

    # 4. 一次性写入网格搜索结果
    print(f"\n正在写入 {len(results_to_write)} 条【训练集】搜索结果到文件...")
    headers = keys + ['Excess', 'Sharpe', 'MaxDD', 'Score']

    with open(RESULT_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(results_to_write)

    print(f"\n✅ 训练集搜索完成！耗时: {duration / 3600:.2f} 小时 ({duration:.2f} 秒)")
    print("-" * 50)

    # 5. 在【测试集】上进行验证
    if best_params is not None and test_data[0] is not None:
        print("\n🏆 开始在【测试集】上验证最优参数...")

        processed_best_params = {k: float(v) for k, v in best_params.items()}

        # ⚠️ 在【测试集】上执行单次回测
        test_excess, test_sharpe, test_max_dd = fast_backtest(processed_best_params, test_data, test_ma, test_factor)
        test_score = calculate_score(test_excess, test_sharpe, test_max_dd)

        print("\n⭐⭐⭐ 验证结果 ⭐⭐⭐")
        print(f"选定最优参数 (基于训练集 Score={best_train_score:.4f}): {best_params}")
        print(f"【测试集】绩效 (Score={test_score:.4f}):")
        print(f"  - 超额收益(Excess): {test_excess:.2%}")
        print(f"  - 夏普比率(Sharpe): {test_sharpe:.2f}")
        print(f"  - 最大回撤(MaxDD): {test_max_dd:.2%}")

        # 评估泛化能力
        if test_score > 0 and test_excess > 0:
            print("\n🎉 结论：测试集表现良好，策略泛化能力强！")
        else:
            print("\n⚠️ 结论：测试集表现不佳，可能存在过度拟合 (Overfitting)！")
    else:
        print("未找到有效参数或测试集数据缺失，跳过验证。")


# ----------------------------------------------------------------------
# 📌 5. 结果分析 (已修改：只分析训练集结果)
# ----------------------------------------------------------------------
def analyze_results():
    if not os.path.exists(RESULT_FILE):
        print("找不到结果文件")
        return

    print(f"\n正在分析【训练集】网格搜索最佳结果...")
    df = pd.read_csv(RESULT_FILE)
    df = df[df['Score'] > 0]

    if df.empty:
        print("没有找到 Score 大于 0 的有效参数组合。")
        return

    df = df.sort_values(by='Score', ascending=False).reset_index(drop=True)

    print("🏆 Top 5 参数组合 (训练集)")
    print(df.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
    # 跑完后自动分析训练集结果
    analyze_results()