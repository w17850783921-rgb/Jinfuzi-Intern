import pandas as pd
import numpy as np
import os

# ==========================================
# 1. 策略全参数配置
# ==========================================
FILE_PATH = r"C:\Users\86178\Desktop\TY.CBT.xlsx"

# --- 资金与交易 ---
EXCHANGE_RATE = 7.05
CONTRACT_MULTIPLIER = 1000
TRADE_UNIT = 5
FEE_RATE = 0.0004

# --- 基础指标 ---
TREND_WINDOW = 40
WINDOW = 20
BASE_GAP = 0.003
ADAPTIVE_SENSITIVITY = 1.0

# --- 过滤器 (通用) ---
RANGE_WINDOW = 30
RANGE_DIFF_THRESHOLD = 1.0
RSI_WINDOW = 14
RSI_BUY_LIMIT = 35
RSI_SELL_LIMIT = 65

# --- 模式 A：建仓期参数 (Aggressive) ---
BUILD_STD_DEV = 1.0
BUILD_BUY_MULT = 1.0
BUILD_SELL_MULT = 1.7  # 锁仓
BUILD_BUY_THR = 0.60
BUILD_SELL_THR = 1.10

# --- 模式 B：常规期参数 (Active) ---
NORMAL_STD_DEV = 1.4
NORMAL_BUY_MULT = 1.2
NORMAL_SELL_MULT = 1.0
NORMAL_BUY_THR_BULL = 0.40  # 多头买入阈值
NORMAL_BUY_THR_BEAR = 0.15  # 空头买入阈值
NORMAL_SELL_THR = 0.95  # 卖出阈值


# ==========================================
# 2. 数据处理工具
# ==========================================
def clean_price(price_val):
    if pd.isna(price_val) or price_val == '': return np.nan
    if isinstance(price_val, str):
        try:
            parts = price_val.split("'")
            return float(parts[0]) + float(parts[1]) / 10.0 / 32.0 if len(parts) > 1 else float(parts[0])
        except:
            return np.nan
    try:
        return float(price_val)
    except:
        return np.nan


def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def get_signal_price(mode, price, ma_boll, std, atr, ma_trend):
    """根据不同模式计算具体的挂单价"""
    if mode == 'build':
        curr_std = BUILD_STD_DEV
        curr_buy_m = BUILD_BUY_MULT
        curr_sell_m = BUILD_SELL_MULT
        buy_thr, sell_thr = BUILD_BUY_THR, BUILD_SELL_THR
    else:
        curr_std = NORMAL_STD_DEV
        curr_buy_m = NORMAL_BUY_MULT
        curr_sell_m = NORMAL_SELL_MULT
        # 常规期看趋势
        if price >= ma_trend:
            buy_thr = NORMAL_BUY_THR_BULL
        else:
            buy_thr = NORMAL_BUY_THR_BEAR
        sell_thr = NORMAL_SELL_THR

    # 计算该模式下的上下轨
    upper = ma_boll + curr_std * std
    lower = ma_boll - curr_std * std
    band_width = upper - lower

    # 1. 策略挂单价 (布林)
    target_buy = lower + buy_thr * band_width
    target_sell = lower + sell_thr * band_width

    # 2. 做T补仓/止盈价 (ATR)
    vol_factor = (atr / price) * ADAPTIVE_SENSITIVITY
    buy_gap = max(BASE_GAP, vol_factor * curr_buy_m)
    sell_gap = max(BASE_GAP, vol_factor * curr_sell_m)

    t_add = price * (1 - buy_gap)
    t_exit = price * (1 + sell_gap)

    return {
        'buy_limit': target_buy,
        'sell_limit': target_sell,
        't_add': t_add,
        't_exit': t_exit,
        'buy_gap_pct': buy_gap * 100,
        'sell_gap_pct': sell_gap * 100,
        'upper': upper,
        'lower': lower
    }


def calculate_real_signals(path):
    if not os.path.exists(path):
        print(f"❌ 错误：找不到文件 {path}")
        return

    # 加载数据
    df = pd.read_excel(path)
    for col in ['开盘价(元)', '最高价(元)', '最低价(元)', '收盘价(元)']:
        if col in df.columns: df[col] = df[col].apply(clean_price)
    df = df.dropna(subset=['收盘价(元)'])
    df['日期'] = pd.to_datetime(df['日期'])
    df = df.sort_values('日期').reset_index(drop=True)

    # 计算指标
    h_l = df['最高价(元)'] - df['最低价(元)']
    h_pc = (df['最高价(元)'] - df['收盘价(元)'].shift(1)).abs()
    l_pc = (df['最低价(元)'] - df['收盘价(元)'].shift(1)).abs()
    df['atr'] = pd.concat([h_l, h_pc, l_pc], axis=1).max(axis=1).rolling(14).mean()

    df['ma_boll'] = df['收盘价(元)'].rolling(WINDOW).mean()
    df['rolling_std'] = df['收盘价(元)'].rolling(WINDOW).std()
    df['ma_trend'] = df['收盘价(元)'].rolling(TREND_WINDOW).mean()
    df['roll_high'] = df['最高价(元)'].rolling(RANGE_WINDOW).max()
    df['roll_low'] = df['最低价(元)'].rolling(RANGE_WINDOW).min()
    df['rsi'] = calculate_rsi(df['收盘价(元)'], RSI_WINDOW)

    # 最新数据
    latest = df.iloc[-1]
    price = latest['收盘价(元)']
    date_str = latest['日期'].strftime('%Y-%m-%d')

    # 基础数据
    atr = latest['atr']
    rsi = latest['rsi']
    roll_high = latest['roll_high']
    roll_low = latest['roll_low']
    ma_trend = latest['ma_trend']
    ma_boll = latest['ma_boll']
    std = latest['rolling_std']

    # 分别计算两种模式的信号
    res_build = get_signal_price('build', price, ma_boll, std, atr, ma_trend)
    res_norm = get_signal_price('normal', price, ma_boll, std, atr, ma_trend)

    # 价差交易信号 (通用)
    diff_buy = roll_high - RANGE_DIFF_THRESHOLD
    diff_sell = roll_low + RANGE_DIFF_THRESHOLD

    # ================= 输出报告 =================
    print("\n" + "=" * 70)
    print(f"📠 全景实盘作战地图 | 日期: {date_str} | 收盘价: {price:.4f}")
    print("=" * 70)

    is_bull = price >= ma_trend
    trend_str = "📈 多头" if is_bull else "📉 空头"
    print(f"环境诊断: {trend_str} (MA{TREND_WINDOW}={ma_trend:.4f}) | RSI={rsi:.1f} | ATR={atr:.4f}")
    print("-" * 70)

    # 1. 独立机制 (无论什么模式都生效)
    print("💎 【独立机制】 (优先关注)")
    print(f"   📉 极值抄底: {diff_buy:.4f} (需 RSI<{RSI_BUY_LIMIT})")
    if price <= diff_buy and rsi < RSI_BUY_LIMIT:
        print("      >>> 🔥🔥🔥 极度恐慌！全仓抄底信号触发！")

    print(f"   📈 极值止盈: {diff_sell:.4f} (需 RSI>{RSI_SELL_LIMIT})")
    if price >= diff_sell and rsi > RSI_SELL_LIMIT:
        print("      >>> 🔥🔥🔥 极度贪婪！全仓止盈信号触发！")
    print("-" * 70)

    # 2. 双模式对比展示
    # 使用格式化字符串对齐
    col_w = 32
    print(f"{'🏗️  建仓模式 (持仓<10手)':<{col_w}} | {'🛡️  常规模式 (持仓>=10手)':<{col_w}}")
    print(f"{'(窄带宽/密补仓/锁仓)':<{col_w}} | {'(宽带宽/稳补仓/快跑)':<{col_w}}")
    print("-" * 70)

    # 策略买入价
    p1 = f"{res_build['buy_limit']:.4f}"
    p2 = f"{res_norm['buy_limit']:.4f}"
    # 判断是否触发
    trig1 = "⚡触发" if price <= res_build['buy_limit'] else ""
    trig2 = "⚡触发" if price <= res_norm['buy_limit'] else ""

    print(f"🔵 布林买入: {p1} {trig1:<10} | 🔵 布林买入: {p2} {trig2}")

    # 做T补仓价
    t1 = f"{res_build['t_add']:.4f} (-{res_build['buy_gap_pct']:.2f}%)"
    t2 = f"{res_norm['t_add']:.4f} (-{res_norm['buy_gap_pct']:.2f}%)"
    print(f"   做T补仓: {t1:<18} |    做T补仓: {t2}")

    # 资金预算
    cost = res_build['buy_limit'] * CONTRACT_MULTIPLIER * EXCHANGE_RATE * 5 * (1 + FEE_RATE)
    print(f"   (5手资金: ¥{cost:,.0f}){' ' * 13} |")

    print("-" * 70)

    # 策略卖出价
    s1 = f"{res_build['sell_limit']:.4f}"
    s2 = f"{res_norm['sell_limit']:.4f}"
    trig_s1 = "⚡触发" if price >= res_build['sell_limit'] else ""
    trig_s2 = "⚡触发" if price >= res_norm['sell_limit'] else ""

    print(f"🟠 布林卖出: {s1} {trig_s1:<10} | 🟠 布林卖出: {s2} {trig_s2}")

    # 做T止盈价
    te1 = f"{res_build['t_exit']:.4f} (+{res_build['sell_gap_pct']:.2f}%)"
    te2 = f"{res_norm['t_exit']:.4f} (+{res_norm['sell_gap_pct']:.2f}%)"
    print(f"   做T止盈: {te1:<18} |    做T止盈: {te2}")

    print("=" * 70)

    # 建议
    print("💡 决策建议:")
    print("   1. 如果你现在空仓或轻仓，请严格盯着左边的【建仓模式】挂单。")
    print("   2. 如果你已经有重仓（>10手），请参考右边的【常规模式】来做T降成本。")
    print("   3. 无论哪种模式，只要【独立机制】触发，优先级最高！")
    print("=" * 70)


if __name__ == "__main__":
    calculate_real_signals(FILE_PATH)