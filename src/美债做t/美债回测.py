import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ==========================================
# 1. 策略参数配置 (最优定稿版)
# ==========================================
FILE_PATH = r"C:\Users\86178\Desktop\TY.CBT.xlsx"
START_DATE = '2025-02-01'
END_DATE = '2099-12-31'

INITIAL_CASH = 20000000
EXCHANGE_RATE = 7.05
CONTRACT_MULTIPLIER = 1000
TRADE_UNIT = 5
MAX_LOTS = 25
FEE_RATE = 0.0004

# --- 基础参数 ---
TREND_WINDOW = 40
WINDOW = 20
BASE_GAP = 0.003
ADAPTIVE_SENSITIVITY = 1.0

# ==========================================
# 🔥 核心：最优参数填入
# ==========================================

# --- A. 降噪过滤器 (Step 1 最佳结果) ---
RANGE_WINDOW = 30
RANGE_DIFF_THRESHOLD = 1.0  # 价差 > 1.0
RSI_WINDOW = 14
RSI_BUY_LIMIT = 35  # RSI < 35 抄底
RSI_SELL_LIMIT = 65  # RSI > 65 止盈

# --- B. 建仓期模式 (Step 2 最佳结果) ---
BUILD_PHASE_LIMIT_LOTS = 10  # 持仓 < 10手
BUILD_STD_DEV = 1.0  # 极窄带宽，触碰即买
BUILD_BUY_MULT = 1.0  # 正常间距
BUILD_SELL_MULT = 1.7  # 【关键】极大卖出间距，锁仓囤货

# --- C. 常规期模式 (Step 3 激进版结果) ---
NORMAL_STD_DEV = 1.4  # 活跃带宽 (原2.2 -> 1.4)
NORMAL_BUY_MULT = 1.2  # 活跃买入 (原1.5 -> 1.2)
NORMAL_SELL_MULT = 1.0  # 活跃卖出 (原0.8 -> 1.0)

# --- D. 辅助机制 ---
TIME_FORCE_DAYS = 10
TIME_PRICE_CAP = 1.005


# ==========================================
# 2. 数据工具
# ==========================================
def clean_price(price_val):
    if pd.isna(price_val) or price_val == '': return np.nan
    if isinstance(price_val, str):
        price_val = price_val.strip()
        if "'" in price_val:
            parts = price_val.split("'")
            try:
                return float(parts[0]) + float(parts[1]) / 10.0 / 32.0 if parts[1] else float(parts[0])
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


def load_data(path):
    if not os.path.exists(path): return None
    df = pd.read_excel(path)
    for col in ['开盘价(元)', '最高价(元)', '最低价(元)', '收盘价(元)']:
        if col in df.columns: df[col] = df[col].apply(clean_price)
    df = df.dropna(subset=['收盘价(元)'])
    df['日期'] = pd.to_datetime(df['日期'])
    return df.sort_values('日期').reset_index(drop=True)


# ==========================================
# 3. 策略引擎 (Optimized)
# ==========================================
class OptimalStrategy:
    def __init__(self):
        self.daily_total_asset = []
        self.trade_records = []
        self.cash = INITIAL_CASH
        self.hold_lots = 0
        self.trade_count = 0
        self.last_buy_price = 0
        self.last_sell_price = 0
        self.last_action_type = None
        self.last_buy_index = -999

    def run(self, df_full):
        # 预计算指标
        h_l = df_full['最高价(元)'] - df_full['最低价(元)']
        h_pc = (df_full['最高价(元)'] - df_full['收盘价(元)'].shift(1)).abs()
        l_pc = (df_full['最低价(元)'] - df_full['收盘价(元)'].shift(1)).abs()
        df_full['atr'] = pd.concat([h_l, h_pc, l_pc], axis=1).max(axis=1).rolling(14).mean()

        df_full['ma_boll'] = df_full['收盘价(元)'].rolling(WINDOW).mean()
        df_full['rolling_std'] = df_full['收盘价(元)'].rolling(WINDOW).std()
        df_full['ma_trend'] = df_full['收盘价(元)'].rolling(TREND_WINDOW).mean()

        df_full['roll_high'] = df_full['最高价(元)'].rolling(RANGE_WINDOW).max()
        df_full['roll_low'] = df_full['最低价(元)'].rolling(RANGE_WINDOW).min()
        df_full['rsi'] = calculate_rsi(df_full['收盘价(元)'], RSI_WINDOW)

        mask = (df_full['日期'] >= START_DATE) & (df_full['日期'] <= END_DATE)
        df = df_full.loc[mask].reset_index(drop=True)

        upper_rec, lower_rec, mode_rec = [], [], []

        for idx, row in df.iterrows():
            date, price = row['日期'], row['收盘价(元)']
            ma_boll, r_std, ma_trend, atr = row['ma_boll'], row['rolling_std'], row['ma_trend'], row['atr']
            roll_high, roll_low, rsi = row['roll_high'], row['roll_low'], row['rsi']

            if pd.isna(ma_boll) or pd.isna(atr):
                upper_rec.append(np.nan);
                lower_rec.append(np.nan);
                mode_rec.append(0)
                self._record_daily(price)
                continue

            # ==============================
            # 🔄 1. 模式切换
            # ==============================
            is_building = self.hold_lots < BUILD_PHASE_LIMIT_LOTS

            if is_building:
                # 建仓期：易买难卖 (Std=1.0, Sell=1.7)
                curr_std, curr_buy_m, curr_sell_m = BUILD_STD_DEV, BUILD_BUY_MULT, BUILD_SELL_MULT
                mode = 1
            else:
                # 常规期：活跃交易 (Std=1.4, Sell=1.0)
                curr_std, curr_buy_m, curr_sell_m = NORMAL_STD_DEV, NORMAL_BUY_MULT, NORMAL_SELL_MULT
                mode = 0

            # 动态布林带
            upper = ma_boll + curr_std * r_std
            lower = ma_boll - curr_std * r_std
            upper_rec.append(upper);
            lower_rec.append(lower);
            mode_rec.append(mode)

            # 动态间距
            vol_factor = (atr / price) * ADAPTIVE_SENSITIVITY
            buy_gap = max(BASE_GAP, vol_factor * curr_buy_m)
            sell_gap = max(BASE_GAP, vol_factor * curr_sell_m)

            contract_val_rmb = price * CONTRACT_MULTIPLIER * EXCHANGE_RATE
            pb = (price - lower) / (upper - lower) if (upper - lower) != 0 else 0.5

            # 动态趋势阈值
            if is_building:
                buy_thr, sell_thr = 0.60, 1.10  # 建仓期放宽买入
            else:
                if price >= ma_trend:
                    buy_thr, sell_thr = 0.40, 0.95
                else:
                    buy_thr, sell_thr = 0.15, 0.95

            # ==============================
            # 🧠 2. 信号决策
            # ==============================
            final_action = None
            action_tag = ""

            # --- 买入检测 ---
            if self.hold_lots < MAX_LOTS:
                # A. 布林策略 (根据模式不同，参数不同)
                if pb <= buy_thr and (self.last_action_type != 'buy' or price < self.last_buy_price * (1 - buy_gap)):
                    final_action = 'buy'
                    action_tag = "建仓吸筹" if mode == 1 else "活跃加仓"

                # B. 时间补仓 (仅建仓期)
                elif is_building and self.last_buy_index != -999 and (
                        idx - self.last_buy_index >= TIME_FORCE_DAYS) and (
                        price < self.last_buy_price * TIME_PRICE_CAP):
                    final_action = 'buy'
                    action_tag = "⏳时间补仓"

                # C. 价差抄底 (Diff=1.0, RSI<35)
                elif not pd.isna(roll_high) and price <= (roll_high - RANGE_DIFF_THRESHOLD):
                    if rsi < RSI_BUY_LIMIT and (
                            self.last_action_type != 'buy' or price < self.last_buy_price * (1 - BASE_GAP)):
                        final_action = 'buy'
                        action_tag = f"📉极值抄底(RSI={rsi:.0f})"

            # --- 卖出检测 ---
            if final_action is None and self.hold_lots > 0:
                # A. 布林策略 (建仓期Sell=1.7很难卖，常规期Sell=1.0容易卖)
                if pb >= sell_thr and (
                        self.last_action_type != 'sell' or price > self.last_sell_price * (1 + sell_gap)):
                    final_action = 'sell'
                    action_tag = "策略止盈"

                # B. 价差止盈 (Diff=1.0, RSI>65)
                elif not pd.isna(roll_low) and price >= (roll_low + RANGE_DIFF_THRESHOLD):
                    if rsi > RSI_SELL_LIMIT and (
                            self.last_action_type != 'sell' or price > self.last_sell_price * (1 + BASE_GAP)):
                        final_action = 'sell'
                        action_tag = f"📈极值止盈(RSI={rsi:.0f})"

            # ==============================
            # 🎬 3. 执行交易
            # ==============================
            if final_action == 'buy':
                cost = TRADE_UNIT * contract_val_rmb * (1 + FEE_RATE)
                if self.cash >= cost:
                    self.cash -= cost
                    self.hold_lots += TRADE_UNIT
                    self.trade_count += 1
                    self.last_buy_price = price
                    self.last_action_type = 'buy'
                    self.last_buy_index = idx
                    self.trade_records.append({'date': date, 'price': price, 'type': 'buy', 'tag': action_tag})

            elif final_action == 'sell':
                sell_lots = min(self.hold_lots, TRADE_UNIT)
                # 清洗零头
                if self.hold_lots < TRADE_UNIT * 1.5: sell_lots = self.hold_lots

                revenue = sell_lots * contract_val_rmb * (1 - FEE_RATE)
                self.cash += revenue
                self.hold_lots -= sell_lots
                self.trade_count += 1
                self.last_sell_price = price
                self.last_action_type = 'sell'
                self.trade_records.append({'date': date, 'price': price, 'type': 'sell', 'tag': action_tag})

            self._record_daily(price)

        df['upper'] = upper_rec
        df['lower'] = lower_rec
        df['mode'] = mode_rec
        return df

    def _record_daily(self, price):
        market_val = self.hold_lots * price * CONTRACT_MULTIPLIER * EXCHANGE_RATE
        total = self.cash + market_val
        self.daily_total_asset.append(total)

    def plot(self, df):
        dates = df['日期'].values
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12), sharex=True)

        # 1. 资金曲线
        ax1.plot(dates, self.daily_total_asset, color='#d62728', linewidth=2, label='最优策略净值')
        ax1.axhline(y=INITIAL_CASH, color='gray', linestyle='--', label='初始本金')
        ax1.set_title(f'资金曲线 (Build:锁仓 / Normal:活跃)')
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)

        # 2. 交易点位
        ax2.plot(dates, df['收盘价(元)'], color='black', linewidth=1, label='价格')
        ax2.fill_between(dates, df['upper'], df['lower'], color='blue', alpha=0.1, label='动态布林带')

        # 背景色：黄色=建仓期
        for i in range(len(dates) - 1):
            if df['mode'].iloc[i] == 1:
                ax2.axvspan(dates[i], dates[i + 1], color='yellow', alpha=0.1, linewidth=0)

        # 提取点位
        buys = [x for x in self.trade_records if x['type'] == 'buy']
        sells = [x for x in self.trade_records if x['type'] == 'sell']

        # 分类显示
        b_norm = [x for x in buys if '时间' not in x['tag'] and '极值' not in x['tag']]
        b_time = [x for x in buys if '时间' in x['tag']]
        b_range = [x for x in buys if '极值' in x['tag']]

        if b_norm: ax2.scatter([x['date'] for x in b_norm], [x['price'] for x in b_norm], marker='^', color='red', s=50,
                               label='策略买入')
        if b_time: ax2.scatter([x['date'] for x in b_time], [x['price'] for x in b_time], marker='D', color='purple',
                               s=80, label='时间补仓', zorder=6)
        if b_range: ax2.scatter([x['date'] for x in b_range], [x['price'] for x in b_range], marker='*', color='orange',
                                s=150, label='极值抄底', zorder=6)

        if sells: ax2.scatter([x['date'] for x in sells], [x['price'] for x in sells], marker='v', color='green', s=50,
                              label='卖出')

        ax2.set_title('交易点位分布 (黄色区域=建仓期)')
        ax2.legend(loc='lower left', ncol=2)
        ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    df = load_data(FILE_PATH)
    if df is not None:
        strat = OptimalStrategy()
        df_res = strat.run(df)

        profit = strat.daily_total_asset[-1] - INITIAL_CASH
        # 计算回撤
        equity = np.array(strat.daily_total_asset)
        peak = np.maximum.accumulate(equity)
        dd = np.max(peak - equity)

        print("\n" + "=" * 60)
        print(f"📊 策略报告 (全参数最优版)")
        print("=" * 60)
        print(f"💰 净利润:       {profit:,.0f} RMB")
        print(f"📈 收益率:       {(profit / INITIAL_CASH) * 100:.2f}%")
        print(f"📉 最大回撤:     {dd:,.0f} RMB")
        print(f"🔄 总交易次数:   {strat.trade_count} 次")
        print("-" * 60)
        print("💡 策略特性验证:")
        print(f"   [建仓期] 窄带(Std=1.0) + 锁仓(Sell=1.7) -> 快速囤满10手。")
        print(f"   [常规期] 活跃(Std=1.4) + 快跑(Sell=1.0) -> 重仓时灵活做T。")
        print(f"   [过滤器] Diff=1.0 + RSI<35 -> 只在真正的恐慌盘出手。")
        print("=" * 60)

        strat.plot(df_res)