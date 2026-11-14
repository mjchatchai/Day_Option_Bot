"""
Interactive HTML Dashboard Builder for Day Option Bot.
Generates `docs/index.html` with:
- Real-Time EMA 200 + Previous-Day Fibonacci Map (0.000 to 1.000) for all Watchlist tickers
- Real-Time Black-Scholes Option Pricing (Zero 15-min Delay) + Max Limit Buy + TP1/TP2/SL
- Paper Trade Win Rate Scoreboard & 90-Day Backtest Leaderboard
"""

import json
import os
from typing import Dict, List


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DASHBOARD_PATH = os.path.join(BASE_DIR, "docs", "index.html")
BACKTEST_PATH = os.path.join(BASE_DIR, "config", "backtest_90d.json")


def build_html_dashboard(scan_output: Dict, paper_summary: Dict, out_path: str = DASHBOARD_PATH) -> str:
    kpis = paper_summary.get("kpis", {})
    results: List[Dict] = scan_output.get("results", [])

    bt_data = {}
    if os.path.exists(BACKTEST_PATH):
        try:
            with open(BACKTEST_PATH, "r", encoding="utf-8") as f:
                bt_data = json.load(f)
        except Exception:
            bt_data = {}

    cards_html = []
    for item in results:
        if "fibo_levels" not in item:
            continue
        tk = item["ticker"]
        st = item["status"]
        act = item["option_action"]
        is_put = "PUT" in act
        dir_badge_cls = "badge-put" if is_put else "badge-call"
        st_badge_cls = (
            "badge-actionable"
            if st == "ACTIONABLE"
            else ("badge-near" if st == "NEAR_SETUP" else ("badge-blackout" if st == "EARNINGS_BLACKOUT" else "badge-watch"))
        )

        fibo = item["fibo_levels"]
        plan = item["trade_plan"]
        oq = item["option_quote"]
        candle = item["candle_reversal"]

        cards_html.append(f"""
        <div class="card">
          <div class="card-header">
            <div>
              <span class="ticker-title">{tk}</span>
              <span class="badge {dir_badge_cls}">{act}</span>
              <span class="badge {st_badge_cls}">{st}</span>
            </div>
            <div class="price-box">
              <div class="spot-price">${item['spot_price']:.2f}</div>
              <div class="sub-meta">EMA200: ${item['ema200']:.2f} | RVOL: {item['rvol']:.2f}x</div>
            </div>
          </div>

          <div class="status-banner">{item['status_thai']}</div>

          <div class="section-title">📐 กรอบ Fibonacci วันก่อนหน้า ({item['prev_trading_date']})</div>
          <div class="anchor-info">จุด 1: <b>{fibo['anchor_point_1_label']}</b> &rarr; จุด 2: <b>{fibo['anchor_point_2_label']}</b></div>
          <div class="fibo-grid">
            <div class="fibo-cell"><span>0.236</span><strong>${fibo['0.236']:.2f}</strong></div>
            <div class="fibo-cell"><span>0.382</span><strong>${fibo['0.382']:.2f}</strong></div>
            <div class="fibo-cell {'fibo-active' if plan['fibo_entry_level']=='0.500' else ''}"><span>0.500</span><strong>${fibo['0.500']:.2f}</strong></div>
            <div class="fibo-cell {'fibo-active' if plan['fibo_entry_level']=='0.618' else ''}"><span>0.618 (Golden)</span><strong>${fibo['0.618']:.2f}</strong></div>
            <div class="fibo-cell"><span>0.786</span><strong>${fibo['0.786']:.2f}</strong></div>
          </div>

          <div class="section-title">🎫 สัญญา Option แนะนำ (Real-Time Black-Scholes ไร้ดีเลย์ 15 นาที)</div>
          <div class="option-box">
            <div><b>Contract:</b> <code>{tk} {oq['expiration']} ${oq['strike']:.2f} {oq['option_type']}</code> ({oq['dte']} DTE | &Delta; {oq['delta']:+.2f})</div>
            <div><b>ราคาประเมินสด (Real-Time):</b> <span class="highlight-gold">~${oq['realtime_theo_price']:.2f}</span> (Bid ${oq['est_bid']:.2f} / Ask ${oq['est_ask']:.2f})</div>
            <div><b>ตั้ง Limit Buy ไม่เกิน:</b> <span class="highlight-cyan">${oq['max_limit_buy']:.2f}</span> ({oq['recommended_contracts']} สัญญา)</div>
            <div><b>รูปแบบแท่งเทียน:</b> {candle['pattern_name']}</div>
          </div>

          <div class="section-title">🎯 แผนทำกำไร &amp; ตัดขาดทุน (TP / SL อ้างอิงหุ้นแม่ Real-Time)</div>
          <div class="tpsl-list">
            <div class="tp1-row">✅ <b>TP 1 ({plan['tp1_label']}):</b> หุ้นแตะ <b>${plan['tp1_underlying']:.2f}</b> &rarr; Option <b>~${oq['tp1_option_price']:.2f} ({oq['tp1_gain_pct']:+.1f}%)</b> [ขยับ SL บังทุนที่ ${plan['lock_sl_after_tp1']:.2f}]</div>
            <div class="tp2-row">🚀 <b>TP 2 ({plan['tp2_label']}):</b> หุ้นแตะ <b>${plan['tp2_underlying']:.2f}</b> &rarr; Option <b>~${oq['tp2_option_price']:.2f} ({oq['tp2_gain_pct']:+.1f}%)</b></div>
            <div class="sl-row">🛑 <b>SL ({plan['sl_label']}):</b> คัทถ้าหุ้นทะลุ <b>${plan['sl_underlying']:.2f}</b> &rarr; Option <b>~${oq['sl_option_price']:.2f} ({oq['sl_loss_pct']:+.1f}%)</b></div>
          </div>
          <div class="earn-footer">📅 สถานะงบการเงิน: {item['earnings']['status_label']}</div>
        </div>
        """)

    bt_rows = []
    for tk, info in bt_data.get("per_ticker", {}).items():
        g = info.get("golden", {})
        u = info.get("upgraded", {})
        bt_rows.append(f"""
        <tr>
          <td><b>{tk}</b></td>
          <td>{info.get('days_tested', 60)} วัน ({info.get('bars_tested', 8190):,} แท่ง)</td>
          <td><b>{g.get('win_rate_pct', 0.0):.1f}%</b> ({g.get('wins', 0)}W / {g.get('losses', 0)}L)</td>
          <td class="{'pos' if g.get('net_pnl', 0) >= 0 else 'neg'}">${g.get('net_pnl', 0.0):+,.2f}</td>
          <td>{g.get('profit_factor', 0.0):.2f}</td>
          <td>{u.get('win_rate_pct', 0.0):.1f}% (${u.get('net_pnl', 0.0):+,.2f})</td>
        </tr>
        """)

    html = f"""<!DOCTYPE html>
<html lang="th">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Day Option Bot — Real-Time EMA 200 &amp; Fibonacci 0.500/0.618 Dashboard</title>
  <style>
    :root {{
      --bg: #0b0f19;
      --card: #131b2e;
      --border: #23304d;
      --text: #e2e8f0;
      --muted: #94a3b8;
      --green: #10b981;
      --red: #ef4444;
      --gold: #f59e0b;
      --cyan: #06b6d4;
    }}
    body {{
      margin: 0;
      font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
      background: var(--bg);
      color: var(--text);
      padding: 24px;
    }}
    .container {{ max-width: 1360px; margin: 0 auto; }}
    .header {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      border-bottom: 1px solid var(--border);
      padding-bottom: 16px;
      margin-bottom: 24px;
    }}
    .kpi-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 16px;
      margin-bottom: 28px;
    }}
    .kpi-card {{
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 16px;
    }}
    .kpi-label {{ color: var(--muted); font-size: 13px; }}
    .kpi-val {{ font-size: 26px; font-weight: 700; margin-top: 6px; }}
    .cards-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(580px, 1fr));
      gap: 20px;
      margin-bottom: 32px;
    }}
    .card {{
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 20px;
    }}
    .card-header {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 12px;
    }}
    .ticker-title {{ font-size: 24px; font-weight: 800; margin-right: 10px; }}
    .badge {{
      display: inline-block;
      padding: 4px 10px;
      border-radius: 999px;
      font-size: 12px;
      font-weight: 700;
      margin-right: 6px;
    }}
    .badge-put {{ background: rgba(239,68,68,0.2); color: #fca5a5; border: 1px solid #ef4444; }}
    .badge-call {{ background: rgba(16,185,129,0.2); color: #6ee7b7; border: 1px solid #10b981; }}
    .badge-actionable {{ background: var(--gold); color: #000; }}
    .badge-near {{ background: rgba(6,182,212,0.25); color: #67e8f9; border: 1px solid var(--cyan); }}
    .badge-watch {{ background: rgba(148,163,184,0.15); color: var(--muted); }}
    .badge-blackout {{ background: rgba(239,68,68,0.35); color: #fecaca; }}
    .spot-price {{ font-size: 22px; font-weight: 700; text-align: right; }}
    .sub-meta {{ font-size: 12px; color: var(--muted); text-align: right; }}
    .status-banner {{
      background: rgba(255,255,255,0.04);
      padding: 10px 12px;
      border-radius: 8px;
      font-size: 14px;
      margin-bottom: 14px;
    }}
    .section-title {{
      font-size: 13px;
      font-weight: 700;
      color: var(--cyan);
      margin: 12px 0 6px;
      text-transform: uppercase;
    }}
    .anchor-info {{ font-size: 13px; color: var(--muted); margin-bottom: 8px; }}
    .fibo-grid {{
      display: grid;
      grid-template-columns: repeat(5, 1fr);
      gap: 8px;
      margin-bottom: 12px;
    }}
    .fibo-cell {{
      background: rgba(255,255,255,0.03);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 8px;
      text-align: center;
      font-size: 12px;
    }}
    .fibo-cell span {{ display: block; color: var(--muted); margin-bottom: 4px; }}
    .fibo-active {{
      border: 1.5px solid var(--gold);
      background: rgba(245,158,11,0.12);
    }}
    .option-box {{
      background: rgba(6,182,212,0.07);
      border: 1px solid rgba(6,182,212,0.25);
      border-radius: 10px;
      padding: 12px;
      font-size: 13.5px;
      line-height: 1.65;
    }}
    .highlight-gold {{ color: var(--gold); font-weight: 700; }}
    .highlight-cyan {{ color: var(--cyan); font-weight: 700; }}
    .tpsl-list {{ font-size: 13px; line-height: 1.75; margin-top: 6px; }}
    .tp1-row {{ color: #6ee7b7; }}
    .tp2-row {{ color: #34d399; }}
    .sl-row {{ color: #fca5a5; }}
    .earn-footer {{
      margin-top: 12px;
      padding-top: 8px;
      border-top: 1px dashed var(--border);
      font-size: 12px;
      color: var(--muted);
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      background: var(--card);
      border-radius: 12px;
      overflow: hidden;
    }}
    th, td {{
      padding: 12px 14px;
      border-bottom: 1px solid var(--border);
      text-align: left;
      font-size: 14px;
    }}
    th {{ background: rgba(255,255,255,0.04); color: var(--cyan); }}
    .pos {{ color: var(--green); font-weight: 700; }}
    .neg {{ color: var(--red); font-weight: 700; }}
  </style>
</head>
<body>
  <div class="container">
    <div class="header">
      <div>
        <h1 style="margin:0;">⚡ Day Option Bot — Real-Time EMA 200 &amp; Fibonacci 0.500 / 0.618</h1>
        <div style="color:var(--muted);font-size:14px;margin-top:4px;">
          กรองเทรนด์ EMA 200 (TF 2m) + ตี Fibo วันก่อนหน้าอัตโนมัติ + คำนวณราคา Option สดด้วย Black-Scholes (ไร้ดีเลย์ 15 นาที)
        </div>
      </div>
      <div style="text-align:right;font-size:13px;color:var(--muted);">
        อัปเดตล่าสุด: <b>{scan_output.get('timestamp')}</b>
      </div>
    </div>

    <div class="kpi-grid">
      <div class="kpi-card">
        <div class="kpi-label">สัญญาณพร้อมเข้าเทรด (ACTIONABLE / NEAR)</div>
        <div class="kpi-val" style="color:var(--gold);">{scan_output.get('actionable_count', 0)} / {scan_output.get('near_setup_count', 0)}</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">อัตราชนะรวม (Paper Trade Win Rate)</div>
        <div class="kpi-val" style="color:var(--green);">{kpis.get('win_rate_pct', 0.0):.1f}% ({kpis.get('wins', 0)}W/{kpis.get('losses', 0)}L)</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">Win Rate ที่เส้นทองคำ Fibo 0.618</div>
        <div class="kpi-val" style="color:var(--cyan);">{kpis.get('fibo_0618', {}).get('win_rate_pct', 0.0):.1f}%</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">กำไรสุทธิสะสม (Paper Net PnL)</div>
        <div class="kpi-val">${kpis.get('net_pnl_usd', 0.0):+,.2f}</div>
      </div>
    </div>

    <div class="cards-grid">
      {"".join(cards_html)}
    </div>

    <h2 style="margin-bottom:12px;">🏆 สรุปผล Backtest ย้อนหลัง 90 วัน (60 วันทำการ / 81,900 แท่งเทียน)</h2>
    <table>
      <thead>
        <tr>
          <th>หุ้น (Ticker)</th>
          <th>จำนวนวันที่ทดสอบ</th>
          <th>Win Rate (Golden Mode)</th>
          <th>กำไรสุทธิ (Net PnL)</th>
          <th>Profit Factor</th>
          <th>ผลลัพธ์เมื่อเทรดทั้ง 0.500 &amp; 0.618</th>
        </tr>
      </thead>
      <tbody>
        {"".join(bt_rows) if bt_rows else '<tr><td colspan="6">รันคำสั่ง --backtest เพื่อสร้างตารางสรุป 90 วัน</td></tr>'}
      </tbody>
    </table>
  </div>
</body>
</html>
"""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    return out_path
