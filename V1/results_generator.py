"""
results_generator.py
───────────────────
Görsel Raporlama Katmanı - nasdaq_bot_v2

Simülasyon sonuçlarını interaktif grafikler, KPI kartları ve
detaylı işlem tablosu barındıran modern bir Dark Mode HTML raporuna dönüştürür.
"""

import json
import os
from datetime import datetime
from pathlib import Path

# HTML Rapor Tasarımı ve Şablonu
HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="tr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title} - NASDAQ BOT V2 Performans Raporu</title>
    <!-- Google Fonts -->
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
    <!-- Chart.js CDN -->
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        :root {{
            --bg-color: #0b0f19;
            --card-bg: #151d30;
            --border-color: #23324c;
            --text-primary: #e2e8f0;
            --text-secondary: #94a3b8;
            --accent-blue: #38bdf8;
            --success-color: #10b981;
            --danger-color: #ef4444;
            --warning-color: #f59e0b;
            --card-hover: #1e293b;
        }}

        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
            font-family: 'Outfit', sans-serif;
        }}

        body {{
            background-color: var(--bg-color);
            color: var(--text-primary);
            padding: 2rem;
            line-height: 1.5;
        }}

        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}

        /* Header */
        header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 2rem;
            border-bottom: 1px solid var(--border-color);
            padding-bottom: 1.5rem;
        }}

        .logo-area {{
            display: flex;
            align-items: center;
            gap: 12px;
        }}

        .logo-badge {{
            background: linear-gradient(135deg, var(--accent-blue), #0284c7);
            color: #fff;
            padding: 6px 12px;
            border-radius: 8px;
            font-weight: 700;
            font-size: 0.9rem;
            letter-spacing: 1px;
        }}

        h1 {{
            font-size: 1.8rem;
            font-weight: 700;
            background: linear-gradient(to right, #fff, var(--text-secondary));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}

        .timestamp {{
            font-size: 0.9rem;
            color: var(--text-secondary);
        }}

        /* KPI Cards Grid */
        .kpi-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
            gap: 1.5rem;
            margin-bottom: 2rem;
        }}

        .kpi-card {{
            background-color: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 1.5rem;
            transition: transform 0.2s, box-shadow 0.2s;
        }}

        .kpi-card:hover {{
            transform: translateY(-4px);
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.3);
            background-color: var(--card-hover);
        }}

        .kpi-title {{
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: var(--text-secondary);
            margin-bottom: 0.5rem;
            font-weight: 600;
        }}

        .kpi-value {{
            font-size: 1.8rem;
            font-weight: 700;
        }}

        .kpi-subtitle {{
            font-size: 0.8rem;
            margin-top: 0.25rem;
            font-weight: 500;
        }}

        .text-success {{ color: var(--success-color) !important; }}
        .text-danger {{ color: var(--danger-color) !important; }}
        .text-warning {{ color: var(--warning-color) !important; }}
        .text-blue {{ color: var(--accent-blue) !important; }}

        /* Charts Grid */
        .charts-grid {{
            display: grid;
            grid-template-columns: 2fr 1fr;
            gap: 1.5rem;
            margin-bottom: 2rem;
        }}

        @media (max-width: 1024px) {{
            .charts-grid {{
                grid-template-columns: 1fr;
            }}
        }}

        .chart-card {{
            background-color: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 1.5rem;
            min-height: 400px;
        }}

        .chart-title {{
            font-size: 1.1rem;
            font-weight: 600;
            margin-bottom: 1rem;
            border-left: 4px solid var(--accent-blue);
            padding-left: 10px;
        }}

        .chart-container {{
            position: relative;
            width: 100%;
            height: calc(100% - 2rem);
            min-height: 320px;
        }}

        /* Table Card */
        .table-card {{
            background-color: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 1.5rem;
            overflow: hidden;
        }}

        .table-header-area {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 1.2rem;
        }}

        .table-title {{
            font-size: 1.1rem;
            font-weight: 600;
            border-left: 4px solid var(--accent-blue);
            padding-left: 10px;
        }}

        .table-wrapper {{
            overflow-x: auto;
            max-height: 500px;
            overflow-y: auto;
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
            text-align: left;
            font-size: 0.9rem;
        }}

        th {{
            background-color: rgba(35, 50, 76, 0.4);
            color: var(--text-secondary);
            font-weight: 600;
            padding: 12px 16px;
            border-bottom: 2px solid var(--border-color);
            position: sticky;
            top: 0;
            z-index: 10;
        }}

        td {{
            padding: 12px 16px;
            border-bottom: 1px solid var(--border-color);
            color: var(--text-primary);
        }}

        tr:hover td {{
            background-color: rgba(255, 255, 255, 0.02);
        }}

        /* Badges */
        .badge {{
            display: inline-block;
            padding: 4px 8px;
            border-radius: 6px;
            font-size: 0.75rem;
            font-weight: 600;
            text-transform: uppercase;
        }}

        .badge-long {{
            background-color: rgba(16, 185, 129, 0.15);
            color: var(--success-color);
            border: 1px solid rgba(16, 185, 129, 0.3);
        }}

        .badge-short {{
            background-color: rgba(239, 68, 68, 0.15);
            color: var(--danger-color);
            border: 1px solid rgba(239, 68, 68, 0.3);
        }}

        .badge-tp {{
            background-color: rgba(16, 185, 129, 0.15);
            color: var(--success-color);
        }}

        .badge-sl {{
            background-color: rgba(239, 68, 68, 0.15);
            color: var(--danger-color);
        }}

        .badge-force {{
            background-color: rgba(148, 163, 184, 0.15);
            color: var(--text-secondary);
        }}

        .badge-guard {{
            background-color: rgba(245, 158, 11, 0.15);
            color: var(--warning-color);
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div class="logo-area">
                <span class="logo-badge">CRO ROOM</span>
                <div>
                    <h1>NASDAQ BOT V2</h1>
                    <div style="font-size: 0.85rem; color: var(--text-secondary);">{title}</div>
                </div>
            </div>
            <div class="timestamp">
                Rapor Zamanı: {timestamp}
            </div>
        </header>

        <!-- KPI Cards -->
        <div class="kpi-grid">
            <!-- Net PnL -->
            <div class="kpi-card">
                <div class="kpi-title">Net PnL</div>
                <div class="kpi-value {pnl_class}">{net_pnl}</div>
                <div class="kpi-subtitle {pnl_class}">{net_pnl_pct}</div>
            </div>
            <!-- Win Rate -->
            <div class="kpi-card">
                <div class="kpi-title">Win Rate</div>
                <div class="kpi-value text-blue">{win_rate}%</div>
                <div class="kpi-subtitle text-secondary">{winning_trades} Kazanç / {losing_trades} Kayıp</div>
            </div>
            <!-- Max DD -->
            <div class="kpi-card">
                <div class="kpi-title">Maks Drawdown</div>
                <div class="kpi-value text-danger">{max_dd_pct}%</div>
                <div class="kpi-subtitle text-secondary">USD: {max_dd_usd}</div>
            </div>
            <!-- Sharpe -->
            <div class="kpi-card">
                <div class="kpi-title">Sharpe Oranı</div>
                <div class="kpi-value text-warning">{sharpe}</div>
                <div class="kpi-subtitle text-secondary">Risk-Free Rate: 0%</div>
            </div>
            <!-- Total Trades -->
            <div class="kpi-card">
                <div class="kpi-title">Toplam İşlem</div>
                <div class="kpi-value text-primary">{total_trades}</div>
                <div class="kpi-subtitle text-secondary">Profit Factor: {profit_factor}</div>
            </div>
        </div>

        <!-- Charts -->
        <div class="charts-grid">
            <!-- Equity Curve -->
            <div class="chart-card">
                <div class="chart-title">Kazanç Eğrisi (Equity Curve)</div>
                <div class="chart-container">
                    <canvas id="equityChart"></canvas>
                </div>
            </div>
            <!-- Trade Distribution -->
            <div class="chart-card">
                <div class="chart-title">İşlem Dağılımı</div>
                <div class="chart-container">
                    <canvas id="distChart"></canvas>
                </div>
            </div>
        </div>

        <!-- Trade Log Table -->
        <div class="table-card">
            <div class="table-header-area">
                <div class="table-title">İşlem Günlüğü (Trade Log)</div>
            </div>
            <div class="table-wrapper">
                <table>
                    <thead>
                        <tr>
                            <th>#</th>
                            <th>Açılış Zamanı</th>
                            <th>Kapanış Zamanı</th>
                            <th>Yön</th>
                            <th>Lot</th>
                            <th>Giriş Fiyatı</th>
                            <th>Çıkış Fiyatı</th>
                            <th>Net PnL</th>
                            <th>Neden</th>
                        </tr>
                    </thead>
                    <tbody>
                        {table_rows}
                    </tbody>
                </table>
            </div>
        </div>
    </div>

    <!-- Script to Initialize Charts -->
    <script>
        // Equity Curve Data
        const equityData = {equity_data_json};
        const equityLabels = {equity_labels_json};

        // Distribution Data
        const winningTrades = {winning_trades};
        const losingTrades = {losing_trades};

        // Initialize Equity Curve Chart
        const ctxEquity = document.getElementById('equityChart').getContext('2d');
        const equityGradient = ctxEquity.createLinearGradient(0, 0, 0, 400);
        equityGradient.addColorStop(0, 'rgba(56, 189, 248, 0.3)');
        equityGradient.addColorStop(1, 'rgba(56, 189, 248, 0.0)');

        new Chart(ctxEquity, {{
            type: 'line',
            data: {{
                labels: equityLabels,
                datasets: [{{
                    label: 'Portföy Değeri (USD)',
                    data: equityData,
                    borderColor: '#38bdf8',
                    borderWidth: 2,
                    fill: true,
                    backgroundColor: equityGradient,
                    tension: 0.1,
                    pointRadius: equityLabels.length > 500 ? 0 : 2,
                    pointHoverRadius: 6
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    legend: {{
                        display: false
                    }},
                    tooltip: {{
                        mode: 'index',
                        intersect: false,
                        backgroundColor: '#151d30',
                        titleColor: '#e2e8f0',
                        bodyColor: '#e2e8f0',
                        borderColor: '#23324c',
                        borderWidth: 1
                    }}
                }},
                scales: {{
                    x: {{
                        grid: {{
                            color: 'rgba(35, 50, 76, 0.2)'
                        }},
                        ticks: {{
                            color: '#94a3b8',
                            maxTicksLimit: 12
                        }}
                    }},
                    y: {{
                        grid: {{
                            color: 'rgba(35, 50, 76, 0.2)'
                        }},
                        ticks: {{
                            color: '#94a3b8',
                            callback: function(value) {{
                                return '$' + value.toLocaleString();
                            }}
                        }}
                    }}
                }}
            }}
        }});

        // Initialize Distribution Chart (Pie)
        const ctxDist = document.getElementById('distChart').getContext('2d');
        new Chart(ctxDist, {{
            type: 'doughnut',
            data: {{
                labels: ['Kazanan', 'Kaybeden'],
                datasets: [{{
                    data: [winningTrades, losingTrades],
                    backgroundColor: ['#10b981', '#ef4444'],
                    borderColor: '#151d30',
                    borderWidth: 2
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    legend: {{
                        position: 'bottom',
                        labels: {{
                            color: '#e2e8f0',
                            font: {{
                                family: 'Outfit'
                            }}
                        }}
                    }},
                    tooltip: {{
                        backgroundColor: '#151d30',
                        titleColor: '#e2e8f0',
                        bodyColor: '#e2e8f0',
                        borderColor: '#23324c',
                        borderWidth: 1
                    }}
                }},
                cutout: '70%'
            }}
        }});
    </script>
</body>
</html>
"""

def generate_html_report(portfolio, initial_balance: float, scenario_name: str = None) -> Path:
    """
    PortfolioManager verilerinden interaktif bir HTML raporu oluşturur.
    """
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)

    # 1. Dosya adının belirlenmesi
    if scenario_name:
        # Stres testi odasından çağrıldıysa
        filename = f"stress_{scenario_name}.html"
        title = f"Stres Testi Senaryosu: {scenario_name}"
    else:
        # Normal backtest
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"backtest_res_{timestamp_str}.html"
        title = "Normal Backtest Koşumu"

    report_path = results_dir / filename

    # 2. Metriklerin alınması
    summary = portfolio.summary()
    trade_log = portfolio.trade_log
    equity_curve = getattr(portfolio, "equity_curve", [initial_balance])
    equity_timestamps = getattr(portfolio, "equity_timestamps", [str(i) for i in range(len(equity_curve))])

    total_trades = summary["total_trades"]
    winning_trades = summary["winning_trades"]
    losing_trades = total_trades - winning_trades
    win_rate = summary["win_rate_pct"]

    net_pnl = summary["total_pnl"]
    net_pnl_pct = summary["total_pnl_pct"]

    pnl_class = "text-success" if net_pnl >= 0 else "text-danger"
    net_pnl_str = f"${net_pnl:,.2f}"
    if net_pnl > 0:
        net_pnl_str = "+" + net_pnl_str
    net_pnl_pct_str = f"{net_pnl_pct:+.2f}%"

    # Sharpe ve MDD hesapla
    # Sharpe
    import numpy as np
    import math
    if len(equity_curve) >= 2:
        returns = np.diff(equity_curve) / np.array(equity_curve[:-1])
        std = np.std(returns, ddof=1)
        # Yıllıklandırma faktörü
        sharpe = float(np.mean(returns) / std * math.sqrt(17472)) if std > 1e-10 else 0.0
    else:
        sharpe = 0.0

    # Max Drawdown
    arr = np.array(equity_curve)
    peak = np.maximum.accumulate(arr)
    dd = peak - arr
    max_dd_usd = float(np.max(dd))
    max_dd_pct = float(np.max(dd / np.where(peak == 0, 1, peak)) * 100)

    # Profit Factor
    gross_profit = sum(t["net_pnl"] for t in trade_log if t["net_pnl"] > 0)
    gross_loss = abs(sum(t["net_pnl"] for t in trade_log if t["net_pnl"] < 0))
    profit_factor = round(gross_profit / gross_loss, 3) if gross_loss > 0 else (float("inf") if gross_profit > 0 else 0.0)

    # 3. Tablo Satırlarının Oluşturulması
    table_rows = []
    for idx, t in enumerate(trade_log, 1):
        direction_badge = f'<span class="badge badge-long">LONG</span>' if t["direction"] == "STRONG_LONG" else f'<span class="badge badge-short">SHORT</span>'
        
        reason = t["reason"]
        if reason == "TP":
            reason_badge = f'<span class="badge badge-tp">TP</span>'
        elif reason == "TRAIL":
            reason_badge = f'<span class="badge badge-tp">Trailing</span>'
        elif reason == "SL":
            reason_badge = f'<span class="badge badge-sl">SL</span>'
        elif reason == "FORCE_CLOSE":
            reason_badge = f'<span class="badge badge-force">Force Close</span>'
        elif reason == "GUARDRAIL":
            reason_badge = f'<span class="badge badge-guard">Guardrail</span>'
        else:
            reason_badge = f'<span class="badge badge-force">{reason}</span>'

        pnl = t["net_pnl"]
        pnl_td_class = "text-success" if pnl >= 0 else "text-danger"
        pnl_str = f"${pnl:,.2f}"
        if pnl > 0:
            pnl_str = "+" + pnl_str

        # Format open/close times safely
        open_time = t["open_time"]
        if hasattr(open_time, "strftime"):
            open_time_str = open_time.strftime("%Y-%m-%d %H:%M:%S")
        else:
            open_time_str = str(open_time)

        close_time = t["close_time"]
        if hasattr(close_time, "strftime"):
            close_time_str = close_time.strftime("%Y-%m-%d %H:%M:%S")
        else:
            close_time_str = str(close_time)

        row = f"""
        <tr>
            <td>{idx}</td>
            <td>{open_time_str}</td>
            <td>{close_time_str}</td>
            <td>{direction_badge}</td>
            <td>{t["lot_size"]:.4f}</td>
            <td>${t["entry_price"]:.2f}</td>
            <td>${t["exit_price"]:.2f}</td>
            <td class="{pnl_td_class}">{pnl_str}</td>
            <td>{reason_badge}</td>
        </tr>
        """
        table_rows.append(row)

    table_rows_html = "\n".join(table_rows)

    # 4. JSON verilerinin hazırlanması
    equity_data_json = json.dumps(equity_curve)
    equity_labels_json = json.dumps(equity_timestamps)

    # HTML Şablonunu Doldur
    html_content = HTML_TEMPLATE.format(
        title=title,
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        net_pnl=net_pnl_str,
        pnl_class=pnl_class,
        net_pnl_pct=net_pnl_pct_str,
        win_rate=f"{win_rate:.1f}",
        winning_trades=winning_trades,
        losing_trades=losing_trades,
        max_dd_pct=f"{max_dd_pct:.2f}",
        max_dd_usd=f"${max_dd_usd:,.2f}",
        sharpe=f"{sharpe:.4f}",
        total_trades=total_trades,
        profit_factor=str(profit_factor),
        table_rows=table_rows_html,
        equity_data_json=equity_data_json,
        equity_labels_json=equity_labels_json
    )

    # Yaz
    report_path.write_text(html_content, encoding="utf-8")
    print(f"[RAPOR-MOTORU] HTML Performans Raporu oluşturuldu → {report_path.resolve()}")
    return report_path
