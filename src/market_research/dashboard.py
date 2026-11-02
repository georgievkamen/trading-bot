from __future__ import annotations

import json
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from .paper import PAPER_STRATEGY_NAMES, paper_summary
from .equities import PAPER_ACCOUNT_ID as STOCK_PAPER_ACCOUNT_ID
from .storage import MarketStore


PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Market Research · Paper Lab</title>
<style>
:root{color-scheme:dark;--bg:#090d12;--panel:#111821;--panel2:#151f2a;--line:#23303d;--text:#ecf1f7;--muted:#8d9aaa;--green:#43d6a0;--red:#ff7383;--gold:#e8c476;--blue:#8da8ff;--cyan:#73d6e3}
*{box-sizing:border-box}body{margin:0;background:radial-gradient(ellipse at 70% -20%,#1a2a36 0%,var(--bg) 48%);color:var(--text);font:14px/1.5 Segoe UI,Inter,system-ui,sans-serif;min-height:100vh}
.wrap{max-width:1440px;margin:auto;padding:32px 28px 60px}.top{display:flex;justify-content:space-between;align-items:center;margin-bottom:30px}.brand{display:flex;gap:14px;align-items:center}.logo{width:42px;height:42px;border-radius:13px;background:linear-gradient(145deg,#67e2b1,#1f8f73);display:grid;place-items:center;color:#06130e;font-weight:900;font-size:19px;box-shadow:0 6px 26px #2fc39435}.eyebrow{font-size:11px;letter-spacing:.16em;text-transform:uppercase;color:var(--green);font-weight:700}.title{font-size:23px;font-weight:700;letter-spacing:-.03em}.sub{color:var(--muted);font-size:12px;margin-top:2px}.status{display:flex;align-items:center;gap:9px;color:#c5d0db;background:#111a23;border:1px solid var(--line);padding:9px 13px;border-radius:999px;font-size:12px}.dot{height:8px;width:8px;border-radius:50%;background:var(--green);box-shadow:0 0 14px #43d6a080}.dot.warn{background:var(--gold);box-shadow:0 0 14px #e8c47680}
.grid{display:grid;gap:14px}.metrics{grid-template-columns:repeat(4,minmax(0,1fr));margin-bottom:16px}.card{background:linear-gradient(145deg,#131c26eF,#101720ef);border:1px solid var(--line);border-radius:16px;padding:18px;box-shadow:0 14px 40px #0002}.metric .label{color:var(--muted);font-size:12px}.metric .value{font-size:27px;font-weight:700;letter-spacing:-.04em;margin-top:8px;font-variant-numeric:tabular-nums}.metric .hint{color:var(--muted);font-size:11px;margin-top:4px}.pos{color:var(--green)!important}.neg{color:var(--red)!important}
.section-head{display:flex;justify-content:space-between;align-items:end;margin:26px 2px 12px}.section-head h2{font-size:16px;margin:0;letter-spacing:-.02em}.section-head span{font-size:11px;color:var(--muted)}.middle{grid-template-columns:minmax(0,1.7fr) minmax(280px,1fr)}.chart-card{padding:20px 20px 14px}.chart-head{display:flex;justify-content:space-between;align-items:start}.chart-head h2{font-size:15px;margin:0}.legend{display:flex;gap:13px;flex-wrap:wrap;font-size:11px;color:var(--muted)}.legend i{display:inline-block;width:7px;height:7px;border-radius:50%;margin-right:5px}.chart{width:100%;height:255px;margin-top:12px}.chart-note{font-size:11px;color:var(--muted)}
.accounts{grid-template-columns:repeat(auto-fit,minmax(280px,1fr))}.account{padding:18px}.account-top{display:flex;justify-content:space-between;align-items:center}.strategy{font-size:14px;font-weight:700}.tag{border:1px solid var(--line);padding:3px 8px;border-radius:999px;font-size:10px;color:var(--muted)}.account-value{font-size:24px;font-weight:700;letter-spacing:-.04em;margin-top:16px}.account-row{display:flex;justify-content:space-between;color:var(--muted);font-size:12px;margin-top:7px}.account-row strong{color:var(--text);font-weight:600}.mini-bar{height:4px;background:#202a35;border-radius:9px;margin-top:15px;overflow:hidden}.mini-bar b{display:block;height:100%;background:var(--green);border-radius:9px}.regime{border-top:1px solid var(--line);margin-top:15px;padding-top:12px}.regime-line{display:flex;justify-content:space-between;font-size:12px}.regime-line span{color:var(--muted)}.regime p{font-size:11px;color:#a9b5c3;margin:7px 0 0;line-height:1.45}
.lower{grid-template-columns:1fr 1fr}.table-wrap{overflow:auto}.table{width:100%;border-collapse:collapse;font-size:12px}.table th{text-align:left;color:var(--muted);font-weight:500;padding:10px 8px;border-bottom:1px solid var(--line);white-space:nowrap}.table td{padding:11px 8px;border-bottom:1px solid #1d2732;white-space:nowrap}.empty{color:var(--muted);text-align:center;padding:24px 8px!important}.footer{color:#667585;font-size:11px;text-align:center;margin-top:28px}.refresh{color:var(--muted);font-size:11px}a{color:var(--cyan)}
@media(max-width:900px){.metrics{grid-template-columns:repeat(2,minmax(0,1fr))}.middle,.lower{grid-template-columns:1fr}.accounts{grid-template-columns:1fr}.wrap{padding:22px 15px 42px}.top{align-items:flex-start;gap:12px}.status{font-size:0}.status .dot{font-size:12px}}
</style>
</head>
<body><main class="wrap">
<header class="top"><div class="brand"><div class="logo">P</div><div><div class="eyebrow">Local simulation</div><div class="title">Market Research <span style="color:var(--muted);font-weight:400">/ Paper Lab</span></div><div class="sub">Independent simulated experiments · no live orders</div></div></div><div class="status"><i class="dot" id="healthDot"></i><span id="healthText">Connecting to local paper accounts…</span></div></header>
<section class="grid metrics" id="metrics"></section>
<section class="grid middle"><div class="card chart-card"><div class="chart-head"><div><h2>Equity over time</h2><div class="sub">Marked account value, including open positions</div></div><div class="legend" id="legend"></div></div><canvas id="equityChart" class="chart"></canvas><div class="chart-note">Updates every 15 seconds · values are simulated</div></div><div class="card"><div class="section-head" style="margin:0 0 8px"><h2>Universe</h2><span id="universeCount"></span></div><div class="sub" style="line-height:1.8">Crypto Llama classifies 15-minute regimes across liquid coins. Position limits remain fixed.</div><div id="aiNotes" style="margin-top:15px"></div></div></section>
<div class="section-head"><h2>Paper accounts</h2><span>Independent strategy trials · $1,000 starting cash each</span></div><section class="grid accounts" id="accounts"></section>
<section class="grid lower"><div><div class="section-head"><h2>Open positions</h2><span>All experiments</span></div><div class="card table-wrap"><table class="table"><thead><tr><th>Strategy</th><th>Market</th><th>Mark</th><th>Unrealized P&amp;L</th></tr></thead><tbody id="positions"></tbody></table></div></div><div><div class="section-head"><h2>Recent closed trades</h2><span>Newest first</span></div><div class="card table-wrap"><table class="table"><thead><tr><th>Strategy</th><th>Market</th><th>Exit</th><th>Net P&amp;L</th></tr></thead><tbody id="trades"></tbody></table></div></div></section>
<div class="footer">Local dashboard · public market data · paper execution only · <span id="updated">Waiting for data</span></div>
</main>
<script>
const colors={trend:'#43d6a0',mean_reversion:'#e8c476',mean_reversion_filtered:'#73d6e3',llama_regime:'#8da8ff',stocks_llama_regime:'#d297ff'};
const labels={trend:'Trend baseline',mean_reversion:'Mean-reversion baseline',mean_reversion_filtered:'Trend-filtered mean reversion (experiment)',llama_regime:'Crypto · Llama regime strategy',stocks_llama_regime:'US equities · Llama daily strategy'};
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const money=n=>new Intl.NumberFormat('en-US',{style:'currency',currency:'USD',maximumFractionDigits:2}).format(Number(n||0));
const pct=n=>`${Number(n||0)>=0?'+':''}${Number(n||0).toFixed(2)}%`;
function accountCard(a){const color=colors[a.strategy]||'#43d6a0';const pnl=Number(a.pnl||0);const exposure=Math.min(100,Math.max(0,(Number(a.starting_cash||1000)-Number(a.cash||0))/Number(a.starting_cash||1000)*100));const marketStatus=a.market_status||{};const fetchCount=marketStatus.requested_markets?`${Number(a.successful_market_count||0)} succeeded · ${Number(a.failed_market_count||0)} failed`:'awaiting first poll';const modeLabel=a.last_processing_mode==='historical_replay'?'Historical replay':a.last_processing_mode==='live_forward'?'Live forward':'not yet run';const gap=a.last_downtime_gap;const gapLabel=gap?`${Number(gap.estimated_missed_intervals||0)} interval(s) skipped`:'none recorded';let ai='';if(a.strategy==='llama_regime'){const notes=Object.entries(a.model_notes||{});const learning=a.learning_stats||{};ai='<div class="regime"><div class="regime-line"><strong>Local model</strong><span>'+esc(a.model||'Llama')+'</span></div><div class="regime-line" style="margin-top:6px"><span>Latest regime notes</span><strong>'+notes.length+' markets</strong></div><div class="regime-line" style="margin-top:6px"><span>Resolved coin outcomes</span><strong>'+Number(learning.resolved_samples||0)+'</strong></div><div class="regime-line" style="margin-top:6px"><span>Independent 4h windows</span><strong>'+Number(learning.independent_windows||0)+'</strong></div><div class="regime-line" style="margin-top:6px"><span>Feedback state</span><strong>'+(learning.ready?'feedback available':'collecting data')+'</strong></div></div>'}return `<article class="card account"><div class="account-top"><div class="strategy">${esc(labels[a.strategy]||a.strategy)}</div><span class="tag">${a.online?(Number(a.failed_market_count||0)?'● ACTIVE · FEED ISSUES':'● RUNNING'):'○ STALE'}</span></div><div class="account-value">${money(a.equity)}</div><div class="account-row"><span>Net P&amp;L</span><strong class="${pnl>=0?'pos':'neg'}">${money(pnl)} · ${pct(a.return_pct)}</strong></div><div class="account-row"><span>Cash</span><strong>${money(a.cash)}</strong></div><div class="account-row"><span>Realized</span><strong>${money(a.realized_pnl)}</strong></div><div class="account-row"><span>Closed trades / open</span><strong>${a.closed_trades} / ${(a.open_positions||[]).length}</strong></div><div class="account-row"><span>Market requests</span><strong>${fetchCount}</strong></div><div class="account-row"><span>Excluded for snapshot skew</span><strong>${Number(a.snapshot_excluded_market_count||0)}</strong></div><div class="account-row"><span>Processing mode</span><strong>${modeLabel}</strong></div><div class="account-row"><span>Last downtime gap</span><strong>${gapLabel}</strong></div><div class="mini-bar"><b style="width:${exposure}%;background:${color}"></b></div>${ai}</article>`}
function drawChart(accounts){const c=document.getElementById('equityChart'),r=c.getBoundingClientRect(),d=window.devicePixelRatio||1;c.width=Math.max(320,r.width*d);c.height=255*d;const x=c.getContext('2d');x.scale(d,d);const w=r.width,h=255,p={l:54,r:12,t:14,b:24};x.clearRect(0,0,w,h);const sets=accounts.map(a=>({a,pts:a.equity_curve||[]})).filter(s=>s.pts.length>1);if(!sets.length){x.fillStyle='#8997a6';x.font='13px Segoe UI';x.fillText('The chart will draw after the first closed candle is processed.',p.l,100);return}const vals=sets.flatMap(s=>s.pts.map(q=>q.equity));let lo=Math.min(...vals),hi=Math.max(...vals);if(hi-lo<1){hi+=1;lo-=1}const y=v=>p.t+(hi-v)/(hi-lo)*(h-p.t-p.b);x.strokeStyle='#273442';x.lineWidth=1;x.fillStyle='#8291a1';x.font='10px Segoe UI';for(let i=0;i<4;i++){const v=lo+(hi-lo)*i/3,yy=y(v);x.beginPath();x.moveTo(p.l,yy);x.lineTo(w-p.r,yy);x.stroke();x.fillText(money(v),4,yy+3)}for(const s of sets){const pts=s.pts;x.strokeStyle=colors[s.a.strategy];x.lineWidth=2;x.beginPath();pts.forEach((q,i)=>{const xx=p.l+(w-p.l-p.r)*(i/Math.max(1,pts.length-1));i?x.lineTo(xx,y(q.equity)):x.moveTo(xx,y(q.equity))});x.stroke()}document.getElementById('legend').innerHTML=sets.map(s=>`<span><i style="background:${colors[s.a.strategy]}"></i>${esc(labels[s.a.strategy])}</span>`).join('')}
function render(data){const ac=data.accounts||[];const totals=ac.reduce((t,a)=>({start:t.start+Number(a.starting_cash||0),equity:t.equity+Number(a.equity||0),pnl:t.pnl+Number(a.pnl||0),realized:t.realized+Number(a.realized_pnl||0),online:t.online+(a.online?1:0),success:t.success+Number(a.successful_market_count||0),failed:t.failed+Number(a.failed_market_count||0)}),{start:0,equity:0,pnl:0,realized:0,online:0,success:0,failed:0});document.getElementById('metrics').innerHTML=[['Virtual capital',money(totals.start),'Across separate experiments'],['Combined marked equity',money(totals.equity),'Separate experiments, not one account'],['Combined net P&L',money(totals.pnl),`${pct(totals.start?totals.pnl/totals.start*100:0)} overall`],['Feed runners',`${totals.online} / ${ac.length}`,`${totals.success} market fetches succeeded · ${totals.failed} failed · ${(data.intervals||[60]).map(i=>i===1440?'1d':i+'m').join(', ')} bars`]].map((x,i)=>`<article class="card metric"><div class="label">${x[0]}</div><div class="value ${i===2?(totals.pnl>=0?'pos':'neg'):''}">${x[1]}</div><div class="hint">${x[2]}</div></article>`).join('');document.getElementById('accounts').innerHTML=ac.map(accountCard).join('');document.getElementById('universeCount').textContent=`${data.market_count||0} tracked`;const online=ac.length>0&&totals.online===ac.length;document.getElementById('healthDot').classList.toggle('warn',!online||totals.failed>0);document.getElementById('healthText').textContent=!online?'One or more paper runners need attention':totals.failed?`Runners active · ${totals.failed} market fetch failures`:'Paper runners checking market data';document.getElementById('updated').textContent=`Updated ${new Date().toLocaleTimeString()}`;const pos=ac.flatMap(a=>(a.open_positions||[]).map(p=>({strategy:a.strategy,...p})));document.getElementById('positions').innerHTML=pos.length?pos.map(p=>`<tr><td>${esc(labels[p.strategy])}</td><td>${esc(p.pair)}</td><td>${money(p.mark_price)}</td><td class="${Number(p.unrealized_pnl)>=0?'pos':'neg'}">${money(p.unrealized_pnl)}</td></tr>`).join(''):'<tr><td colspan="4" class="empty">No open simulated positions yet</td></tr>';const trades=ac.flatMap(a=>(a.trade_log||[]).map(t=>({strategy:a.strategy,...t}))).sort((a,b)=>Number(b.exit_timestamp||0)-Number(a.exit_timestamp||0)).slice(0,12);document.getElementById('trades').innerHTML=trades.length?trades.map(t=>`<tr><td>${esc(labels[t.strategy])}</td><td>${esc(t.pair)}</td><td>${t.exit_timestamp?new Date(t.exit_timestamp*1000).toLocaleString():'—'}</td><td class="${Number(t.pnl)>=0?'pos':'neg'}">${money(t.pnl)}</td></tr>`).join(''):'<tr><td colspan="4" class="empty">Closed trades will appear here</td></tr>';const notes=ac.find(a=>a.strategy==='llama_regime')?.model_notes||{};const items=Object.entries(notes);document.getElementById('aiNotes').innerHTML=items.length?items.slice(0,8).map(([pair,n])=>{const news=(n.news||[]).slice(0,3).map(h=>'<div class="sub" style="margin:4px 0 0"><a href="'+esc(h.url)+'" target="_blank" rel="noopener noreferrer">'+esc(h.title)+'</a><br>'+esc(h.source)+' · '+esc(h.published_at)+'</div>').join('');return '<div class="regime-line" style="padding:8px 0;border-top:1px solid var(--line)"><strong>'+esc(pair)+'</strong><span>'+esc(n.regime)+' · '+(Number(n.confidence||0)*100).toFixed(0)+'%</span></div><div class="sub" style="margin:-4px 0 8px">'+esc(n.rationale||'')+'</div><div class=\"sub\" style=\"margin:-4px 0 8px\">News feed: '+esc(n.news_status||'not checked')+'</div>'+news}).join(''):'<div class="sub">The local model’s notes appear after it evaluates the next completed 15-minute bar.</div>';drawChart(ac)}
const renderCryptoDashboard=render;
render=function(data){renderCryptoDashboard(data);const stock=(data.accounts||[]).find(a=>a.strategy==='stocks_llama_regime');const universe=document.querySelector('.middle .card .sub[style*="line-height"]');if(universe)universe.textContent=stock?'Crypto uses 15-minute Llama regimes across liquid coins. The separate U.S. equities experiment uses daily bars and after-close targets. Both accounts are simulated.':'Crypto Llama classifies 15-minute regimes across liquid coins. Position limits remain fixed.';if(!stock)return;const card=[...document.querySelectorAll('#accounts .account')].find(node=>node.querySelector('.strategy')?.textContent===labels.stocks_llama_regime);if(card){const mode=[...card.querySelectorAll('.account-row')].find(row=>row.firstElementChild?.textContent==='Processing mode');if(mode)mode.lastElementChild.textContent=stock.last_processing_mode==='live_forward_daily_close'?'Next-session target staged':stock.last_processing_mode==='live_forward_daily_close_skipped_stale_targets'?'Stale target skipped':'Waiting for US close';const details=document.createElement('div');details.className='regime';details.innerHTML='<div class="regime-line"><strong>Data</strong><span>Yahoo Finance · daily</span></div><div class="regime-line" style="margin-top:6px"><span>Dividend income</span><strong>'+money(stock.dividend_income||0)+'</strong></div><div class="regime-line" style="margin-top:6px"><span>Next-session buy targets</span><strong>'+Number(stock.pending_target_count||0)+'</strong></div><div class="regime-line" style="margin-top:6px"><span>Latest regime notes</span><strong>'+Object.keys(stock.model_notes||{}).length+' symbols</strong></div>';card.appendChild(details)}const notes=Object.entries(stock.model_notes||{});if(notes.length)document.getElementById('aiNotes').innerHTML='<div class="regime-line"><strong>US equities · daily regimes</strong><span>after close</span></div>'+notes.slice(0,8).map(([symbol,n])=>'<div class="regime-line" style="padding:8px 0;border-top:1px solid var(--line)"><strong>'+esc(symbol)+'</strong><span>'+esc(n.regime||'unknown')+' · '+(Number(n.confidence||0)*100).toFixed(0)+'%</span></div><div class="sub" style="margin:-4px 0 8px">'+esc(n.rationale||'')+'</div>').join('')};
async function refresh(){try{const response=await fetch('/api/dashboard',{cache:'no-store'});if(!response.ok)throw new Error('dashboard API unavailable');render(await response.json())}catch(e){document.getElementById('healthDot').classList.add('warn');document.getElementById('healthText').textContent='Waiting for local paper data'}}
refresh();setInterval(refresh,15000);window.addEventListener('resize',refresh);
</script></body></html>"""


def dashboard_data(database: str | Path, stocks_database: str | Path | None = None) -> dict[str, Any]:
    store = MarketStore(database)
    accounts: list[dict[str, Any]] = []
    now = int(datetime.now(timezone.utc).timestamp())
    market_count = 0
    intervals: set[int] = set()
    for name in PAPER_STRATEGY_NAMES:
        state = store.get_paper_state(name)
        if not state or not state.get("active", True):
            continue
        summary = paper_summary(state)
        intervals.add(int(state.get("interval", 60)))
        last_check = int(state.get("last_check_timestamp", 0))
        errors = state.get("feed_errors", [])
        market_status = state.get("market_status", {})
        summary.update({
            "equity_curve": state.get("equity_curve", []),
            "last_check_timestamp": last_check,
            "online": bool(last_check and now - last_check <= 900),
            "feed_errors": errors,
            "market_status": market_status,
            "last_processing_mode": state.get("last_processing_mode"),
            "last_downtime_gap": state.get("last_downtime_gap"),
            "successful_market_count": int(market_status.get("successful_markets", 0)),
            "failed_market_count": int(market_status.get("failed_markets", len(errors))),
            "snapshot_excluded_market_count": int(market_status.get("snapshot_excluded_markets", 0)),
        })
        market_count = max(market_count, len(state.get("pairs", [])))
        accounts.append(summary)
    if stocks_database and Path(stocks_database).exists():
        stock_store = MarketStore(stocks_database)
        state = stock_store.get_paper_state(STOCK_PAPER_ACCOUNT_ID)
        if state and state.get("active", True):
            summary = paper_summary(state)
            status = state.get("market_status", {})
            errors = state.get("feed_errors", [])
            last_check = int(state.get("last_check_timestamp", 0))
            intervals.add(int(state.get("interval", 1440)))
            summary.update({
                "equity_curve": state.get("equity_curve", []),
                "last_check_timestamp": last_check,
                "online": bool(last_check and now - last_check <= 900),
                "feed_errors": errors,
                "market_status": status,
                "last_processing_mode": state.get("last_processing_mode"),
                "last_downtime_gap": state.get("last_downtime_gap"),
                "pending_target_count": sum(
                    1 for target in state.get("pending_targets", {}).values()
                    if isinstance(target, dict) and target.get("long")
                ),
                "successful_market_count": int(status.get("successful_markets", 0)),
                "failed_market_count": int(status.get("failed_markets", len(errors))),
                "snapshot_excluded_market_count": 0,
            })
            market_count += len(state.get("pairs", []))
            accounts.append(summary)
    return {"accounts": accounts, "market_count": market_count, "intervals": sorted(intervals), "updated_utc": now}


class DashboardHandler(BaseHTTPRequestHandler):
    server: "DashboardServer"

    def do_GET(self) -> None:
        path = urlsplit(self.path).path
        if path == "/" or path == "/index.html":
            body = PAGE.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
        elif path == "/api/dashboard":
            body = json.dumps(dashboard_data(self.server.database, self.server.stocks_database), allow_nan=False).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
        else:
            body = b"Not found"
            self.send_response(404)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        return


class DashboardServer(ThreadingHTTPServer):
    def __init__(self, host: str, port: int, database: str | Path, stocks_database: str | Path | None = None) -> None:
        self.database = str(database)
        self.stocks_database = str(stocks_database) if stocks_database else None
        super().__init__((host, port), DashboardHandler)


def serve_dashboard(host: str = "127.0.0.1", port: int = 8765, database: str = "data/market_research.sqlite3", stocks_database: str | Path | None = "data/stocks_market_research.sqlite3") -> None:
    if host not in {"127.0.0.1", "localhost"}:
        raise ValueError("The local dashboard may only bind to localhost")
    server = DashboardServer(host, port, database, stocks_database)
    print(f"Paper dashboard: http://{host}:{port}/", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
