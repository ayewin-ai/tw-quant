const J = n => fetch(`data/${n}?v=${Date.now()}`).then(r => r.ok ? r.json() : null).catch(() => null);
const pct = (x, d = 2, s = true) => x == null ? 'NA' : `${s && x > 0 ? '+' : ''}${(x * 100).toFixed(d)}%`;
const cls = x => x == null ? '' : (x > 0 ? 'pos' : 'neg');
const $ = id => document.getElementById(id);
Chart.defaults.color = '#c9d1d9'; Chart.defaults.borderColor = '#2a3644';
Chart.defaults.font.family = '-apple-system,"PingFang TC","Noto Sans TC",sans-serif';
window.charts = {};

(async () => {
  const [meta, val, eq, hit, live, err, lp, lr] = await Promise.all(
    ['meta.json', 'validation.json', 'equity.json', 'hitrate_oos.json', 'performance_live.json',
     'error_patterns.json', 'latest_prediction.json', 'latest_review.json'].map(J));
  const [abl, pre] = await Promise.all(['ablation.json', 'latest_preopen.json'].map(J));
  if (abl) renderAblation(abl);
  if (pre) renderPreopen(pre);
  const ph = await J('pred_history.json');
  if (ph) renderPredHistory(ph);
  window.siteData = {meta, val, eq, hit, live, err, lp, lr};
  if (meta) {
    $('gen').textContent = `資料更新：${meta.generated_at} (UTC)`;
    const c = meta.coverage;
    $('coverage').innerHTML = `<h3>目前資料庫涵蓋</h3><table><tr><th>表</th><th>筆數</th><th>期間</th></tr>
      ${['prices', 'inst', 'margin'].map(t => `<tr><td>${t}</td><td>${c[t].rows.toLocaleString()}</td><td>${c[t].start} ~ ${c[t].end}（${c[t].days} 交易日）</td></tr>`).join('')}
      <tr><td>revenue</td><td>${c.revenue.rows.toLocaleString()}</td><td>${c.revenue.start} ~ ${c.revenue.end}</td></tr>
      <tr><td>us_daily</td><td>${c.us_daily.rows.toLocaleString()}</td><td>${c.us_daily.start} ~ ${c.us_daily.end}</td></tr></table>
      <p class="muted">共 ${c.stocks.toLocaleString()} 檔證券；歷史回補仍持續往 2024-07 延伸。</p>`;
  }
  if (val) renderVal(val);
  if (eq) renderEq(eq);
  if (val && val.decile_avg_excess_bp) renderDec(val.decile_avg_excess_bp);
  if (hit) renderHit(hit);
  renderLive(live);
  if (err) renderErr(err);
  if (lp) renderPred(lp);
  if (lr) renderReview(lr);
  if (meta) renderReports(meta.reports);
})();

function renderVal(v) {
  const b = v.backtest, n = 10;
  $('kpis').innerHTML = [
    ['可執行策略累積（扣成本）', b.cum_ret_top_open2close_net], ['加權指數同期', b.cum_ret_taiex],
    ['理論值 收盤→收盤（做不到）', b.cum_ret_top_close2close_gross]].map(([k, x]) =>
    `<div class="kpi"><b class="${cls(x)}">${pct(x, 1)}</b><span>${k}</span></div>`).join('') +
    `<div class="kpi"><b>${v.auc.toFixed(3)}</b><span>樣本外 AUC（0.5＝亂猜）</span></div>` +
    `<div class="kpi"><b>${pct(v[`top${n}_precision_excess`], 1, false)}</b><span>Top-10 勝過加權指數（全體 ${pct(v.base_rate_excess_up, 1, false)}）</span></div>`;
  $('valPeriod').textContent = `測試期間 ${v.test_period[0]} ~ ${v.test_period[1]}，${v.n_days} 個交易日、${v.n_obs.toLocaleString()} 筆股票-日，完全樣本外。`;
  const rows = [
    ['AUC（跑贏中位數）', v.auc.toFixed(4)], ['AUC（勝過加權指數）', v.auc_vs_taiex_excess.toFixed(4)],
    ['每日 Rank IC 平均', v.rank_ic_mean.toFixed(4)],
    ['Top-10 勝過加權指數比例', `${pct(v[`top${n}_precision_excess`], 1, false)}（全體基準 ${pct(v.base_rate_excess_up, 1, false)}）`],
    ['Top-10 隔日上漲比例', `${pct(v[`top${n}_precision_abs_up`], 1, false)}（全體基準 ${pct(v.universe_abs_up_rate, 1, false)}）`],
    ['Top-10 平均超額（收盤→收盤）', `${v[`top${n}_avg_excess_bp`].toFixed(1)} bp`],
    ['平均換手率', pct(b.avg_turnover, 0, false)],
    ['隔日開盤→收盤 累積（毛 / 扣成本）', `${pct(b.cum_ret_top_open2close_gross, 1)} / ${pct(b.cum_ret_top_open2close_net, 1)}`],
    ['收盤→收盤 累積（毛 / 扣成本，無法實現）', `${pct(b.cum_ret_top_close2close_gross, 0)} / ${pct(b.cum_ret_top_close2close_net, 0)}`],
    ['加權指數 累積', pct(b.cum_ret_taiex, 1)]];
  if (v.baseline_reversal) rows.push(['對照組：買前一天跌最多（AUC / 開盤→收盤扣成本）',
    `${v.baseline_reversal.auc.toFixed(3)} / ${pct(v.baseline_reversal.cum_ret_top_open2close_net, 1)}`]);
  $('valTable').innerHTML = `<table>${rows.map(r => `<tr><td>${r[0]}</td><td>${r[1]}</td></tr>`).join('')}</table>`;
}

function renderEq(e) {
  const ds = (label, data, color, dash, width = 2) => ({label, data, borderColor: color, backgroundColor: color,
    borderDash: dash ? [6, 4] : [], borderWidth: width, pointRadius: 0, tension: 0});
  charts.eq = new Chart($('eqChart'), {type: 'line', data: {labels: e.dates, datasets: [
    ds('隔日開盤→收盤（扣成本，可執行）', e.oc_net, '#ff6b6b', false, 3),
    ds('隔日開盤→收盤（毛）', e.oc_gross, '#ffa94d', false),
    ds('加權指數', e.taiex, '#4fa3ff', false, 3),
    ds('收盤→收盤（毛，無法實現）', e.cc_gross, '#8b949e', true),
    ds('收盤→收盤（扣成本，無法實現）', e.cc_net, '#5c6773', true)]},
    options: {responsive: true, maintainAspectRatio: false, interaction: {mode: 'index', intersect: false},
      scales: {y: {type: 'logarithmic', title: {display: true, text: '淨值（起始=1）'}}, x: {ticks: {maxTicksLimit: 12}}},
      plugins: {tooltip: {callbacks: {label: c => `${c.dataset.label}: ${c.parsed.y.toFixed(3)}`}}}}});
  $('logScale').onchange = ev => { charts.eq.options.scales.y.type = ev.target.checked ? 'logarithmic' : 'linear'; charts.eq.update(); };
}

function renderDec(d) {
  const k = Object.keys(d), v = Object.values(d);
  charts.dec = new Chart($('decChart'), {type: 'bar', data: {labels: k.map(x => `D${x}`), datasets: [{label: '平均隔日超額報酬（bp，收盤→收盤）',
    data: v, backgroundColor: v.map(x => x >= 0 ? '#3ddc97' : '#ff6b6b')}]},
    options: {responsive: true, maintainAspectRatio: false, scales: {x: {title: {display: true, text: '模型分數十分位（D0 最低 → D9 最高）'}}}}});
}

function renderHit(h) {
  const line = (label, data, color, dash) => ({label, data, borderColor: color, borderDash: dash ? [5, 4] : [], pointRadius: 0, borderWidth: 2});
  charts.hit = new Chart($('hitChart'), {type: 'line', data: {labels: h.date, datasets: [
    line('Top-10 累積命中率', h.cum_hit_ex, '#3ddc97'), line('Top-10 20日移動命中率', h.roll20_hit_ex, '#ffd166'),
    line('全體股票勝過指數比例（累積）', h.base_ex, '#8b949e', true)]},
    options: {responsive: true, maintainAspectRatio: false, scales: {y: {min: 0, max: 1, ticks: {callback: v => `${v * 100}%`}}, x: {ticks: {maxTicksLimit: 12}}}}});
}

function renderLive(p) {
  if (!p || !p.pred_date || !p.pred_date.length) { $('liveNote').textContent = '尚無上線紀錄。'; return; }
  charts.live = new Chart($('liveChart'), {type: 'line', data: {labels: p.target_date, datasets: [
    {label: 'Top-10 累積上漲比例', data: p.cum_hit_up, borderColor: '#4fa3ff', pointRadius: 3},
    {label: 'Top-10 累積勝過指數比例', data: p.cum_hit_ex, borderColor: '#3ddc97', pointRadius: 3}]},
    options: {responsive: true, maintainAspectRatio: false, scales: {y: {min: 0, max: 1, ticks: {callback: v => `${v * 100}%`}}}}});
  $('liveNote').textContent = `已檢討 ${p.pred_date.length} 個交易日；累積超額（收盤→收盤，未扣成本）${pct(p.cum_excess[p.cum_excess.length - 1])}。樣本還很少，請勿過度解讀。`;
}

function renderErr(e) {
  charts.err = new Chart($('errChart'), {type: 'bar', data: {labels: e.causes, datasets: [
    {label: '判斷錯誤', data: e.wrong_pick, backgroundColor: '#ff6b6b'}, {label: '漏掉大漲', data: e.missed_winner, backgroundColor: '#ffd166'}]},
    options: {indexAxis: 'y', responsive: true, maintainAspectRatio: false, scales: {x: {stacked: true, title: {display: true, text: '累積次數'}}, y: {stacked: true}},
      plugins: {title: {display: true, text: '累積錯誤原因統計'}}}});
}

function renderPred(p) {
  const m = p.market;
  $('latestPred').innerHTML = `<p>依 <b>${p.pred_date}</b> 收盤資料，預測 <b>${p.target_date}</b>（股票池 ${p.universe_size} 檔；訓練 ${p.model.train_start} ~ ${p.model.train_end}，${p.model.train_rows.toLocaleString()} 筆）。</p>
  <p class="muted">市場：加權 ${pct(m.mkt_r1)}、櫃買 ${pct(m.otc_r1)}、上漲家數 ${pct(m.breadth, 0, false)}；前一晚費半 ${pct(m.us_sox)}、台積電 ADR ${pct(m.us_tsm)}。機率＝隔日跑贏全體中位數的機率。</p>
  <table><tr><th>#</th><th>代號</th><th>名稱</th><th>產業</th><th>機率</th><th>收盤</th><th>當日</th><th>主要看多因素</th><th>扣分因素</th></tr>
  ${p.top.map(r => `<tr><td>${r.rank}</td><td>${r.code}</td><td>${r.name}</td><td>${r.industry}</td><td>${(r.prob * 100).toFixed(1)}%</td><td>${r.close}</td>
    <td class="${cls(r.r1)}">${pct(r.r1)}</td><td>${r.pos.join('；')}</td><td>${r.neg.join('；')}</td></tr>`).join('')}</table>
  <p class="muted">⚠️ 非投資建議。前一日漲停股隔日常開高或買不到。</p>`;
}

function renderReview(r) {
  const p = r.perf;
  const cases = r.cases.map(c => `<div class="case"><b>${c.kind === 'wrong_pick' ? '❌ 判斷錯誤' : '🚀 漏掉大漲'}｜${c.code} ${c.name}</b>（${c.industry}，排名 #${c.rank}）
    實際 <span class="${cls(c.ret)}">${pct(c.ret)}</span>、超額 ${pct(c.excess)}、產業超額 ${pct(c.sector_excess)}<br>
    原因：${(c.causes || '').split(';').filter(Boolean).map(x => `<span class="tag">${x}</span>`).join('')}
    ${c.news_tags ? `<br><span class="muted">新聞標籤：${c.news_tags.split(';').join('、')}</span>` : ''}</div>`).join('');
  $('latestReview').innerHTML = `<p><b>${r.pred_date}</b> 的預測 vs <b>${p.target_date}</b> 實際：Top-10 上漲 ${pct(p.top_hit_abs, 0, false)}、勝過加權指數 ${pct(p.top_hit_excess, 0, false)}；
    平均報酬 ${pct(p.top_avg_ret)}、平均超額 ${pct(p.top_avg_excess)}；全體 AUC ${p.auc ? p.auc.toFixed(3) : 'NA'}、Rank IC ${p.rank_ic.toFixed(3)}；加權 ${pct(p.taiex)}。</p>
    <table><tr><th>#</th><th>代號</th><th>名稱</th><th>產業</th><th>機率</th><th>實際</th><th>超額</th></tr>
    ${r.top.map(t => `<tr><td>${t.rank}</td><td>${t.code}</td><td>${t.name}</td><td>${t.industry}</td><td>${(t.prob * 100).toFixed(1)}%</td>
      <td class="${cls(t.ret)}">${pct(t.ret)}</td><td class="${cls(t.excess)}">${pct(t.excess)}</td></tr>`).join('')}</table>
    <h3>根因分析</h3>${cases}
    <p><a onclick="showMd('reports/review/${r.markdown}')" href="#reports">看完整檢討報告（含新聞標題與特徵貢獻）</a></p>`;
}

function renderReports(rep) {
  const link = (path, label) => `<a onclick="showMd('${path}')">${label}</a>`;
  $('reportList').innerHTML = `<div class="replist"><p><b>總覽：</b>${link('reports/validation.md', '驗證報告')} ${link('reports/error_patterns.md', '累積錯誤模式')}</p>
    <p><b>每日預測：</b>${rep.predict.slice().reverse().map(f => link(`reports/predict/${f}`, f.replace('predict_', '').replace('.md', ''))).join('')}</p>
    <p><b>每日檢討：</b>${rep.review.slice().reverse().map(f => link(`reports/review/${f}`, f.replace('review_', '').replace('.md', ''))).join('')}</p></div>`;
}

async function showMd(path) {
  const t = await fetch(path).then(r => r.text());
  $('mdView').innerHTML = marked.parse(t);
  $('mdView').scrollIntoView({behavior: 'smooth'});
}


async function renderAblation(a) {
  const rows = a.rows, pc = x => x == null ? 'NA' : `${x > 0 ? '+' : ''}${(x * 100).toFixed(1)}%`;
  $('ablationTable').innerHTML = `<table><tr><th>特徵組</th><th>目標</th><th>特徵數</th><th>AUC(開→收)</th><th>Rank IC</th><th>Top-10 開→收平均</th><th>全體開→收平均</th><th>開→收累積(扣成本)</th><th>加權指數</th></tr>
  ${rows.map(r => `<tr><td>${a.names[r.set] || r.set}</td><td>${r.label === 'y_oc' ? '開→收' : '收→收'}</td><td>${r.n_features}</td><td>${r.auc_oc.toFixed(4)}</td>
    <td>${r.rank_ic.toFixed(4)}</td><td>${r.top_avg_oc_bp.toFixed(1)} bp</td><td>${r.universe_avg_oc_bp.toFixed(1)} bp</td>
    <td class="${cls(r.cum_oc_net)}">${pc(r.cum_oc_net)}</td><td>${pc(r.cum_taiex)}</td></tr>`).join('')}</table>`;
  const colors = ['#8b949e', '#4fa3ff', '#ffd166', '#c084fc', '#3ddc97', '#ff6b6b'];
  const sets = rows.filter(r => r.label === 'y_oc');
  const ds = [];
  let labels = null, taiex = null;
  for (const [i, r] of sets.entries()) {
    const e = await J(`ablation_eq_${r.set.replace('+', 'p')}_${r.label}.json`);
    if (!e) continue;
    labels = labels || e.dates; taiex = taiex || e.taiex;
    ds.push({label: `${a.names[r.set] || r.set}（開→收扣成本）`, data: e.oc_net, borderColor: colors[i % 6], pointRadius: 0, borderWidth: r.set === 'preopen' ? 3 : 1.5});
  }
  if (labels) {
    ds.push({label: '加權指數', data: taiex, borderColor: '#ffffff', borderDash: [6, 4], pointRadius: 0, borderWidth: 2});
    charts.abl = new Chart($('ablChart'), {type: 'line', data: {labels, datasets: ds},
      options: {responsive: true, maintainAspectRatio: false, interaction: {mode: 'index', intersect: false},
        scales: {y: {title: {display: true, text: '淨值（起始=1，開盤買收盤賣、扣成本）'}}, x: {ticks: {maxTicksLimit: 12}}}}});
  }
  const best = sets.reduce((m, r) => (m == null || r.cum_oc_net > m.cum_oc_net) ? r : m, null);
  if (best) $('ablNote').innerHTML = `可執行版本中最好的是「${a.names[best.set] || best.set}」：開→收扣成本累積 <b class="${cls(best.cum_oc_net)}">${pc(best.cum_oc_net)}</b>，同期加權指數 ${pc(best.cum_taiex)}。` +
    (best.cum_oc_net > best.cum_taiex ? '' : ' <b>仍未勝過大盤</b>，請勿當作投資依據。');
}

function renderPreopen(p) {
  const o = p.overnight || {};
  $('latestPreopen').innerHTML = `<p>目標日 <b>${p.target_date}</b>（以 ${p.pred_date} 收盤資料＋隔夜資訊；特徵組 ${p.model.feature_set}，目標＝開盤→收盤）。</p>
  <p class="muted">台指期夜盤 ${pct(o.tx_night_ret)}、費半 ${pct(o.us_sox_on)}、台積電 ADR ${pct(o.us_tsm_on)}、那指 ${pct(o.us_ndx_on)}、美元/台幣 ${pct(o.usdtwd_on)}</p>
  <table><tr><th>#</th><th>代號</th><th>名稱</th><th>產業</th><th>機率</th><th>前日漲跌</th><th>主要因素</th></tr>
  ${p.top.map(r => `<tr><td>${r.rank}</td><td>${r.code}</td><td>${r.name}${r.limitup ? ' ⚠️漲停追高' : ''}</td><td>${r.industry}</td><td>${(r.prob * 100).toFixed(1)}%</td>
    <td class="${cls(r.r1)}">${pct(r.r1)}</td><td>${r.pos.join('；')}</td></tr>`).join('')}</table>`;
}


function renderPredHistory(ph) {
  const rows = ph.rows.map(r => Object.fromEntries(ph.columns.map((c, i) => [c, r[i]])));
  const D = rows.map(r => r.target_date), v = (r, k) => r[k] == null ? null : r[k];
  const cum = (k) => { let e = 1; return rows.map(r => { if (r[k] != null) e *= 1 + r[k]; return e - 1; }); };
  const cumHit = (k) => { let s = 0, n = 0; return rows.map(r => { if (r[k] != null) { s += r[k]; n++; } return n ? s / n : null; }); };
  const roll = (k, w = 20) => rows.map((_, i) => { const xs = rows.slice(Math.max(0, i - w + 1), i + 1).map(r => r[k]).filter(x => x != null); return xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : null; });
  const L = (label, data, color, extra = {}) => ({label, data, borderColor: color, backgroundColor: color, pointRadius: 2, borderWidth: 2, spanGaps: true, ...extra});
  const opt = (t, pctAxis = true) => ({responsive: true, maintainAspectRatio: false, interaction: {mode: 'index', intersect: false},
    plugins: {title: {display: true, text: t}}, scales: {y: pctAxis ? {ticks: {callback: x => (x * 100).toFixed(1) + '%'}} : {}}});
  charts.phCum = new Chart($('phCum'), {type: 'line', data: {labels: D, datasets: [
    L('收盤後 Top-10 累積超額（收→收，對加權，未扣成本、實務難執行）', cum('close_top10_avg_excess'), '#4fa3ff'),
    L('盤前 Top-10 累積報酬（開→收，扣成本）', cum('preopen_top10_net_ret'), '#3ddc97'),
    L('加權指數累積', cum('taiex_ret'), '#ffffff', {borderDash: [6, 4]})]}, options: opt('累積報酬')});
  charts.phHit = new Chart($('phHit'), {type: 'line', data: {labels: D, datasets: [
    L('收盤後 上漲比例（累積）', cumHit('close_top10_up'), '#4fa3ff'),
    L('收盤後 勝過加權（累積）', cumHit('close_top10_beat'), '#a5c8ff', {borderDash: [4, 3]}),
    L('盤前 上漲比例（累積）', cumHit('preopen_top10_up'), '#3ddc97'),
    L('盤前 勝過全體平均（累積）', cumHit('preopen_top10_beat'), '#9ff0c8', {borderDash: [4, 3]}),
    L('收盤後 勝過加權（20日滾動）', roll('close_top10_beat'), '#ffd166', {pointRadius: 0, borderWidth: 1}),
    L('50%', D.map(() => 0.5), '#888', {pointRadius: 0, borderWidth: 1, borderDash: [2, 2]})]}, options: opt('Top-10 命中率')});
  charts.phAuc = new Chart($('phAuc'), {type: 'line', data: {labels: D, datasets: [
    L('收盤後 AUC', rows.map(r => v(r, 'close_auc')), '#4fa3ff'),
    L('盤前 AUC', rows.map(r => v(r, 'preopen_auc')), '#3ddc97'),
    L('收盤後 Rank IC', rows.map(r => v(r, 'close_rank_ic')), '#c084fc', {borderDash: [4, 3]}),
    L('盤前 Rank IC', rows.map(r => v(r, 'preopen_rank_ic')), '#ff6b6b', {borderDash: [4, 3]}),
    L('AUC 0.5', D.map(() => 0.5), '#888', {pointRadius: 0, borderWidth: 1, borderDash: [2, 2]}),
    L('IC 0', D.map(() => 0), '#555', {pointRadius: 0, borderWidth: 1, borderDash: [2, 2]})]}, options: opt('每日 AUC / Rank IC', false)});
  charts.phSplit = new Chart($('phSplit'), {type: 'line', data: {labels: D, datasets: [
    L('收盤後 Top-10 累積 跳空', cum('close_top10_gap'), '#4fa3ff'),
    L('收盤後 Top-10 累積 盤中', cum('close_top10_intraday'), '#ffd166'),
    L('收盤後 Top-10 累積 總計', cum('close_top10_total'), '#c084fc', {borderWidth: 3}),
    L('盤前 Top-10 累積 盤中', cum('preopen_top10_intraday'), '#3ddc97'),
    L('0050 累積 跳空', cum('bm0050_gap'), '#a5c8ff', {borderDash: [5, 3]}),
    L('0050 累積 盤中', cum('bm0050_intraday'), '#ffe7a3', {borderDash: [5, 3]}),
    L('0050 累積 總計', cum('bm0050_total'), '#ffffff', {borderDash: [5, 3]})]}, options: opt('累積：跳空 vs 盤中 vs 總計（未扣成本）')});
  charts.phStack = new Chart($('phStack'), {type: 'bar', data: {labels: D, datasets: [
    {label: '收盤後 Top-10 跳空', data: rows.map(r => r.close_top10_gap), backgroundColor: '#4fa3ff', stack: 'c'},
    {label: '收盤後 Top-10 盤中', data: rows.map(r => r.close_top10_intraday), backgroundColor: '#ffd166', stack: 'c'},
    {label: '盤前 Top-10 跳空', data: rows.map(r => r.preopen_top10_gap), backgroundColor: '#1f8f63', stack: 'p'},
    {label: '盤前 Top-10 盤中', data: rows.map(r => r.preopen_top10_intraday), backgroundColor: '#3ddc97', stack: 'p'}]},
    options: {...opt('每日 Top-10 報酬貢獻：跳空 vs 盤中'), scales: {x: {stacked: true}, y: {stacked: true, ticks: {callback: x => (x * 100).toFixed(1) + '%'}}}}});
  const cols = [['target_date', '目標日', 'd'], ['close_top10_up', '收盤後 上漲比例', 'r'], ['close_top10_beat', '收盤後 勝過加權', 'r'],
    ['close_top10_avg_ret', '收盤後 平均報酬', 'p'], ['close_top10_avg_excess', '收盤後 平均超額', 'p'], ['close_auc', '收盤後 AUC', 'n'],
    ['close_rank_ic', '收盤後 Rank IC', 'n'], ['preopen_top10_up', '盤前 上漲比例', 'r'], ['preopen_top10_beat', '盤前 勝過全體', 'r'],
    ['preopen_top10_avg_ret', '盤前 開→收平均', 'p'], ['preopen_top10_avg_excess', '盤前 超額', 'p'], ['preopen_auc', '盤前 AUC', 'n'],
    ['preopen_rank_ic', '盤前 Rank IC', 'n'], ['taiex_ret', '加權指數', 'p'],
    ['close_top10_gap', '收盤後 跳空', 'p'], ['close_top10_intraday', '收盤後 盤中', 'p'], ['close_top10_total', '收盤後 總計', 'p'],
    ['preopen_top10_gap', '盤前 跳空', 'p'], ['preopen_top10_intraday', '盤前 盤中', 'p'], ['preopen_top10_total', '盤前 總計', 'p'],
    ['bm0050_gap', '0050 跳空', 'p'], ['bm0050_intraday', '0050 盤中', 'p'], ['bm0050_total', '0050 總計', 'p']];
  const fmt = (x, t) => x == null ? '—' : t === 'r' ? `${Math.round(x * 100)}%` : t === 'p' ? `<span class="${cls(x)}">${pct(x)}</span>` : t === 'n' ? x.toFixed(3) : x;
  let sortK = 'target_date', asc = false;
  const draw = () => {
    const rs = [...rows].sort((a, b) => { const x = a[sortK], y = b[sortK]; if (x == null) return 1; if (y == null) return -1; return (x < y ? -1 : x > y ? 1 : 0) * (asc ? 1 : -1); });
    $('phTable').innerHTML = `<table class="sortable"><tr>${cols.map(([k, h]) => `<th data-k="${k}" style="cursor:pointer">${h}${k === sortK ? (asc ? ' ▲' : ' ▼') : ''}</th>`).join('')}</tr>
      ${rs.map(r => `<tr>${cols.map(([k, , t]) => `<td>${fmt(r[k], t)}</td>`).join('')}</tr>`).join('')}</table>`;
    $('phTable').querySelectorAll('th').forEach(th => th.onclick = () => { const k = th.dataset.k; asc = k === sortK ? !asc : false; sortK = k; draw(); });
  };
  draw();
}
