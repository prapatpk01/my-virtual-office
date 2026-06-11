/**
 * Trading Bot Panel — Virtual Office integration
 * Connects to /api/trading/* endpoints and renders live state.
 */

class TradingPanel {
  constructor() {
    this.state = null;
    this.pollInterval = null;
    this.panel = null;
    this._init();
  }

  _init() {
    this._createPanel();
    this._startPolling();
    this._listenWS();
  }

  // -----------------------------------------------------------------------
  // Panel DOM
  // -----------------------------------------------------------------------

  _createPanel() {
    const existing = document.getElementById('trading-panel');
    if (existing) existing.remove();

    this.panel = document.createElement('div');
    this.panel.id = 'trading-panel';
    this.panel.innerHTML = `
      <div class="tp-header">
        <span class="tp-title">📈 Trading Bot</span>
        <span class="tp-badge paper" id="tp-mode-badge">PAPER</span>
        <button class="tp-toggle" id="tp-collapse-btn" title="Toggle panel">▼</button>
      </div>
      <div class="tp-body" id="tp-body">
        <!-- Balance row -->
        <div class="tp-row tp-balance-row">
          <div class="tp-stat">
            <div class="tp-stat-label">Balance</div>
            <div class="tp-stat-value" id="tp-balance">—</div>
          </div>
          <div class="tp-stat">
            <div class="tp-stat-label">P&L (total)</div>
            <div class="tp-stat-value" id="tp-pnl-total">—</div>
          </div>
          <div class="tp-stat">
            <div class="tp-stat-label">Positions</div>
            <div class="tp-stat-value" id="tp-pos-count">0</div>
          </div>
        </div>

        <!-- Controls -->
        <div class="tp-controls" id="tp-controls">
          <button class="tp-btn tp-btn-start" id="tp-start-btn">▶ Start</button>
          <button class="tp-btn tp-btn-stop" id="tp-stop-btn" disabled>■ Stop</button>
          <button class="tp-btn tp-btn-config" id="tp-config-btn">⚙ Config</button>
        </div>

        <!-- Error banner -->
        <div class="tp-error" id="tp-error" style="display:none"></div>

        <!-- Strategy signals -->
        <div class="tp-section">
          <div class="tp-section-title">Latest Signals</div>
          <div id="tp-signals" class="tp-signals-list"></div>
        </div>

        <!-- Open positions -->
        <div class="tp-section">
          <div class="tp-section-title">Open Positions</div>
          <div id="tp-positions" class="tp-positions-list"></div>
        </div>

        <!-- Recent trades -->
        <div class="tp-section">
          <div class="tp-section-title">Recent Trades</div>
          <div id="tp-trades" class="tp-trades-list"></div>
        </div>
      </div>

      <!-- Config modal -->
      <div class="tp-modal" id="tp-config-modal" style="display:none">
        <div class="tp-modal-content">
          <div class="tp-modal-header">
            <span>Bot Configuration</span>
            <button id="tp-modal-close">✕</button>
          </div>
          <div class="tp-modal-body">
            <label>Exchange
              <select id="cfg-exchange">
                <option value="binance">Binance (Crypto)</option>
                <option value="bybit">Bybit (Crypto)</option>
                <option value="alpaca">Alpaca (Stocks)</option>
              </select>
            </label>
            <label>Symbol(s) <small>(comma-separated)</small>
              <input id="cfg-symbols" type="text" value="BTC/USDT,ETH/USDT" />
            </label>
            <label>Strategies
              <div class="tp-checkboxes">
                <label><input type="checkbox" id="cfg-ma" checked /> MA Crossover</label>
                <label><input type="checkbox" id="cfg-rsi" checked /> RSI + MACD</label>
                <label><input type="checkbox" id="cfg-grid" /> Grid Trading</label>
                <label><input type="checkbox" id="cfg-ai" /> AI Signal (Claude)</label>
                <label><input type="checkbox" id="cfg-mcdx" /> MCDX Plus v2.0 📊</label>
                <label><input type="checkbox" id="cfg-sentinel" /> Sentinel Signal v1.0 🎯</label>
              </div>
            </label>
            <label>Timeframe
              <select id="cfg-timeframe">
                <option value="5m">5 min</option>
                <option value="15m">15 min</option>
                <option value="1h" selected>1 hour</option>
                <option value="4h">4 hours</option>
                <option value="1d">1 day</option>
              </select>
            </label>
            <label>Interval (seconds)
              <input id="cfg-interval" type="number" value="60" min="10" />
            </label>
            <label><input type="checkbox" id="cfg-paper" checked /> Paper Trading (simulation)</label>
            <div class="tp-key-section">
              <div class="tp-section-title">API Keys (stored locally, never sent to server in live mode)</div>
              <label>Exchange API Key <input id="cfg-api-key" type="password" placeholder="leave blank for paper mode" /></label>
              <label>Exchange Secret   <input id="cfg-api-secret" type="password" placeholder="leave blank for paper mode" /></label>
              <label>Anthropic Key (for AI Signal) <input id="cfg-anthropic-key" type="password" placeholder="sk-ant-..." /></label>
            </div>
            <div class="tp-key-section">
              <div class="tp-section-title">📲 Telegram Alerts</div>
              <label>Bot Token
                <input id="cfg-tg-token" type="password" placeholder="123456:ABC-xyz (from @BotFather)" />
              </label>
              <label>Chat ID
                <input id="cfg-tg-chat" type="text" placeholder="your Telegram chat_id" />
              </label>
              <label>Min signal confidence for alert
                <input id="cfg-tg-conf" type="number" value="0.5" min="0" max="1" step="0.05" />
              </label>
              <div style="display:flex;gap:6px;margin-top:4px;">
                <button class="tp-btn tp-btn-config" id="tp-tg-test-btn" style="flex:1">📤 Test Message</button>
                <span id="tp-tg-test-result" style="font-size:11px;align-self:center;color:#68d391"></span>
              </div>
              <div style="font-size:10px;color:#718096;margin-top:4px;line-height:1.5">
                Commands from Telegram: /status /positions /trades /balance /start_bot /stop_bot /help
              </div>
            </div>
          </div>
          <div class="tp-modal-footer">
            <button class="tp-btn tp-btn-start" id="tp-save-config">Save & Apply</button>
          </div>
        </div>
      </div>
    `;
    document.body.appendChild(this.panel);
    this._injectStyles();
    this._bindEvents();
  }

  _bindEvents() {
    const $id = id => document.getElementById(id);

    $id('tp-collapse-btn').onclick = () => {
      const body = $id('tp-body');
      const btn = $id('tp-collapse-btn');
      const hidden = body.style.display === 'none';
      body.style.display = hidden ? '' : 'none';
      btn.textContent = hidden ? '▼' : '▲';
    };

    $id('tp-start-btn').onclick = () => this._apiAction('start');
    $id('tp-stop-btn').onclick  = () => this._apiAction('stop');
    $id('tp-config-btn').onclick = () => { $id('tp-config-modal').style.display = 'flex'; };
    $id('tp-modal-close').onclick = () => { $id('tp-config-modal').style.display = 'none'; };
    $id('tp-save-config').onclick = () => this._saveConfig();
    $id('tp-tg-test-btn').onclick = () => this._testTelegram();

    // Load saved config
    this._loadConfig();
  }

  async _testTelegram() {
    const res = await fetch('/api/trading/telegram/test');
    const data = await res.json();
    const el = document.getElementById('tp-tg-test-result');
    if (el) {
      el.textContent = data.ok ? '✓ Sent!' : '✗ ' + (data.error || 'failed');
      el.style.color = data.ok ? '#68d391' : '#fc8181';
    }
  }

  // -----------------------------------------------------------------------
  // API
  // -----------------------------------------------------------------------

  async _apiAction(action) {
    const config = this._readConfig();
    const res = await fetch(`/api/trading/${action}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(config),
    });
    const data = await res.json();
    if (data.error) this._showError(data.error);
    else this._refresh();
  }

  async _refresh() {
    try {
      const res = await fetch('/api/trading/state');
      const data = await res.json();
      this._render(data);
    } catch (e) {
      // silent — WS will deliver updates anyway
    }
  }

  // -----------------------------------------------------------------------
  // WebSocket listener
  // -----------------------------------------------------------------------

  _listenWS() {
    // Attach to existing office WebSocket if available
    const checkWS = () => {
      if (window._officeWS && window._officeWS.readyState === WebSocket.OPEN) {
        const origOnMessage = window._officeWS.onmessage;
        window._officeWS.onmessage = (evt) => {
          if (origOnMessage) origOnMessage(evt);
          try {
            const msg = JSON.parse(evt.data);
            if (msg.type === 'trading_update') this._render(msg.state);
          } catch {}
        };
      } else {
        setTimeout(checkWS, 1000);
      }
    };
    checkWS();
  }

  _startPolling() {
    this._refresh();
    this.pollInterval = setInterval(() => this._refresh(), 5000);
  }

  // -----------------------------------------------------------------------
  // Render
  // -----------------------------------------------------------------------

  _render(state) {
    if (!state) return;
    this.state = state;

    const $id = id => document.getElementById(id);

    // Mode badge
    const badge = $id('tp-mode-badge');
    badge.textContent = state.paper ? 'PAPER' : 'LIVE';
    badge.className = 'tp-badge ' + (state.paper ? 'paper' : 'live');

    // Balance
    $id('tp-balance').textContent = `$${(state.balance || 0).toLocaleString('en', {minimumFractionDigits: 2})}`;

    // PnL
    const pnl = state.pnl_total || 0;
    const pnlEl = $id('tp-pnl-total');
    pnlEl.textContent = (pnl >= 0 ? '+' : '') + `$${pnl.toFixed(2)}`;
    pnlEl.className = 'tp-stat-value ' + (pnl >= 0 ? 'pos' : 'neg');

    // Position count
    $id('tp-pos-count').textContent = (state.positions || []).length;

    // Buttons
    $id('tp-start-btn').disabled = state.running;
    $id('tp-stop-btn').disabled = !state.running;

    // Error
    const errEl = $id('tp-error');
    if (state.error) { errEl.textContent = state.error; errEl.style.display = ''; }
    else errEl.style.display = 'none';

    // Signals
    const sigEl = $id('tp-signals');
    sigEl.innerHTML = (state.signals || []).slice(0, 8).map(s => `
      <div class="tp-signal-row sig-${s.type}">
        <span class="tp-sig-type">${s.type.toUpperCase()}</span>
        <span class="tp-sig-strat">${s.strategy}</span>
        <span class="tp-sig-sym">${s.symbol}</span>
        <span class="tp-sig-conf">${Math.round((s.confidence||0)*100)}%</span>
        <span class="tp-sig-reason">${s.reason || ''}</span>
      </div>
    `).join('') || '<div class="tp-empty">No signals yet</div>';

    // Positions
    const posEl = $id('tp-positions');
    posEl.innerHTML = (state.positions || []).map(p => `
      <div class="tp-pos-row">
        <span class="tp-pos-sym">${p.symbol}</span>
        <span class="tp-pos-side ${p.side}">${p.side}</span>
        <span class="tp-pos-entry">@${p.entry}</span>
        <span class="tp-pos-sl">SL: ${p.stop_loss || '—'}</span>
        <span class="tp-pos-tp">TP: ${p.take_profit || '—'}</span>
      </div>
    `).join('') || '<div class="tp-empty">No open positions</div>';

    // Trades
    const tradesEl = $id('tp-trades');
    tradesEl.innerHTML = (state.recent_trades || []).slice(0, 8).map(t => `
      <div class="tp-trade-row">
        <span class="tp-trade-side ${t.side}">${t.side.toUpperCase()}</span>
        <span class="tp-trade-sym">${t.symbol}</span>
        <span class="tp-trade-price">@${t.price?.toFixed ? t.price.toFixed(4) : t.price}</span>
        <span class="tp-trade-strat">[${t.strategy}]</span>
        ${t.paper ? '<span class="tp-paper-tag">paper</span>' : ''}
      </div>
    `).join('') || '<div class="tp-empty">No trades yet</div>';
  }

  // -----------------------------------------------------------------------
  // Config helpers
  // -----------------------------------------------------------------------

  _readConfig() {
    const $ = id => document.getElementById(id);
    return {
      exchange: $('cfg-exchange').value,
      symbols: $('cfg-symbols').value.split(',').map(s => s.trim()).filter(Boolean),
      strategies: {
        ma_crossover: $('cfg-ma').checked,
        rsi_macd:     $('cfg-rsi').checked,
        grid_trading: $('cfg-grid').checked,
        ai_signal:    $('cfg-ai').checked,
      },
      timeframe: $('cfg-timeframe').value,
      interval:  parseInt($('cfg-interval').value) || 60,
      paper:     $('cfg-paper').checked,
      api_key:    $('cfg-api-key').value,
      api_secret: $('cfg-api-secret').value,
      anthropic_key: $('cfg-anthropic-key').value,
      telegram_token:   $('cfg-tg-token').value,
      telegram_chat_id: $('cfg-tg-chat').value,
      tg_min_confidence: parseFloat($('cfg-tg-conf').value) || 0.5,
    };
  }

  _saveConfig() {
    const cfg = this._readConfig();
    localStorage.setItem('trading_bot_config', JSON.stringify(cfg));
    document.getElementById('tp-config-modal').style.display = 'none';
  }

  _loadConfig() {
    try {
      const saved = JSON.parse(localStorage.getItem('trading_bot_config') || '{}');
      if (!saved.exchange) return;
      const $ = id => document.getElementById(id);
      if (saved.exchange) $('cfg-exchange').value = saved.exchange;
      if (saved.symbols)  $('cfg-symbols').value  = saved.symbols.join(', ');
      if (saved.timeframe) $('cfg-timeframe').value = saved.timeframe;
      if (saved.interval)  $('cfg-interval').value  = saved.interval;
      if (saved.paper !== undefined) $('cfg-paper').checked = saved.paper;
      if (saved.strategies) {
        $('cfg-ma').checked   = saved.strategies.ma_crossover ?? true;
        $('cfg-rsi').checked  = saved.strategies.rsi_macd ?? true;
        $('cfg-grid').checked = saved.strategies.grid_trading ?? false;
        $('cfg-ai').checked   = saved.strategies.ai_signal ?? false;
      }
      if (saved.telegram_token)   $('cfg-tg-token').value = saved.telegram_token;
      if (saved.telegram_chat_id) $('cfg-tg-chat').value  = saved.telegram_chat_id;
      if (saved.tg_min_confidence) $('cfg-tg-conf').value = saved.tg_min_confidence;
    } catch {}
  }

  _showError(msg) {
    const el = document.getElementById('tp-error');
    if (el) { el.textContent = msg; el.style.display = ''; setTimeout(() => el.style.display = 'none', 5000); }
  }

  // -----------------------------------------------------------------------
  // Styles
  // -----------------------------------------------------------------------

  _injectStyles() {
    if (document.getElementById('trading-panel-styles')) return;
    const style = document.createElement('style');
    style.id = 'trading-panel-styles';
    style.textContent = `
      #trading-panel {
        position: fixed; bottom: 16px; right: 16px;
        width: 360px; max-height: 92vh;
        background: #1a1d2e; color: #e2e8f0;
        border: 1px solid #2d3748; border-radius: 10px;
        font-family: 'Courier New', monospace; font-size: 12px;
        box-shadow: 0 4px 24px rgba(0,0,0,.6);
        z-index: 9999; overflow: hidden;
        display: flex; flex-direction: column;
      }
      .tp-header {
        display: flex; align-items: center; gap: 8px;
        padding: 8px 12px; background: #161927;
        border-bottom: 1px solid #2d3748;
      }
      .tp-title { font-weight: bold; font-size: 13px; flex: 1; }
      .tp-badge {
        padding: 2px 8px; border-radius: 4px; font-size: 10px; font-weight: bold;
      }
      .tp-badge.paper { background: #2d4a7a; color: #90cdf4; }
      .tp-badge.live  { background: #7f1d1d; color: #fc8181; }
      .tp-toggle { background: none; border: none; color: #a0aec0; cursor: pointer; padding: 0 4px; }
      .tp-body { padding: 10px; overflow-y: auto; max-height: calc(92vh - 40px); }
      .tp-row { display: flex; gap: 8px; margin-bottom: 8px; }
      .tp-balance-row { justify-content: space-between; }
      .tp-stat { flex: 1; background: #252840; border-radius: 6px; padding: 6px 8px; text-align: center; }
      .tp-stat-label { font-size: 10px; color: #718096; margin-bottom: 2px; }
      .tp-stat-value { font-size: 14px; font-weight: bold; }
      .tp-stat-value.pos { color: #68d391; }
      .tp-stat-value.neg { color: #fc8181; }
      .tp-controls { display: flex; gap: 6px; margin-bottom: 8px; }
      .tp-btn { flex: 1; padding: 5px 8px; border: none; border-radius: 5px; cursor: pointer; font-size: 11px; font-weight: bold; }
      .tp-btn:disabled { opacity: .4; cursor: default; }
      .tp-btn-start { background: #276749; color: #9ae6b4; }
      .tp-btn-stop  { background: #742a2a; color: #fc8181; }
      .tp-btn-config{ background: #2d3748; color: #a0aec0; }
      .tp-error { background: #742a2a; color: #fc8181; padding: 5px 8px; border-radius: 5px; margin-bottom: 8px; font-size: 11px; }
      .tp-section { margin-bottom: 8px; }
      .tp-section-title { color: #718096; font-size: 10px; text-transform: uppercase; letter-spacing: .5px; margin-bottom: 4px; }
      .tp-signal-row {
        display: flex; align-items: center; gap: 4px;
        padding: 3px 6px; border-radius: 4px; margin-bottom: 2px;
        font-size: 11px; background: #1e2235;
      }
      .tp-signal-row.sig-buy  { border-left: 3px solid #68d391; }
      .tp-signal-row.sig-sell { border-left: 3px solid #fc8181; }
      .tp-signal-row.sig-hold { border-left: 3px solid #4a5568; }
      .tp-signal-row.sig-error{ border-left: 3px solid #f6ad55; }
      .tp-sig-type { font-weight: bold; min-width: 36px; }
      .sig-buy  .tp-sig-type { color: #68d391; }
      .sig-sell .tp-sig-type { color: #fc8181; }
      .sig-hold .tp-sig-type { color: #718096; }
      .tp-sig-strat { color: #90cdf4; min-width: 80px; font-size: 10px; }
      .tp-sig-sym { color: #f6e05e; min-width: 60px; }
      .tp-sig-conf { color: #d69e2e; min-width: 30px; font-size: 10px; }
      .tp-sig-reason { color: #a0aec0; font-size: 10px; overflow: hidden; white-space: nowrap; text-overflow: ellipsis; max-width: 120px; }
      .tp-pos-row, .tp-trade-row {
        display: flex; gap: 4px; align-items: center;
        padding: 3px 6px; background: #1e2235; border-radius: 4px; margin-bottom: 2px;
      }
      .tp-pos-sym, .tp-trade-sym { color: #f6e05e; min-width: 70px; font-weight: bold; }
      .tp-pos-side.long,  .tp-trade-side.buy  { color: #68d391; }
      .tp-pos-side.short, .tp-trade-side.sell { color: #fc8181; }
      .tp-pos-entry, .tp-pos-sl, .tp-pos-tp { color: #a0aec0; font-size: 10px; }
      .tp-trade-price { color: #a0aec0; }
      .tp-trade-strat { color: #90cdf4; font-size: 10px; }
      .tp-paper-tag { background: #2b4c7e; color: #90cdf4; padding: 0 4px; border-radius: 3px; font-size: 9px; }
      .tp-empty { color: #4a5568; font-style: italic; padding: 4px 6px; }
      /* Modal */
      .tp-modal {
        display: none; position: fixed; inset: 0;
        background: rgba(0,0,0,.7); z-index: 10001;
        align-items: center; justify-content: center;
      }
      .tp-modal-content {
        background: #1a1d2e; border: 1px solid #2d3748;
        border-radius: 10px; width: 380px; max-height: 90vh;
        overflow-y: auto;
      }
      .tp-modal-header {
        display: flex; justify-content: space-between; align-items: center;
        padding: 10px 14px; border-bottom: 1px solid #2d3748; font-weight: bold;
      }
      .tp-modal-header button { background: none; border: none; color: #a0aec0; cursor: pointer; font-size: 16px; }
      .tp-modal-body { padding: 14px; display: flex; flex-direction: column; gap: 10px; }
      .tp-modal-body label { display: flex; flex-direction: column; gap: 3px; font-size: 11px; color: #a0aec0; }
      .tp-modal-body input, .tp-modal-body select {
        background: #252840; border: 1px solid #2d3748; color: #e2e8f0;
        padding: 5px 8px; border-radius: 5px; font-size: 12px;
      }
      .tp-checkboxes { display: flex; flex-direction: column; gap: 4px; padding: 4px 0; }
      .tp-checkboxes label { flex-direction: row; align-items: center; gap: 6px; }
      .tp-key-section { border-top: 1px solid #2d3748; padding-top: 10px; display: flex; flex-direction: column; gap: 8px; }
      .tp-modal-footer { padding: 10px 14px; border-top: 1px solid #2d3748; display: flex; justify-content: flex-end; }
    `;
    document.head.appendChild(style);
  }
}

// Auto-initialise when DOM is ready
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', () => { window.tradingPanel = new TradingPanel(); });
} else {
  window.tradingPanel = new TradingPanel();
}
