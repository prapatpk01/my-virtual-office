# V3.1.1 OKX Position Recovery Fix

- Startup recovery now reads live OKX TP/SL from multiple API response shapes.
- Removed the unsafe 3% fallback SL.
- Existing tracked positions are re-synced during reconciliation and /positions.
- Current verified ETH short recovery values:
  - Entry 1857.85
  - SL 1872.08
  - TP1 1846.88
  - TP2 1828.88
- The verified override only applies when symbol, side, entry and amount match the current position.
