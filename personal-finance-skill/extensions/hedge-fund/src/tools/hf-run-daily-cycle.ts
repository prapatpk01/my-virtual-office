import type { DailyCycleReport, CycleMode, ToolContext, ToolResult } from "../types.js"
import { hfResearchBriefTool } from "./hf-research-brief.js"
import { hfGenerateSignalsTool } from "./hf-generate-signals.js"
import { hfRiskReviewTool } from "./hf-risk-review.js"
import { hfExecuteTradesTool } from "./hf-execute-trades.js"

function cycleId(): string {
  return `hf-${new Date().toISOString().slice(0, 10)}-${Math.random().toString(36).slice(2, 7)}`
}

function buildSummary(r: Omit<DailyCycleReport, "summary">): string {
  const parts: string[] = [`[${r.mode}] symbols: ${r.symbols.join(", ")}`]
  if (r.research) parts.push(`market: ${r.research.marketCondition}`)
  if (r.signals) {
    const dirs = r.signals.signals.reduce(
      (acc, s) => { acc[s.direction] = (acc[s.direction] ?? 0) + 1; return acc },
      {} as Record<string, number>
    )
    parts.push(`signals: ${dirs.BUY ?? 0} BUY / ${dirs.SELL ?? 0} SELL / ${dirs.HOLD ?? 0} HOLD (regime: ${r.signals.marketRegime})`)
  }
  if (r.risk) {
    const approved = r.risk.decisions.filter(d => d.approved).length
    const halted = r.risk.tradingHalted ? " ⚠ HALTED" : ""
    parts.push(`risk: ${approved} approved${halted}`)
  }
  if (r.execution) {
    parts.push(
      `execution: ${r.execution.successCount}/${r.execution.trades.length} orders submitted ($${r.execution.totalNotional.toFixed(2)})`
    )
  }
  return parts.join(" | ")
}

export const hfRunDailyCycleTool = {
  name: "hf_run_daily_cycle",
  description:
    "Run the full multi-team hedge fund cycle: Research → Quant Signals → Risk Review → Execution. Control how far to run with the mode parameter. Use full_dry_run (default) to see decisions without placing orders.",
  input_schema: {
    type: "object",
    properties: {
      symbols: {
        type: "array",
        items: { type: "string" },
        description: "Override the default trading universe for this cycle",
      },
      mode: {
        type: "string",
        enum: ["research_only", "signals_only", "full_dry_run", "full"],
        description:
          "research_only — stop after Research | signals_only — stop after Quant | full_dry_run — run all teams, skip order submission | full — execute live orders (requires confirm: true)",
        default: "full_dry_run",
      },
      confirm: {
        type: "boolean",
        description: "Required when mode is 'full' to authorise live order submission",
      },
      base_notional_per_trade: {
        type: "number",
        description: "Override base dollar notional per trade for this cycle",
      },
    },
    additionalProperties: false,
  },
  async handler(
    input: {
      symbols?: string[]
      mode?: CycleMode
      confirm?: boolean
      base_notional_per_trade?: number
    },
    context: ToolContext
  ): Promise<ToolResult<DailyCycleReport>> {
    try {
      const mode: CycleMode = input.mode ?? "full_dry_run"
      const symbols = (input.symbols ?? [...context.fundConfig.universe]).map(s => s.toUpperCase())
      const id = cycleId()
      const runAt = new Date().toISOString()

      // ── 1. Research Team ──
      const researchRes = await hfResearchBriefTool.handler({ symbols }, context)
      if (!researchRes.success) return { success: false, error: `Research failed: ${researchRes.error}` }
      const research = researchRes.data!

      if (mode === "research_only") {
        const partial = { cycleId: id, runAt, mode, symbols, research, signals: null, risk: null, execution: null }
        return { success: true, data: { ...partial, summary: buildSummary(partial) } }
      }

      // ── 2. Quant Team ──
      const signalsRes = await hfGenerateSignalsTool.handler(
        { symbols, research_brief: research, base_notional_per_trade: input.base_notional_per_trade },
        context
      )
      if (!signalsRes.success) return { success: false, error: `Quant failed: ${signalsRes.error}` }
      const signals = signalsRes.data!

      if (mode === "signals_only") {
        const partial = { cycleId: id, runAt, mode, symbols, research, signals, risk: null, execution: null }
        return { success: true, data: { ...partial, summary: buildSummary(partial) } }
      }

      // ── 3. Risk Management Team ──
      const riskRes = await hfRiskReviewTool.handler(
        { signals: [...signals.signals] },
        context
      )
      if (!riskRes.success) return { success: false, error: `Risk review failed: ${riskRes.error}` }
      const risk = riskRes.data!

      if (mode === "full_dry_run") {
        const partial = { cycleId: id, runAt, mode, symbols, research, signals, risk, execution: null }
        return { success: true, data: { ...partial, summary: buildSummary(partial) } }
      }

      // ── 4. Execution Team ──
      if (input.confirm !== true) {
        return { success: false, error: "mode='full' requires confirm: true to submit live orders" }
      }
      const execRes = await hfExecuteTradesTool.handler(
        { risk_decisions: [...risk.decisions], confirm: true },
        context
      )
      if (!execRes.success) return { success: false, error: `Execution failed: ${execRes.error}` }
      const execution = execRes.data!

      const partial = { cycleId: id, runAt, mode, symbols, research, signals, risk, execution }
      return { success: true, data: { ...partial, summary: buildSummary(partial) } }
    } catch (error) {
      return { success: false, error: error instanceof Error ? error.message : String(error) }
    }
  },
}
