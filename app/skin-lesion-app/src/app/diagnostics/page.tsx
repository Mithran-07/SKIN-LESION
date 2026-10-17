"use client";

import { useEffect, useState, useCallback } from "react";
import {
  Activity,
  Cpu,
  HardDrive,
  RefreshCw,
  ShieldCheck,
  Zap,
  CheckCircle2,
  AlertTriangle,
} from "lucide-react";
import {
  API_URL,
  resolveApiEndpoints,
  MPS_PERFORMANCE_PROFILE,
  OOD_EVALUATION_V2,
  PREPROCESSING_STEPS,
} from "@/lib/constants";

interface LiveDiagnostics {
  status: string;
  uptime_seconds: number;
  device: string;
  warmup_completed: boolean;
  warmup_time_ms: number;
  memory: {
    mps_current_allocated_mb: number | null;
    mps_driver_allocated_mb: number | null;
    process_peak_rss_mb: number | null;
    active_gradcam_hooks: number;
  };
  model_provenance: {
    model_name: string;
    model_version: string;
    checkpoint_loaded: boolean;
    checkpoint_sha256: string;
    checkpoint_sha256_short: string;
    checkpoint_sha256_matches_canonical: boolean;
    checkpoint_epoch: number | null;
    preprocessing_version: string;
    class_mapping_version: string;
  };
  ood_gate: {
    version: string;
    calibrated: boolean;
    threshold: number;
    architecture: string;
  };
  telemetry: {
    total_predictions_served: number;
    total_explanations_served: number;
    total_ood_rejections: number;
    last_inference_latency_ms: number | null;
  };
}

export default function DiagnosticsPage() {
  const [diag, setDiag] = useState<LiveDiagnostics | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [lastUpdated, setLastUpdated] = useState<string>("");

  const fetchDiagnostics = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const candidates = resolveApiEndpoints("/diagnostics");
      let res: Response | null = null;
      let lastErr: unknown = null;
      for (const url of candidates) {
        try {
          const r = await fetch(url, { cache: "no-store" });
          if (r.ok) {
            res = r;
            break;
          }
        } catch (e) {
          lastErr = e;
        }
      }
      if (!res) {
        throw lastErr ?? new Error("Diagnostics endpoint unreachable");
      }
      const data: LiveDiagnostics = await res.json();
      setDiag(data);
      setLastUpdated(new Date().toLocaleTimeString());
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "FastAPI backend unreachable on 127.0.0.1:8000"
      );
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchDiagnostics();
  }, [fetchDiagnostics]);

  return (
    <div className="px-4 sm:px-6 lg:px-8 py-6 max-w-6xl mx-auto space-y-8">
      {/* Header */}
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div>
          <h1 className="font-headline-md text-xl font-bold text-on-surface">
            System &amp; Apple M4 MPS Hardware Diagnostics
          </h1>
          <p className="font-body-sm text-sm text-on-surface-variant mt-0.5">
            Live backend telemetry, unified memory allocation, checkpoint SHA-256 verification, and
            before/after latency profiling.
          </p>
        </div>

        <button
          onClick={fetchDiagnostics}
          disabled={loading}
          className="inline-flex items-center gap-2 px-3.5 py-2 rounded border border-primary/40 bg-primary/10 text-primary font-technical-label text-xs font-bold hover:bg-primary/20 transition-colors disabled:opacity-50"
        >
          <RefreshCw className={`w-3.5 h-3.5 ${loading ? "animate-spin" : ""}`} />
          REFRESH LIVE TELEMETRY
        </button>
      </div>

      {/* Live Backend Status Banner */}
      <div
        className={`p-4 rounded-xl border flex flex-wrap items-center justify-between gap-4 ${
          diag
            ? "border-status-benign/30 bg-status-benign/5"
            : "border-status-warning/30 bg-status-warning/5"
        }`}
      >
        <div className="flex items-center gap-3">
          {diag ? (
            <CheckCircle2 className="w-5 h-5 text-status-benign shrink-0" />
          ) : (
            <AlertTriangle className="w-5 h-5 text-status-warning shrink-0" />
          )}
          <div>
            <div className="font-technical-label text-xs font-bold text-on-surface">
              {diag
                ? `LIVE FASTAPI ENGINE ONLINE · ${diag.model_provenance.model_version} · DEVICE: ${diag.device.toUpperCase()}`
                : "SHOWING VERIFIED M4 MPS PROFILE SNAPSHOT (LIVE BACKEND OFFLINE)"}
            </div>
            <div className="font-technical-data text-[11px] text-on-surface-variant">
              {diag
                ? `Checkpoint Verified: ${diag.model_provenance.checkpoint_sha256_matches_canonical ? "YES (strict=True)" : "NO"} · Shader Warmup: ${diag.warmup_time_ms} ms · Active Hooks: ${diag.memory.active_gradcam_hooks} · Updated ${lastUpdated}`
                : `Reason: ${error ?? "Connecting..."} · Displaying verified benchmark telemetry from results/verified_v2/post_optimization_performance_profile.json`}
            </div>
          </div>
        </div>
        <div className="font-technical-data text-[11px] px-2.5 py-1 rounded bg-surface-container border border-outline-variant/20 text-on-surface">
          Endpoint: {API_URL}/diagnostics
        </div>
      </div>

      {/* Live Hardware & Memory Cards */}
      <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <div className="p-4 rounded-xl bg-surface-container border border-outline-variant/20 space-y-1">
          <div className="flex items-center justify-between text-on-surface-variant">
            <span className="font-technical-label text-[10px] uppercase">MPS Tensor Allocated</span>
            <HardDrive className="w-4 h-4 text-primary" />
          </div>
          <div className="font-technical-data text-2xl font-bold text-primary">
            {diag?.memory.mps_current_allocated_mb != null
              ? `${diag.memory.mps_current_allocated_mb.toFixed(2)} MB`
              : `${MPS_PERFORMANCE_PROFILE.after.mpsAllocatedAfter25ReqsMb.toFixed(2)} MB`}
          </div>
          <div className="font-technical-data text-[10px] text-status-benign">
            Down 55.7% from 157.84 MB baseline
          </div>
        </div>

        <div className="p-4 rounded-xl bg-surface-container border border-outline-variant/20 space-y-1">
          <div className="flex items-center justify-between text-on-surface-variant">
            <span className="font-technical-label text-[10px] uppercase">MPS Driver Pool</span>
            <Cpu className="w-4 h-4 text-primary" />
          </div>
          <div className="font-technical-data text-2xl font-bold text-on-surface">
            {diag?.memory.mps_driver_allocated_mb != null
              ? `${diag.memory.mps_driver_allocated_mb.toFixed(2)} MB`
              : "657.64 MB"}
          </div>
          <div className="font-technical-data text-[10px] text-on-surface-variant">
            Process Peak RSS:{" "}
            {diag?.memory.process_peak_rss_mb != null
              ? `${diag.memory.process_peak_rss_mb.toFixed(1)} MB`
              : "556.8 MB"}
          </div>
        </div>

        <div className="p-4 rounded-xl bg-surface-container border border-outline-variant/20 space-y-1">
          <div className="flex items-center justify-between text-on-surface-variant">
            <span className="font-technical-label text-[10px] uppercase">Metal Shader Warmup</span>
            <Zap className="w-4 h-4 text-status-benign" />
          </div>
          <div className="font-technical-data text-2xl font-bold text-status-benign">
            {diag ? (diag.warmup_completed ? "WARMED" : "COLD") : "WARMED"}
          </div>
          <div className="font-technical-data text-[10px] text-on-surface-variant">
            First-request latency: 3,765 ms → 236 ms
          </div>
        </div>

        <div className="p-4 rounded-xl bg-surface-container border border-outline-variant/20 space-y-1">
          <div className="flex items-center justify-between text-on-surface-variant">
            <span className="font-technical-label text-[10px] uppercase">Live Request Counters</span>
            <Activity className="w-4 h-4 text-primary" />
          </div>
          <div className="font-technical-data text-xl font-bold text-on-surface">
            {diag
              ? `${diag.telemetry.total_predictions_served} Pred / ${diag.telemetry.total_explanations_served} CAM`
              : "Ready"}
          </div>
          <div className="font-technical-data text-[10px] text-on-surface-variant">
            OOD Gate Rejections: {diag ? diag.telemetry.total_ood_rejections : 0}
          </div>
        </div>
      </div>

      {/* Before vs After Apple Silicon M4 MPS Optimization Table */}
      <div className="bg-surface-container rounded-xl border border-outline-variant/20 overflow-hidden">
        <div className="p-4 border-b border-outline-variant/15 bg-surface-container-high flex flex-wrap items-center justify-between gap-2">
          <span className="font-technical-label text-xs text-on-surface uppercase tracking-wider">
            Phase 4 Benchmark: Before vs. After Apple Silicon M4 MPS Optimization
          </span>
          <span className="font-technical-data text-[10px] text-primary">
            Hardware: {MPS_PERFORMANCE_PROFILE.hardware}
          </span>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-left font-technical-data text-xs">
            <thead className="bg-surface-container-lowest text-on-surface-variant uppercase text-[10px] border-b border-outline-variant/15">
              <tr>
                <th className="py-2.5 px-4">Performance / Memory Metric</th>
                <th className="py-2.5 px-4">Legacy V1 Baseline</th>
                <th className="py-2.5 px-4">Optimized V2 Engine</th>
                <th className="py-2.5 px-4">Measured Improvement</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-outline-variant/10">
              <tr>
                <td className="py-3 px-4 font-semibold text-on-surface">
                  First User /predict Latency (Post-Startup)
                </td>
                <td className="py-3 px-4 text-status-warning">
                  {MPS_PERFORMANCE_PROFILE.before.coldPredictMs.toFixed(2)} ms
                </td>
                <td className="py-3 px-4 text-status-benign font-bold">
                  {MPS_PERFORMANCE_PROFILE.after.firstUserPredictMs.toFixed(2)} ms
                </td>
                <td className="py-3 px-4 text-primary font-bold">
                  {MPS_PERFORMANCE_PROFILE.improvements.firstPredictSpeedup} (Startup Shader Warmup)
                </td>
              </tr>
              <tr>
                <td className="py-3 px-4 font-semibold text-on-surface">
                  First User /predict/explain Latency
                </td>
                <td className="py-3 px-4 text-status-warning">
                  {MPS_PERFORMANCE_PROFILE.before.coldExplainMs.toFixed(2)} ms
                </td>
                <td className="py-3 px-4 text-status-benign font-bold">
                  {MPS_PERFORMANCE_PROFILE.after.firstUserExplainMs.toFixed(2)} ms
                </td>
                <td className="py-3 px-4 text-primary font-bold">
                  {MPS_PERFORMANCE_PROFILE.improvements.firstExplainSpeedup}
                </td>
              </tr>
              <tr>
                <td className="py-3 px-4 font-semibold text-on-surface">
                  Warm /predict Median Latency (P50 / P95)
                </td>
                <td className="py-3 px-4">
                  {MPS_PERFORMANCE_PROFILE.before.warmPredictMedianMs.toFixed(2)} ms (P95:{" "}
                  {MPS_PERFORMANCE_PROFILE.before.warmPredictP95Ms.toFixed(2)} ms)
                </td>
                <td className="py-3 px-4 text-status-benign font-bold">
                  {MPS_PERFORMANCE_PROFILE.after.warmPredictMedianMs.toFixed(2)} ms (P95:{" "}
                  {MPS_PERFORMANCE_PROFILE.after.warmPredictP95Ms.toFixed(2)} ms)
                </td>
                <td className="py-3 px-4 text-primary font-bold">
                  {MPS_PERFORMANCE_PROFILE.improvements.warmPredictLatencyReduction} (Single-Pass OOD
                  Feature Reuse)
                </td>
              </tr>
              <tr>
                <td className="py-3 px-4 font-semibold text-on-surface">
                  Warm /predict/explain Median Latency (P50 / P95)
                </td>
                <td className="py-3 px-4">
                  {MPS_PERFORMANCE_PROFILE.before.warmExplainMedianMs.toFixed(2)} ms (P95:{" "}
                  {MPS_PERFORMANCE_PROFILE.before.warmExplainP95Ms.toFixed(2)} ms)
                </td>
                <td className="py-3 px-4 text-status-benign font-bold">
                  {MPS_PERFORMANCE_PROFILE.after.warmExplainMedianMs.toFixed(2)} ms (P95:{" "}
                  {MPS_PERFORMANCE_PROFILE.after.warmExplainP95Ms.toFixed(2)} ms)
                </td>
                <td className="py-3 px-4 text-primary font-bold">
                  {MPS_PERFORMANCE_PROFILE.improvements.warmExplainLatencyReduction} (Merged Forward +
                  Scoped CAM Hooks)
                </td>
              </tr>
              <tr>
                <td className="py-3 px-4 font-semibold text-on-surface">
                  Active MPS Tensor Memory (After 25 Back-to-Back Requests)
                </td>
                <td className="py-3 px-4 text-status-warning">
                  {MPS_PERFORMANCE_PROFILE.before.mpsAllocatedAfter25ReqsMb.toFixed(2)} MB
                </td>
                <td className="py-3 px-4 text-status-benign font-bold">
                  {MPS_PERFORMANCE_PROFILE.after.mpsAllocatedAfter25ReqsMb.toFixed(2)} MB
                </td>
                <td className="py-3 px-4 text-primary font-bold">
                  {MPS_PERFORMANCE_PROFILE.improvements.mpsActiveMemoryReduction} (Zero Graph Leak)
                </td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>

      {/* Checkpoint & OOD Contract Verification */}
      <div className="grid sm:grid-cols-2 gap-4">
        <div className="p-5 rounded-xl bg-surface-container border border-outline-variant/20 space-y-3">
          <div className="font-technical-label text-xs text-on-surface uppercase tracking-wider flex items-center gap-2">
            <ShieldCheck className="w-4 h-4 text-primary" /> Checkpoint &amp; OOD Gate Contract
          </div>
          <div className="font-technical-data text-xs space-y-2 text-on-surface-variant">
            <div>
              <strong className="text-on-surface">Active Model:</strong>{" "}
              {diag?.model_provenance.model_name ?? "EfficientNet-B4"} (
              {diag?.model_provenance.model_version ?? "timm/efficientnet_b4_ham10000_v2"})
            </div>
            <div className="break-all">
              <strong className="text-on-surface">Verified SHA-256:</strong>{" "}
              <span className="text-primary">
                {diag?.model_provenance.checkpoint_sha256 ??
                  "125340ffa59ca4a18fb0cd0bdca024eae9ca2fe3ce54d1f230b465bd9ad9658e"}
              </span>
            </div>
            <div>
              <strong className="text-on-surface">OOD Gate Version:</strong>{" "}
              {diag?.ood_gate.version ?? OOD_EVALUATION_V2.version}
            </div>
            <div>
              <strong className="text-on-surface">Calibrated Threshold (τ):</strong>{" "}
              {diag?.ood_gate.threshold ?? OOD_EVALUATION_V2.threshold}
            </div>
            <div>
              <strong className="text-on-surface">Held-Out Test AUROC:</strong>{" "}
              {OOD_EVALUATION_V2.auroc} (ID Accept: {OOD_EVALUATION_V2.idAcceptanceRate}% · OOD
              Reject: {OOD_EVALUATION_V2.oodRejectionRate}%)
            </div>
          </div>
        </div>

        <div className="p-5 rounded-xl bg-surface-container border border-outline-variant/20 space-y-3">
          <div className="font-technical-label text-xs text-on-surface uppercase tracking-wider">
            Deterministic Preprocessing Contract
          </div>
          <div className="space-y-2">
            {PREPROCESSING_STEPS.map((s, idx) => (
              <div
                key={s.label}
                className="flex items-center justify-between text-xs font-technical-data border-b border-outline-variant/10 pb-1.5 last:border-0"
              >
                <span className="text-on-surface font-semibold">
                  0{idx + 1}. {s.label}
                </span>
                <span className="text-on-surface-variant">{s.value}</span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
