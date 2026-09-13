"use client";

import { CheckCircle2, Loader2, ShieldAlert, XCircle, AlertTriangle, Cpu } from "lucide-react";

export type StatusVariant =
  | "ready"
  | "validating"
  | "processing"
  | "success"
  | "rejected"
  | "error"
  | "unavailable";

const VARIANTS: Record<
  StatusVariant,
  { label: string; className: string; Icon: React.ElementType; spin?: boolean }
> = {
  ready:       { label: "READY",             Icon: CheckCircle2, className: "text-status-benign   bg-status-benign/10   border-status-benign/30"   },
  validating:  { label: "VALIDATING",        Icon: Loader2,      className: "text-primary          bg-primary/10          border-primary/30",         spin: true },
  processing:  { label: "RUNNING INFERENCE", Icon: Cpu,          className: "text-primary          bg-primary/10          border-primary/30",         spin: true },
  success:     { label: "ANALYSIS COMPLETE", Icon: CheckCircle2, className: "text-status-benign   bg-status-benign/10   border-status-benign/30"   },
  rejected:    { label: "ANALYSIS REJECTED", Icon: ShieldAlert,  className: "text-status-critical  bg-status-critical/10  border-status-critical/30" },
  error:       { label: "ERROR",             Icon: AlertTriangle, className: "text-status-warning  bg-status-warning/10   border-status-warning/30"  },
  unavailable: { label: "MODEL UNAVAILABLE", Icon: XCircle,      className: "text-status-warning  bg-status-warning/10   border-status-warning/30"  },
};

export function StatusBadge({ variant }: { variant: StatusVariant }) {
  const v = VARIANTS[variant];
  return (
    <span
      className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded border font-technical-data text-[11px] font-semibold ${v.className}`}
    >
      <v.Icon className={`w-3 h-3 ${v.spin ? "animate-spin" : ""}`} />
      {v.label}
    </span>
  );
}
