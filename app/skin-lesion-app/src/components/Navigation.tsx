"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { Microscope, ShieldAlert, Cpu, CheckCircle2, XCircle, Loader2, Fingerprint } from "lucide-react";
import { useEffect, useState } from "react";
import { resolveApiEndpoints } from "@/lib/constants";

interface HealthStatus {
  model_loaded: boolean;
  hardware_device: string;
  model_name: string;
  checkpoint_sha256_short?: string;
  ood_calibrated?: boolean;
  ood_threshold?: number;
}

const NAV_ITEMS = [
  { href: "/",             label: "Overview"      },
  { href: "/classify",     label: "Workstation"   },
  { href: "/dashboard",    label: "Benchmarks"    },
  { href: "/research",     label: "Dataset"       },
  { href: "/architecture", label: "Architecture"  },
  { href: "/diagnostics",  label: "Diagnostics"   },
];

export function Navigation() {
  const pathname = usePathname();
  const [health, setHealth] = useState<HealthStatus | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let active = true;
    async function loadHealth() {
      for (const url of resolveApiEndpoints("/health")) {
        try {
          const r = await fetch(url);
          if (r.ok) {
            const d: HealthStatus = await r.json();
            if (active) {
              setHealth(d);
              setLoading(false);
            }
            return;
          }
        } catch {
          // Try next endpoint candidate
        }
      }
      if (active) {
        setHealth(null);
        setLoading(false);
      }
    }
    loadHealth();
    return () => {
      active = false;
    };
  }, [pathname]);

  return (
    <>
      {/* Non-Clinical Academic Research Disclaimer Banner */}
      <div className="bg-surface-container-lowest border-b border-status-warning/20 px-4 py-1.5 flex items-center justify-center gap-2 text-center sticky top-0 z-50">
        <ShieldAlert className="w-3.5 h-3.5 text-status-warning shrink-0" aria-hidden="true" />
        <span className="font-technical-label tracking-wide text-[11px] text-status-warning">
          <strong>ACADEMIC RESEARCH PROTOTYPE ONLY:</strong> Not for clinical diagnosis. Consult a board-certified dermatologist for medical evaluation.
        </span>
      </div>

      {/* Top Navigation Bar */}
      <header className="bg-background/90 backdrop-blur-xl w-full border-b border-outline-variant/15 sticky top-[31px] z-40">
        <div className="max-w-[1440px] mx-auto px-4 sm:px-6 lg:px-8 h-14 flex items-center justify-between gap-4">

          {/* Brand + Links */}
          <div className="flex items-center gap-6">
            <Link href="/" className="flex items-center gap-2.5 group">
              <div className="w-8 h-8 rounded bg-surface-container-high border border-primary/30 flex items-center justify-center group-hover:border-primary transition-colors">
                <Microscope className="w-4 h-4 text-primary" aria-hidden="true" />
              </div>
              <div>
                <span className="font-headline-md text-base font-bold text-primary tracking-tight">DermAI</span>
                <span className="hidden lg:inline text-on-surface-variant font-technical-label text-[10px] ml-2 opacity-75 uppercase tracking-widest">
                  Dermal Intelligence Lab · V2
                </span>
              </div>
            </Link>

            {/* Desktop Links */}
            <nav className="hidden md:flex items-center gap-5" aria-label="Main navigation">
              {NAV_ITEMS.map((item) => {
                const active = pathname === item.href;
                return (
                  <Link
                    key={item.href}
                    href={item.href}
                    className={`font-technical-label text-[12px] tracking-wider py-1 relative transition-colors ${
                      active
                        ? "text-primary font-bold"
                        : "text-on-surface-variant hover:text-on-surface"
                    }`}
                  >
                    {item.label}
                    {active && (
                      <span className="absolute -bottom-px left-0 right-0 h-[2px] bg-primary rounded-full" />
                    )}
                  </Link>
                );
              })}
            </nav>
          </div>

          {/* Live Provenance & Hardware Status */}
          <div className="flex items-center gap-2" role="status" aria-live="polite">
            {loading ? (
              <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded border border-outline-variant/20 bg-surface-container font-technical-data text-[11px] text-on-surface-variant">
                <Loader2 className="w-3 h-3 animate-spin" aria-hidden="true" /> CHECKING…
              </span>
            ) : health?.model_loaded ? (
              <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded border border-status-benign/30 bg-status-benign/10 font-technical-data text-[11px] text-status-benign">
                <CheckCircle2 className="w-3 h-3" aria-hidden="true" /> EFFICIENTNET-B4 READY
              </span>
            ) : (
              <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded border border-status-critical/30 bg-status-critical/10 font-technical-data text-[11px] text-status-critical">
                <XCircle className="w-3 h-3" aria-hidden="true" /> MODEL UNAVAILABLE
              </span>
            )}
            {health?.checkpoint_sha256_short && (
              <Link
                href="/diagnostics"
                title="Verified Checkpoint SHA-256 Digest (Click for System Diagnostics)"
                className="hidden xl:inline-flex items-center gap-1 px-2 py-1 rounded border border-outline-variant/15 bg-surface-container hover:border-primary/40 font-technical-data text-[10px] text-on-surface-variant transition-colors"
              >
                <Fingerprint className="w-3 h-3 text-primary" aria-hidden="true" />
                {health.checkpoint_sha256_short.slice(0, 10)}
              </Link>
            )}
            {health && (
              <Link
                href="/diagnostics"
                className="hidden sm:inline-flex items-center gap-1 px-2 py-1 rounded border border-outline-variant/15 bg-surface-container hover:border-primary/40 font-technical-data text-[11px] text-on-surface-variant transition-colors"
              >
                <Cpu className="w-3 h-3 text-primary" aria-hidden="true" />
                {health.hardware_device.toUpperCase()}
              </Link>
            )}
          </div>
        </div>

        {/* Mobile Nav Row */}
        <div className="md:hidden flex items-center justify-around px-2 py-1.5 border-t border-outline-variant/10 bg-surface-container-lowest overflow-x-auto">
          {NAV_ITEMS.map((item) => {
            const active = pathname === item.href;
            return (
              <Link
                key={item.href}
                href={item.href}
                className={`font-technical-label text-[10px] py-1 px-2 rounded whitespace-nowrap transition-colors ${
                  active ? "text-primary bg-primary/10 font-bold" : "text-on-surface-variant hover:text-primary"
                }`}
              >
                {item.label}
              </Link>
            );
          })}
        </div>
      </header>
    </>
  );
}
