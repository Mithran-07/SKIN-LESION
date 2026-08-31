"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { Sparkles, LayoutDashboard, BookOpen, Layers, Microscope, ShieldAlert, Cpu, Terminal, Radio } from "lucide-react";

export function Navigation() {
  const pathname = usePathname();

  const navItems = [
    { href: "/", label: "Home" },
    { href: "/classify", label: "Workstation" },
    { href: "/dashboard", label: "Dashboard" },
    { href: "/research", label: "Dataset & Story" },
    { href: "/architecture", label: "Architecture" },
  ];

  return (
    <>
      {/* Strict Medical Disclaimer Header Banner */}
      <div className="bg-surface-container-lowest border-b border-status-warning/20 px-4 py-1.5 text-xs text-status-warning flex items-center justify-center gap-2 text-center sticky top-0 z-50">
        <ShieldAlert className="w-3.5 h-3.5 text-status-warning shrink-0" />
        <span className="font-technical-label tracking-wide text-[11px]">
          <strong>ACADEMIC RESEARCH PROTOTYPE ONLY:</strong> Not intended for clinical diagnosis. Consult a board-certified dermatologist for medical evaluation.
        </span>
      </div>

      {/* Stitch TopNavBar */}
      <header className="bg-background/85 backdrop-blur-xl w-full border-b border-outline-variant/15 sticky top-[31px] z-40">
        <div className="max-w-[1440px] mx-auto px-4 sm:px-6 lg:px-8 h-16 flex justify-between items-center">
          
          {/* Brand */}
          <div className="flex items-center gap-4 sm:gap-6">
            <Link href="/" className="flex items-center gap-2.5 group">
              <div className="w-8 h-8 rounded bg-surface-container-high border border-primary/30 flex items-center justify-center text-primary group-hover:border-primary transition-all">
                <Microscope className="w-4 h-4 text-primary" />
              </div>
              <div className="font-headline-md text-base sm:text-lg font-bold tracking-tight text-primary">
                DermAI <span className="text-on-surface-variant font-normal text-xs sm:text-sm hidden md:inline">| Dermal Intelligence Lab</span>
              </div>
            </Link>

            {/* Navigation Links */}
            <nav className="hidden md:flex items-center gap-6 ml-4">
              {navItems.map((item) => {
                const isActive = pathname === item.href;
                return (
                  <Link
                    key={item.href}
                    href={item.href}
                    className={`font-technical-label text-[13px] tracking-wider transition-all relative py-1 ${
                      isActive
                        ? "text-primary font-bold opacity-100"
                        : "text-on-surface-variant font-medium hover:text-primary opacity-80"
                    }`}
                  >
                    {item.label}
                    {isActive && (
                      <span className="absolute bottom-0 left-0 right-0 h-[2px] bg-primary rounded-full" />
                    )}
                  </Link>
                );
              })}
            </nav>
          </div>

          {/* Right Telemetry Indicators */}
          <div className="flex items-center gap-3 text-on-surface-variant">
            <div className="hidden sm:flex items-center gap-2 px-2.5 py-1 rounded bg-surface-container border border-outline-variant/20 font-technical-data text-[11px] text-on-surface-variant">
              <span className="w-1.5 h-1.5 rounded-full bg-status-benign pulse-dot-emerald"></span>
              <span>MPS ACCELERATED</span>
            </div>

            <div className="flex items-center gap-1.5 text-primary">
              <span title="Apple Silicon MPS Active" className="p-1.5 rounded hover:bg-surface-variant transition-colors">
                <Cpu className="w-4 h-4 text-primary/80 hover:text-primary" />
              </span>
              <span title="FastAPI 0.115 Connected" className="p-1.5 rounded hover:bg-surface-variant transition-colors">
                <Terminal className="w-4 h-4 text-primary/80 hover:text-primary" />
              </span>
              <span title="Inference Engine Ready" className="p-1.5 rounded hover:bg-surface-variant transition-colors">
                <Radio className="w-4 h-4 text-primary/80 hover:text-primary" />
              </span>
            </div>
          </div>

        </div>

        {/* Mobile Navigation Row */}
        <div className="md:hidden flex items-center justify-around px-2 py-2 border-t border-outline-variant/10 bg-surface-container-lowest">
          {navItems.map((item) => {
            const isActive = pathname === item.href;
            return (
              <Link
                key={item.href}
                href={item.href}
                className={`font-technical-label text-[11px] py-1 px-2 rounded ${
                  isActive ? "text-primary bg-primary/10 font-bold" : "text-on-surface-variant hover:text-primary"
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
