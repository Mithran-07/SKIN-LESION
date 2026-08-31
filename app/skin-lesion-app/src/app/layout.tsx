import type { Metadata } from "next";
import Link from "next/link";
import { Activity, ShieldAlert, Sparkles, LayoutDashboard, BookOpen, Layers, Microscope } from "lucide-react";
import "./globals.css";

export const metadata: Metadata = {
  title: "Skin Lesion AI — EfficientNet-B4 Dermoscopic Classification",
  description: "Academic Research Prototype for Skin Lesion Classification with Grad-CAM Model Attribution.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark">
      <body className="min-h-screen bg-slate-950 text-slate-100 flex flex-col antialiased selection:bg-cyan-500/20 selection:text-cyan-300">
        
        {/* Strict Medical Disclaimer Header Banner */}
        <div className="bg-amber-950/70 border-b border-amber-500/30 px-4 py-2 text-xs font-medium text-amber-200 flex items-center justify-center gap-2 text-center backdrop-blur-md sticky top-0 z-50">
          <ShieldAlert className="w-4 h-4 text-amber-400 shrink-0" />
          <span>
            <strong>ACADEMIC RESEARCH PROTOTYPE ONLY:</strong> This system is not intended for clinical diagnosis or treatment. Consult a board-certified dermatologist for medical evaluation.
          </span>
        </div>

        {/* Global Navigation Bar */}
        <header className="border-b border-slate-800/80 bg-slate-900/60 backdrop-blur-lg sticky top-8 z-40">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
            <Link href="/" className="flex items-center gap-3 group">
              <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-cyan-500 to-blue-600 flex items-center justify-center shadow-lg shadow-cyan-500/20 group-hover:scale-105 transition-transform">
                <Microscope className="w-5 h-5 text-white" />
              </div>
              <div>
                <div className="font-bold text-base tracking-tight text-white flex items-center gap-2">
                  DermAI Research
                  <span className="text-[10px] uppercase font-semibold px-2 py-0.5 rounded-full bg-cyan-500/10 text-cyan-400 border border-cyan-500/20">
                    EfficientNet-B4
                  </span>
                </div>
                <div className="text-xs text-slate-400">Non-Melanoma & Melanoma Classification</div>
              </div>
            </Link>

            {/* Nav Links */}
            <nav className="flex items-center gap-1 sm:gap-2">
              <Link
                href="/classify"
                className="flex items-center gap-2 px-3 py-2 rounded-lg text-sm font-medium text-slate-200 hover:text-cyan-300 hover:bg-slate-800/60 transition-colors"
              >
                <Sparkles className="w-4 h-4 text-cyan-400" />
                <span>Classify</span>
              </Link>

              <Link
                href="/dashboard"
                className="flex items-center gap-2 px-3 py-2 rounded-lg text-sm font-medium text-slate-200 hover:text-cyan-300 hover:bg-slate-800/60 transition-colors"
              >
                <LayoutDashboard className="w-4 h-4 text-blue-400" />
                <span>Dashboard</span>
              </Link>

              <Link
                href="/research"
                className="flex items-center gap-2 px-3 py-2 rounded-lg text-sm font-medium text-slate-200 hover:text-cyan-300 hover:bg-slate-800/60 transition-colors"
              >
                <BookOpen className="w-4 h-4 text-emerald-400" />
                <span>Research</span>
              </Link>

              <Link
                href="/architecture"
                className="flex items-center gap-2 px-3 py-2 rounded-lg text-sm font-medium text-slate-200 hover:text-cyan-300 hover:bg-slate-800/60 transition-colors"
              >
                <Layers className="w-4 h-4 text-purple-400" />
                <span>Architecture</span>
              </Link>
            </nav>
          </div>
        </header>

        {/* Main Content Area */}
        <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-8">
          {children}
        </main>

        {/* Global Footer */}
        <footer className="border-t border-slate-800/80 bg-slate-950/80 py-6 text-xs text-slate-500">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 flex flex-col sm:flex-row items-center justify-between gap-4 text-center sm:text-left">
            <div>
              <p className="text-slate-400 font-medium">Advanced Deep Learning for Non-Melanoma Dermoscopic Classification</p>
              <p className="text-slate-500 text-[11px] mt-0.5">HAM10000 Dataset • EfficientNet-B4 vs Dual-Branch CNN Empirical Investigation</p>
            </div>
            <div className="flex items-center gap-4 text-slate-400">
              <span className="inline-flex items-center gap-1.5 text-emerald-400 bg-emerald-500/10 px-2.5 py-1 rounded-full border border-emerald-500/20 text-[11px]">
                <Activity className="w-3 h-3" />
                Apple M4 MPS Verified
              </span>
              <span>College Final Project</span>
            </div>
          </div>
        </footer>

      </body>
    </html>
  );
}
