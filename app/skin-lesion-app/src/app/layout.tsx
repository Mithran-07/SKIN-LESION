import type { Metadata } from "next";
import { Navigation } from "@/components/Navigation";
import { Cpu, ShieldCheck } from "lucide-react";
import "./globals.css";

export const metadata: Metadata = {
  title: "DermAI Research | Dermal Intelligence Lab",
  description: "Academic Research Prototype for Skin Lesion Classification with Grad-CAM Model Attribution.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark">
      <body className="min-h-screen bg-background text-on-surface flex flex-col antialiased selection:bg-primary/20 selection:text-primary">
        
        {/* Navigation Shell */}
        <Navigation />

        {/* Main Content Area */}
        <div className="flex-1 w-full max-w-[1440px] mx-auto">
          {children}
        </div>

        {/* Stitch Quiet Futurism Footer */}
        <footer className="border-t border-outline-variant/15 bg-surface-container-lowest py-6 text-xs text-on-surface-variant/70 mt-auto">
          <div className="max-w-[1440px] mx-auto px-4 sm:px-6 lg:px-8 flex flex-col sm:flex-row items-center justify-between gap-4 text-center sm:text-left">
            <div>
              <p className="text-on-surface font-technical-label text-xs">
                DermAI • Advanced Deep Learning for Non-Melanoma Dermoscopic Classification
              </p>
              <p className="text-on-surface-variant font-technical-data text-[11px] mt-0.5 opacity-75">
                HAM10000 Dataset • EfficientNet-B4 (Deployed Model) • Decoupled Dual-Branch CNN Investigation
              </p>
            </div>
            <div className="flex items-center gap-4">
              <span className="inline-flex items-center gap-1.5 text-status-benign bg-status-benign/10 px-2.5 py-1 rounded border border-status-benign/20 font-technical-data text-[11px]">
                <Cpu className="w-3 h-3" />
                Apple M4 MPS Hardware Active
              </span>
              <span className="font-technical-label text-[11px] text-on-surface-variant">
                College Final Project
              </span>
            </div>
          </div>
        </footer>

      </body>
    </html>
  );
}
