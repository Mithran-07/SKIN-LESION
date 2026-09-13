import type { Metadata } from "next";
import { Navigation } from "@/components/Navigation";
import "./globals.css";

export const metadata: Metadata = {
  title: "DermAI Research | Dermal Intelligence Lab",
  description:
    "Academic Research Prototype — EfficientNet-B4 dermoscopic classification, Grad-CAM visual attribution, and CNN benchmarking on HAM10000.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className="dark">
      <body className="min-h-screen bg-background text-on-surface flex flex-col antialiased selection:bg-primary/20 selection:text-primary">
        <Navigation />

        <main className="flex-1 w-full max-w-[1440px] mx-auto" id="main-content">
          {children}
        </main>

        <footer className="border-t border-outline-variant/15 bg-surface-container-lowest py-4">
          <div className="max-w-[1440px] mx-auto px-4 sm:px-6 lg:px-8 flex flex-col sm:flex-row items-center justify-between gap-2 text-center sm:text-left">
            <div>
              <p className="font-technical-label text-[11px] text-on-surface">
                DermAI — Dermal Intelligence Lab
              </p>
              <p className="font-technical-data text-[10px] text-on-surface-variant mt-0.5 opacity-70">
                HAM10000 Dataset · EfficientNet-B4 (Best Evaluated) · Decoupled Dual-Branch CNN Investigation
              </p>
            </div>
            <p className="font-technical-data text-[10px] text-on-surface-variant opacity-60">
              Academic research only — not for clinical diagnosis
            </p>
          </div>
        </footer>
      </body>
    </html>
  );
}
