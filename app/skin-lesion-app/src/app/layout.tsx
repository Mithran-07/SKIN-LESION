import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";
import { Navigation } from "@/components/Navigation";

const inter = Inter({ subsets: ["latin"], variable: "--font-inter" });

export const metadata: Metadata = {
  title: "SkinAI — Dermoscopic Skin Lesion Classification",
  description:
    "Academic research prototype for AI-assisted dermoscopic skin lesion classification using EfficientNet-B4 trained on HAM10000. Not for clinical use.",
  keywords: ["dermoscopy", "skin lesion", "AI", "deep learning", "EfficientNet", "HAM10000", "academic research"],
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className={inter.variable}>
      <body className="bg-slate-950 text-slate-100 min-h-screen antialiased">
        <Navigation />
        <main className="pt-16">{children}</main>
        <footer className="border-t border-slate-800 mt-24 py-8 text-center text-slate-500 text-sm">
          <p>
            Academic Research Prototype · Dual-Branch CNN for Dermoscopic Classification ·{" "}
            <span className="text-amber-400">Not for clinical use</span>
          </p>
        </footer>
      </body>
    </html>
  );
}
