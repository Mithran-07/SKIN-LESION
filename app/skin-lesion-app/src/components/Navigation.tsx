"use client";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { Microscope } from "lucide-react";

const navItems = [
  { href: "/", label: "Home" },
  { href: "/classify", label: "Classify" },
  { href: "/dashboard", label: "Research Dashboard" },
  { href: "/research", label: "Research Story" },
  { href: "/architecture", label: "Architecture" },
];

export function Navigation() {
  const pathname = usePathname();
  return (
    <header className="fixed top-0 left-0 right-0 z-50 glass border-b border-slate-800">
      <div className="max-w-7xl mx-auto px-6 h-16 flex items-center justify-between">
        <Link href="/" className="flex items-center gap-2.5 group">
          <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-sky-400 to-indigo-600 flex items-center justify-center">
            <Microscope size={16} className="text-white" />
          </div>
          <span className="font-semibold text-white">SkinAI</span>
          <span className="hidden sm:block text-slate-500 text-sm">Research</span>
        </Link>
        <nav className="flex items-center gap-1">
          {navItems.map((item) => (
            <Link
              key={item.href}
              href={item.href}
              className={`px-3 py-1.5 rounded-lg text-sm transition-colors ${
                pathname === item.href
                  ? "bg-sky-500/20 text-sky-300"
                  : "text-slate-400 hover:text-slate-200 hover:bg-slate-800"
              }`}
            >
              {item.label}
            </Link>
          ))}
        </nav>
      </div>
    </header>
  );
}
