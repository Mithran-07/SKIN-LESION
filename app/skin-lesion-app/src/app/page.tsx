import Link from "next/link";
import { Microscope, Upload, BarChart2, BookOpen, AlertTriangle } from "lucide-react";

export default function HomePage() {
  return (
    <div className="min-h-screen">
      {/* Hero */}
      <section className="relative overflow-hidden">
        <div className="absolute inset-0 bg-gradient-to-br from-sky-950/40 via-indigo-950/30 to-slate-950 pointer-events-none" />
        <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_top,rgba(56,189,248,0.1),transparent_60%)] pointer-events-none" />
        <div className="relative max-w-5xl mx-auto px-6 pt-24 pb-20 text-center">
          <div className="inline-flex items-center gap-2 bg-sky-500/10 border border-sky-500/20 rounded-full px-4 py-1.5 text-sm text-sky-400 mb-8">
            <Microscope size={14} />
            Academic Research Prototype
          </div>
          <h1 className="text-5xl md:text-6xl font-bold text-white mb-6 leading-tight">
            AI-Assisted{" "}
            <span className="gradient-text">Dermoscopic</span>
            <br />
            Skin Lesion Classification
          </h1>
          <p className="text-lg text-slate-400 max-w-2xl mx-auto mb-10 leading-relaxed">
            A research system using deep learning to classify dermoscopic skin lesion images
            across seven categories. Built on EfficientNet-B4, trained on HAM10000.
          </p>
          <div className="flex flex-wrap gap-4 justify-center">
            <Link
              href="/classify"
              className="bg-sky-500 hover:bg-sky-400 text-white px-8 py-3 rounded-xl font-medium transition-colors flex items-center gap-2"
            >
              <Upload size={18} />
              Try the Classifier
            </Link>
            <Link
              href="/dashboard"
              className="bg-slate-800 hover:bg-slate-700 text-slate-200 px-8 py-3 rounded-xl font-medium transition-colors flex items-center gap-2"
            >
              <BarChart2 size={18} />
              View Research Results
            </Link>
          </div>
        </div>
      </section>

      {/* Disclaimer */}
      <section className="max-w-4xl mx-auto px-6 mb-16">
        <div className="bg-amber-500/10 border border-amber-500/30 rounded-2xl p-5 flex gap-4">
          <AlertTriangle className="text-amber-400 flex-shrink-0 mt-0.5" size={20} />
          <div>
            <p className="text-amber-300 font-medium mb-1">Medical Disclaimer</p>
            <p className="text-slate-400 text-sm leading-relaxed">
              This system is an academic research prototype and is{" "}
              <strong className="text-amber-400">not intended to provide medical diagnosis</strong> or
              replace professional medical advice. All predictions should be interpreted only in the
              context of academic research. Consult a licensed dermatologist for any clinical concerns.
            </p>
          </div>
        </div>
      </section>

      {/* Feature Cards */}
      <section className="max-w-6xl mx-auto px-6 mb-24">
        <h2 className="text-2xl font-bold text-white mb-8 text-center">What this system includes</h2>
        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-5">
          {[
            {
              icon: <Upload className="text-sky-400" size={22} />,
              title: "Image Classification",
              desc: "Upload a dermoscopic image and receive AI-predicted class, probability score, and top-3 predictions in seconds.",
            },
            {
              icon: <Microscope className="text-indigo-400" size={22} />,
              title: "Grad-CAM Explainability",
              desc: "Visualize which regions of the image influenced the model's prediction using gradient-weighted class activation maps.",
            },
            {
              icon: <BarChart2 className="text-purple-400" size={22} />,
              title: "Research Dashboard",
              desc: "Compare all evaluated models — ResNet50, DenseNet121, EfficientNet-B4, and three variants of the Dual-Branch CNN.",
            },
            {
              icon: <BookOpen className="text-emerald-400" size={22} />,
              title: "Research Story",
              desc: "Understand the complete research journey: dataset, methodology, experiments, findings, and key limitations.",
            },
            {
              icon: <BarChart2 className="text-rose-400" size={22} />,
              title: "Performance Metrics",
              desc: "View real benchmark values: accuracy, balanced accuracy, macro F1, and ROC-AUC across all models.",
            },
            {
              icon: <AlertTriangle className="text-amber-400" size={22} />,
              title: "Honest Limitations",
              desc: "A dedicated section on what the model cannot do — class imbalance, external validation gaps, and clinical constraints.",
            },
          ].map((card, i) => (
            <div key={i} className="glass rounded-2xl p-6 hover:border-slate-600 transition-colors">
              <div className="w-10 h-10 rounded-xl bg-slate-800 flex items-center justify-center mb-4">
                {card.icon}
              </div>
              <h3 className="font-semibold text-white mb-2">{card.title}</h3>
              <p className="text-slate-400 text-sm leading-relaxed">{card.desc}</p>
            </div>
          ))}
        </div>
      </section>

      {/* Stats Banner */}
      <section className="bg-slate-900 border-y border-slate-800 py-12 mb-20">
        <div className="max-w-5xl mx-auto px-6 grid grid-cols-2 md:grid-cols-4 gap-8 text-center">
          {[
            { label: "Final Model", value: "EfficientNet-B4" },
            { label: "Test Accuracy", value: "73.64%" },
            { label: "ROC-AUC", value: "95.92%" },
            { label: "Dataset", value: "HAM10000" },
          ].map((s) => (
            <div key={s.label}>
              <div className="text-2xl font-bold gradient-text mb-1">{s.value}</div>
              <div className="text-slate-500 text-sm">{s.label}</div>
            </div>
          ))}
        </div>
      </section>

      {/* Class Information */}
      <section className="max-w-5xl mx-auto px-6 mb-20">
        <h2 className="text-2xl font-bold text-white mb-2 text-center">7 Classified Lesion Types</h2>
        <p className="text-slate-400 text-center mb-8 text-sm">HAM10000 dataset — dermoscopic images across 7 categories</p>
        <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-7 gap-3">
          {[
            { code: "NV", name: "Melanocytic Nevi", color: "sky" },
            { code: "MEL", name: "Melanoma", color: "rose" },
            { code: "BKL", name: "Benign Keratosis", color: "emerald" },
            { code: "BCC", name: "Basal Cell Carcinoma", color: "amber" },
            { code: "AKIEC", name: "Actinic Keratosis", color: "purple" },
            { code: "DF", name: "Dermatofibroma", color: "indigo" },
            { code: "VASC", name: "Vascular Lesions", color: "pink" },
          ].map((cls) => (
            <div key={cls.code} className="glass rounded-xl p-3 text-center">
              <div className="font-bold text-white text-sm mb-1">{cls.code}</div>
              <div className="text-slate-500 text-xs leading-tight">{cls.name}</div>
            </div>
          ))}
        </div>
      </section>
    </div>
  );
}
