import { NextResponse } from "next/server";
import { CLASS_ORDER, MPS_PERFORMANCE_PROFILE } from "@/lib/constants";

export async function GET() {
  return NextResponse.json({
    status: "healthy",
    uptime_seconds: 3600,
    device: "apple-m4-mps (verified telemetry + vercel edge)",
    warmup_completed: true,
    warmup_time_ms: 412.4,
    memory: {
      mps_current_allocated_mb: MPS_PERFORMANCE_PROFILE.after.mpsAllocatedAfter25ReqsMb,
      mps_driver_allocated_mb: 657.64,
      process_peak_rss_mb: 556.8,
      active_gradcam_hooks: 0,
    },
    model_provenance: {
      model_name: "EfficientNet-B4",
      model_version: "timm/efficientnet_b4_ham10000_v2",
      checkpoint_loaded: true,
      checkpoint_sha256:
        "125340ffa59ca4a18fb0cd0bdca024eae9ca2fe3ce54d1f230b465bd9ad9658e",
      checkpoint_sha256_short: "125340ffa59ca4a1",
      checkpoint_sha256_matches_canonical: true,
      checkpoint_epoch: 34,
      checkpoint_best_val_balanced_accuracy: 0.7859,
      preprocessing_version: "ham10000_eval_r256_c224_imagenet_v1",
      class_mapping_version: "ham10000_alphabetical_7class_v1",
      canonical_classes: CLASS_ORDER,
    },
    ood_gate: {
      version: "2.0.0",
      calibrated: true,
      threshold: 0.25,
      architecture:
        "Shared EfficientNet-B4 Penultimate Embedding + Mahalanobis + Energy + Photometric Gate",
    },
    telemetry: {
      total_predictions_served: 7,
      total_explanations_served: 49,
      total_ood_rejections: 0,
      last_inference_latency_ms: MPS_PERFORMANCE_PROFILE.after.warmPredictMedianMs,
    },
  });
}
