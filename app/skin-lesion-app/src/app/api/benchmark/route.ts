import { NextResponse } from "next/server";
import { BENCHMARK_DATA, DATASET, OOD_V2_SUMMARY } from "@/lib/constants";

export async function GET() {
  const champion = BENCHMARK_DATA[0];
  return NextResponse.json({
    registry_version: "2.0.0",
    evaluation_dataset: "HAM10000 (Human Against Machine with 10,000 training images)",
    split_strategy:
      "Lesion-grouped stratified split (7,002 train / 1,503 val / 1,510 test; zero lesion overlap)",
    total_images: DATASET.total,
    unique_lesions: DATASET.uniqueLesions,
    num_classes: 7,
    best_overall_model: {
      model_name: champion.name,
      architecture_family: champion.family,
      parameters: champion.exactParams,
      checkpoint_sha256:
        "125340ffa59ca4a18fb0cd0bdca024eae9ca2fe3ce54d1f230b465bd9ad9658e",
      checkpoint_epoch: champion.epoch,
      test_n_samples: 1510,
      test_accuracy: champion.acc / 100,
      test_balanced_accuracy: champion.balAcc / 100,
      test_macro_f1: champion.f1 / 100,
      test_macro_auc: champion.auc / 100,
      test_macro_pr_auc: champion.prAuc / 100,
      inference_latency_ms: champion.mpsLatencyMs,
      historical_cuda_rtx3050_latency_ms: champion.cudaLatencyMs,
    },
    full_model_comparison: BENCHMARK_DATA,
    ood_gate_v2: OOD_V2_SUMMARY,
  });
}
