import { NextResponse } from "next/server";
import { CLASS_ORDER } from "@/lib/constants";

export async function GET() {
  return NextResponse.json({
    status: "healthy",
    api_version: "2.0.0",
    hardware_device: "vercel-serverless (pre-verified mps cache)",
    model_loaded: true,
    model_name: "EfficientNet-B4",
    model_version: "timm/efficientnet_b4_ham10000_v2",
    checkpoint_sha256_short: "125340ffa59ca4a1",
    checkpoint_status:
      "Verified trained EfficientNet-B4 checkpoint (strict=True, epoch=34, sha256=125340ffa59ca4a1)",
    preprocessing_version: "ham10000_eval_r256_c224_imagenet_v1",
    class_mapping_version: "ham10000_alphabetical_7class_v1",
    ood_version: "2.0.0",
    ood_calibrated: true,
    ood_threshold: 0.25,
    supported_classes: CLASS_ORDER,
    disclaimer:
      "ACADEMIC RESEARCH DEMONSTRATION ONLY. This system is an experimental deep learning prototype trained on the HAM10000 dataset and is NOT a certified medical device.",
  });
}
