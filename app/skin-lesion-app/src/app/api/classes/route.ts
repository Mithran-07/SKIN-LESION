import { NextResponse } from "next/server";
import { CLASS_METADATA, CLASS_ORDER } from "@/lib/constants";

export async function GET() {
  return NextResponse.json({
    num_classes: CLASS_ORDER.length,
    canonical_order: CLASS_ORDER,
    class_mapping_version: "ham10000_alphabetical_7class_v1",
    taxonomy: CLASS_METADATA,
    disclaimer:
      "ACADEMIC RESEARCH DEMONSTRATION ONLY. This system is an experimental deep learning prototype trained on the HAM10000 dataset and is NOT a certified medical device.",
  });
}
