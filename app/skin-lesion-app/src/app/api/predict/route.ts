import { NextRequest } from "next/server";
import { POST as explainPost } from "./explain/route";

export async function POST(req: NextRequest) {
  return explainPost(req);
}
