"use client";

import { useEffect } from "react";
import { useRouter } from "next/navigation";

export default function AuditPage() {
  const router = useRouter();

  useEffect(() => {
    router.replace("/?tab=audit");
  }, [router]);

  return null;
}
