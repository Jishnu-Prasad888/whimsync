ALTER TABLE "ingestion_records" ALTER COLUMN "parsing_tier" SET DEFAULT 'balanced';--> statement-breakpoint
UPDATE "ingestion_records" SET "parsing_tier" = 'balanced' WHERE "parsing_tier" IN ('smart', 'fast');
