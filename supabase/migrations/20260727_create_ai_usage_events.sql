create table if not exists public.ai_usage_events (
  id uuid primary key,
  user_id text not null,
  request_id text,
  tool text not null,
  modality text not null,
  model text,
  input_tokens bigint not null default 0 check (input_tokens >= 0),
  output_tokens bigint not null default 0 check (output_tokens >= 0),
  media_tokens bigint not null default 0 check (media_tokens >= 0),
  total_tokens bigint not null default 0 check (total_tokens >= 0),
  details jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now()
);

create index if not exists ai_usage_events_user_created_idx
  on public.ai_usage_events (user_id, created_at desc);
create index if not exists ai_usage_events_tool_created_idx
  on public.ai_usage_events (tool, created_at desc);

alter table public.ai_usage_events enable row level security;

drop policy if exists "Users read their own AI usage" on public.ai_usage_events;
create policy "Users read their own AI usage"
  on public.ai_usage_events for select
  using (auth.uid()::text = user_id);
