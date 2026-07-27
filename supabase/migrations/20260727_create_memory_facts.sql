create table if not exists public.memory_facts (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users(id) on delete cascade,
  subject text not null,
  predicate text not null,
  value text not null,
  category text not null default 'general',
  status text not null default 'active' check (status in ('active','historical','superseded','disputed')),
  sensitivity text not null default 'private' check (sensitivity in ('public','internal','private','highly_sensitive')),
  confidence double precision not null default 1.0 check (confidence >= 0 and confidence <= 1),
  valid_from timestamptz,
  valid_until timestamptz,
  source text,
  source_date timestamptz,
  supersedes_id uuid references public.memory_facts(id) on delete set null,
  last_verified_at timestamptz,
  review_after timestamptz,
  metadata jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create index if not exists memory_facts_user_status_idx
  on public.memory_facts (user_id, status, confidence desc, created_at desc);
create index if not exists memory_facts_subject_idx
  on public.memory_facts (user_id, subject);
create index if not exists memory_facts_review_after_idx
  on public.memory_facts (review_after) where review_after is not null;

alter table public.memory_facts enable row level security;

create policy "Users can read their own memory facts"
  on public.memory_facts for select to authenticated
  using (auth.uid() = user_id);
create policy "Users can insert their own memory facts"
  on public.memory_facts for insert to authenticated
  with check (auth.uid() = user_id);
create policy "Users can update their own memory facts"
  on public.memory_facts for update to authenticated
  using (auth.uid() = user_id) with check (auth.uid() = user_id);
create policy "Users can delete their own memory facts"
  on public.memory_facts for delete to authenticated
  using (auth.uid() = user_id);

comment on table public.memory_facts is
  'Layer 3 structured memory with freshness, confidence, sensitivity, and superseding support.';
