create table if not exists public.core_profiles (
  user_id uuid primary key references auth.users(id) on delete cascade,
  display_name text,
  location text,
  bio text,
  stable_facts jsonb not null default '[]'::jsonb,
  preferences jsonb not null default '{}'::jsonb,
  goals jsonb not null default '[]'::jsonb,
  communication jsonb not null default '{}'::jsonb,
  projects jsonb not null default '[]'::jsonb,
  privacy jsonb not null default '{}'::jsonb,
  version integer not null default 1 check (version > 0),
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create index if not exists core_profiles_updated_at_idx
  on public.core_profiles (updated_at desc);

alter table public.core_profiles enable row level security;

create policy "Users can read their own core profile"
  on public.core_profiles
  for select
  to authenticated
  using (auth.uid() = user_id);

create policy "Users can insert their own core profile"
  on public.core_profiles
  for insert
  to authenticated
  with check (auth.uid() = user_id);

create policy "Users can update their own core profile"
  on public.core_profiles
  for update
  to authenticated
  using (auth.uid() = user_id)
  with check (auth.uid() = user_id);

create policy "Users can delete their own core profile"
  on public.core_profiles
  for delete
  to authenticated
  using (auth.uid() = user_id);

comment on table public.core_profiles is
  'Layer 1 durable user profile used by the CriderGPT engine.';
