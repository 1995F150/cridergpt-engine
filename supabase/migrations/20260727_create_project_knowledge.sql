create table if not exists public.project_knowledge (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users(id) on delete cascade,
  project_key text not null,
  project_name text not null,
  category text not null default 'general',
  title text not null,
  content text not null,
  status text not null default 'active'
    check (status in ('active', 'planned', 'paused', 'completed', 'historical', 'superseded')),
  priority integer not null default 50 check (priority between 0 and 100),
  source text,
  metadata jsonb not null default '{}'::jsonb,
  is_active boolean not null default true,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create index if not exists project_knowledge_user_project_idx
  on public.project_knowledge (user_id, project_key, is_active);
create index if not exists project_knowledge_user_priority_idx
  on public.project_knowledge (user_id, priority desc, updated_at desc);

alter table public.project_knowledge enable row level security;

create policy "Users can read their own project knowledge"
  on public.project_knowledge for select to authenticated
  using (auth.uid() = user_id);
create policy "Users can insert their own project knowledge"
  on public.project_knowledge for insert to authenticated
  with check (auth.uid() = user_id);
create policy "Users can update their own project knowledge"
  on public.project_knowledge for update to authenticated
  using (auth.uid() = user_id) with check (auth.uid() = user_id);
create policy "Users can delete their own project knowledge"
  on public.project_knowledge for delete to authenticated
  using (auth.uid() = user_id);

comment on table public.project_knowledge is
  'Layer 2 durable project facts and documentation for user-scoped retrieval.';
