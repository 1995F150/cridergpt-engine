create extension if not exists vector;

create table if not exists public.rag_sources (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users(id) on delete cascade,
  title text not null,
  source_type text not null default 'document',
  source_uri text,
  metadata jsonb not null default '{}'::jsonb,
  content_hash text,
  is_active boolean not null default true,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create table if not exists public.rag_chunks (
  id uuid primary key default gen_random_uuid(),
  source_id uuid not null references public.rag_sources(id) on delete cascade,
  user_id uuid not null references auth.users(id) on delete cascade,
  chunk_index integer not null,
  content text not null,
  token_estimate integer,
  metadata jsonb not null default '{}'::jsonb,
  embedding vector(768) not null,
  created_at timestamptz not null default now(),
  unique(source_id, chunk_index)
);

create index if not exists rag_sources_user_idx on public.rag_sources(user_id, is_active);
create index if not exists rag_chunks_user_idx on public.rag_chunks(user_id);
create index if not exists rag_chunks_embedding_idx on public.rag_chunks
using ivfflat (embedding vector_cosine_ops) with (lists = 100);

alter table public.rag_sources enable row level security;
alter table public.rag_chunks enable row level security;

create policy "Users manage their RAG sources"
on public.rag_sources for all
using (auth.uid() = user_id)
with check (auth.uid() = user_id);

create policy "Users manage their RAG chunks"
on public.rag_chunks for all
using (auth.uid() = user_id)
with check (auth.uid() = user_id);

create or replace function public.match_rag_chunks(
  query_embedding vector(768),
  match_user_id uuid,
  match_count integer default 6,
  match_threshold double precision default 0.55
)
returns table (
  source_id uuid,
  title text,
  content text,
  source_type text,
  source_uri text,
  similarity double precision
)
language sql stable security definer
set search_path = public
as $$
  select
    c.source_id,
    s.title,
    c.content,
    s.source_type,
    s.source_uri,
    1 - (c.embedding <=> query_embedding) as similarity
  from public.rag_chunks c
  join public.rag_sources s on s.id = c.source_id
  where c.user_id = match_user_id
    and s.user_id = match_user_id
    and s.is_active = true
    and 1 - (c.embedding <=> query_embedding) >= match_threshold
  order by c.embedding <=> query_embedding
  limit greatest(1, least(match_count, 20));
$$;

revoke all on function public.match_rag_chunks(vector, uuid, integer, double precision) from public;
grant execute on function public.match_rag_chunks(vector, uuid, integer, double precision) to authenticated, service_role;
