create table if not exists public.video_generation_jobs (
  id uuid primary key,
  user_id uuid not null references auth.users(id) on delete cascade,
  provider text not null default 'local',
  provider_job_id text not null unique,
  status text not null default 'queued',
  progress integer not null default 0 check (progress between 0 and 100),
  prompt text not null,
  negative_prompt text,
  model text,
  duration_seconds integer,
  aspect_ratio text,
  reference_image_url text,
  output_url text,
  preview_url text,
  error_message text,
  provider_payload jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

alter table public.video_generation_jobs add column if not exists progress integer not null default 0;

create index if not exists video_generation_jobs_user_created_idx
  on public.video_generation_jobs (user_id, created_at desc);
create index if not exists video_generation_jobs_status_idx
  on public.video_generation_jobs (status);
create index if not exists video_generation_jobs_local_queue_idx
  on public.video_generation_jobs (created_at)
  where provider = 'local' and status = 'queued';

alter table public.video_generation_jobs enable row level security;

drop policy if exists "Users can view their video jobs" on public.video_generation_jobs;
create policy "Users can view their video jobs"
  on public.video_generation_jobs for select
  using (auth.uid() = user_id);

drop policy if exists "Users can create their video jobs" on public.video_generation_jobs;
create policy "Users can create their video jobs"
  on public.video_generation_jobs for insert
  with check (auth.uid() = user_id);

drop policy if exists "Users can update their video jobs" on public.video_generation_jobs;
create policy "Users can update their video jobs"
  on public.video_generation_jobs for update
  using (auth.uid() = user_id)
  with check (auth.uid() = user_id);

create or replace function public.set_video_job_updated_at()
returns trigger language plpgsql as $$
begin
  new.updated_at = now();
  return new;
end;
$$;

drop trigger if exists set_video_job_updated_at on public.video_generation_jobs;
create trigger set_video_job_updated_at
before update on public.video_generation_jobs
for each row execute function public.set_video_job_updated_at();
