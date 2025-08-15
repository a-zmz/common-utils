import spikeinterface as si

# set si job_kwargs
job_kwargs = dict(
    #mp_context="fork", # linux, but does not work still on 2025 aug 12
    mp_context="spawn", # mac & win
    chunk_duration='1s',
    progress_bar=True,
    n_jobs=0.8,
    max_threads_per_worker=1,
)
si.set_global_job_kwargs(**job_kwargs)
