import tqdm


def make_overlap_chunks(
    n,
    chunk_size,
    overlap=1,
):
    assert 0 <= overlap < chunk_size
    step = chunk_size - overlap
    chunks = []
    start = 0
    while start < n:
        end = min(start + chunk_size, n)
        chunks.append(list(range(start, end)))
        if end == n:
            break
        start += step
    return chunks


def maybe_tqdm(iterable, use_tqdm=True, **kwargs):
    if use_tqdm:
        import tqdm

        return tqdm.tqdm(iterable, **kwargs)
    else:
        return iterable
