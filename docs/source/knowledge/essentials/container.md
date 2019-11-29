# The Container abstraction

All analysis in `scared` start with a container. A `Container` is an abstraction which is responsible to provide traces and metadata to leakage or analysis objects.

Basically, a container is defined by:

- a `TraceHeaderSet` instance
- a `frame` optional
- a list of `preprocesses`

Once initialized, it can provides batches of traces samples and metadatas on-demand, by applying the preprocess chain just-in-time.

A container is created like this:

```python
import scared

ths = scared.traces.read_ths_from_ets_file('traceset.ets')
container = scared.Container(
    ths,
    frame=slice(20, 30),
    preprocesses=[scared.preprocesses.standardize]
)
```
