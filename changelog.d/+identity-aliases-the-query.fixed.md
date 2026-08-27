`Bridge.to_index_space` no longer rewrites the caller's query vector when the adapter is `identity`.

`to_index_space` normalises its result in place, deliberately: it is one allocation off a path budgeted at 15 µs. That is safe only while every adapter returns a new array, which every adapter that multiplies does for free. `IdentityAdapter` has nothing to multiply by — it handed back the input unchanged whenever the widths already matched, so the normalisation landed on the caller's own array.

Measured: `bridge.to_index_space(q)` left `q` normalised. A caller reusing `q` for a second index, for a rerank, or for a log line was working with a different vector from the one they encoded, and nothing raised. Two paths reach it — `fit --method identity` and loading a `.rbs` that records that type — and `auto` never selects that adapter, which is how it survived.

`BaseAdapter.apply` now states the contract the other implementations were already keeping, and a property test asserts it across identity, Procrustes, centred Procrustes and linear: `apply` must not return the caller's array or a view of it, and a full serving call must leave the input unchanged.

Found while writing down whether `Bridge` is safe to share between threads. It is — it is immutable after `load`, holds no per-call state, and needs no lock — and that is now in its docstring, along with why there is no `async` variant.
