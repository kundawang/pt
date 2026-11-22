# pt/safety — geometry input quality assurance

Hardens the untrusted geometry input path of
[`github.com/fogleman/pt`](https://github.com/fogleman/pt) against malicious or
malformed **OBJ**, **STL** and **SDF** input. The upstream loaders trust their
input: out-of-range face indices panic, truncated vertex lines panic, NaN/Inf
coordinates are accepted silently, a forged STL count drives a huge
pre-allocation, and `SDFShape.Intersect` has no bound on a non-returning march.

This package adds an independent validation layer that never shares code with
`pt.LoadOBJ`, `pt.LoadSTL` or `pt.SDFShape`.

## Threat model

| Attack | Upstream behavior | Safety outcome |
| --- | --- | --- |
| OBJ face index out of range | `index out of range` panic | reject at `StageIndex` before the loader runs |
| Truncated `v` / 2-vertex `f` line | panic / silently empty mesh | reject at `StageHeader` |
| `nan`, `inf`, `1e999` coordinates | silently accepted, poison k-d tree | reject at `StageValue` |
| gigantic finite vertex | bbox / memory amplification | reject at `StageMagnitude` |
| forged binary STL triangle count | allocates a slice sized by the forged count | reject at `StageHeader` (framing mismatch), no big allocation |
| dangling ASCII STL facet | silently mis-triangulates | reject at `StageHeader` |
| zero-area / collinear triangle | degenerate node in k-d tree | warning at text layer, reject at post-load `StageDegenerate` if all triangles are bad |
| SDF returning NaN / never converging / panicking | march stalls up to a hardcoded cap or crashes a worker | bounded marcher returns `NoHit`; panic recovered |
| deeply nested composite SDF | unbounded recursion per `Evaluate` | `GuardTree` counts depth; over-nested trees rejected at `StageDepth` |
| wedged loader | hangs forever | per-sample deadline → `StageTimeout` |

## Pipeline

`SafeLoadOBJ` / `SafeLoadSTL` run three independent stages:

1. **Reference check** — `CheckOBJ` / `CheckSTL` parse the raw bytes
   themselves (no call into the renderer loader) and enforce file size,
   vertex/triangle counts, index ranges, finiteness, coordinate magnitude and
   framing. The returned `Report` records the first failing `Stage` and the
   resource counts observed up to that point.
2. **Guarded load** — the real `pt.LoadOBJ` / `pt.LoadSTL` runs inside a
   panic-to-error boundary and a wall-clock deadline.
3. **Post-load mesh check** — `CheckMesh` independently verifies the produced
   `*pt.Mesh`: finiteness, magnitude, triangle count and geometric area.

A sample is either rejected with a structured `*RejectionError` (which unwraps
to `ErrRejected`) or yields a fully validated mesh. A partial mesh is never
returned.

### SDF path

`GuardTree(sdf, limit)` rewrites a composite SDF tree so every node shares one
depth budget and normalizes NaN distances to `+Inf`. `NewBoundedSDFShape`
wraps it in an independent ray marcher bounded by iteration count, a deadline
and finite per-step advance, implementing `pt.Shape` so it drops straight into
a `pt.Scene`. `CheckSDFDepth` rejects over-nested trees structurally before any
ray is cast.

## Limits

`DefaultLimits()`: 16 MiB file, 1M vertices, 1M triangles, SDF depth 256,
coordinate magnitude 1e6, 5 s per sample. Any field can be tightened; zero
fields fall back to the default.

## Tests

```
go test ./pt/safety/ -race -v
```

* **Fixed-seed fuzz/property tests** (`fuzz_test.go`): deterministic OBJ/STL
  corpora (seed `0x5AFE1234`) covering index faults, negative numbers,
  NaN/Inf, huge vertices, degenerate triangles, bad STL headers and
  count-amplification. Invariants: no panic, every result within the resource
  and time budgets, structured stage on rejection.
* **Curated regression samples** (`testdata/`): three checked-in,
  reproducible samples — a valid cube, an out-of-range-index OBJ, and a
  dangling-facet STL.
* **Fault injection** (`faultinject_test.go`): each defect class is injected
  deliberately and asserted to be caught at its documented stage, including
  the upstream loader panic and the non-returning SDF variants.
* **Pixel stability** (`pixel_test.go`): the validated cube renders through
  the raw and hardened load paths with identical hits, normals and materials —
  hardening does not move a single pixel of a well-formed scene.
