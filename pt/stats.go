package pt

import (
	"sync/atomic"
	"time"
)

// counter is a lock-free monotonically non-decreasing 64-bit counter.
// All of its methods are safe to call concurrently from multiple workers
// and from readers taking snapshots while a render is in progress.
type counter struct {
	v uint64
}

func (c *counter) add(n uint64) uint64 {
	return atomic.AddUint64(&c.v, n)
}

func (c *counter) inc() uint64 {
	return atomic.AddUint64(&c.v, 1)
}

func (c *counter) load() uint64 {
	return atomic.LoadUint64(&c.v)
}

// reset is only safe to call when no render workers are running (i.e.
// before a render starts or after all workers have been joined).
func (c *counter) reset() {
	atomic.StoreUint64(&c.v, 0)
}

// BufferStats is a read-only, point-in-time snapshot of a Buffer.
//
// Every field is a value type: a snapshot never aliases the Buffer's
// internal Pixel slice, so mutating the Buffer afterwards cannot change
// a snapshot that has already been taken.
type BufferStats struct {
	// Width and height of the buffer in pixels. Either may be zero for
	// a buffer that was constructed with a zero dimension.
	Width  int
	Height int
	// TotalSamples is the sum of the per-pixel sample counts. It is
	// accumulated with atomic operations and therefore never goes
	// backwards between two snapshots of the same render.
	TotalSamples uint64
	// NonEmptyPixels is the number of pixels that have received at
	// least one sample.
	NonEmptyPixels uint64
	// MinSamples and MaxSamples bound the per-pixel sample counts.
	// When the buffer has zero pixels both are zero.
	MinSamples int
	MaxSamples int
	// MeanSamplesPerPixel is TotalSamples divided by the number of
	// pixels. It is zero for an empty (zero-area) buffer.
	MeanSamplesPerPixel float64
}

// SceneStats is a read-only, point-in-time snapshot of a Scene's
// ray-accounting counters. All fields are value types.
type SceneStats struct {
	// Rays is the total number of calls to Scene.Intersect since the
	// counter was last reset: one increment per intersection test,
	// covering both primary and secondary (shadow / bounce) rays.
	Rays uint64
	// PrimaryRays counts camera-ray intersection tests.
	PrimaryRays uint64
	// SecondaryRays counts bounce and shadow-ray intersection tests.
	// PrimaryRays + SecondaryRays == Rays for every snapshot.
	SecondaryRays uint64
}

// RenderStats is a read-only, point-in-time snapshot of a Renderer.
// All fields are value types; the snapshot owns no mutable state.
type RenderStats struct {
	// Iterations run by this renderer so far. For Renderer.Run /
	// Renderer.Render each invocation completes exactly one iteration;
	// IterativeRender / ChannelRender accumulate across iterations.
	Iterations uint64
	// CompletedRows is the number of scanlines finished by the workers
	// of the currently running (or most recently completed) iteration.
	// It is bounded by [0, Height].
	CompletedRows uint64
	// Workers is the number of parallel worker goroutines used per
	// iteration (clamped to at least one and to the buffer height).
	Workers int
	// Width and height of the renderer's target buffer.
	Width  int
	Height int
	// PrimaryRays counts camera rays cast by the renderer (one per
	// accepted per-pixel sample). Rays that are intentionally dropped
	// via the DropSamples fault-injection hook are not counted.
	PrimaryRays uint64
	// SecondaryRays counts intersection tests that are not primary
	// camera rays, i.e. bounce and shadow rays. It is derived from the
	// scene-wide intersection counter and therefore never negative.
	SecondaryRays uint64
	// TotalRays is PrimaryRays + SecondaryRays when the renderer owns
	// the scene counter exclusively; otherwise it equals the scene
	// counter. It is always >= PrimaryRays.
	TotalRays uint64
	// TotalSamples is the number of samples actually written into the
	// buffer (accepted samples). It can be lower than PrimaryRays when
	// DropSamples is configured to lose samples.
	TotalSamples uint64
	// MeanSamplesPerPixel is TotalSamples divided by the number of
	// pixels; zero for a zero-area buffer.
	MeanSamplesPerPixel float64
	// Elapsed is wall-clock time spent inside completed render
	// iterations at the time of the snapshot. It is monotonic.
	Elapsed time.Duration
	// Running reports whether an iteration is currently executing.
	Running bool
}

// clampNonNeg guards every derived numeric value exposed in snapshots:
// counters are unsigned by construction, but defensively clamping any
// signed subtraction result guarantees no snapshot field can ever go
// negative even if a future refactor changes the counting layout.
func clampNonNeg(x int64) uint64 {
	if x < 0 {
		return 0
	}
	return uint64(x)
}
