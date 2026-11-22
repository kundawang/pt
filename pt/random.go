package pt

// Random is the minimal source of randomness used by the renderer. Both
// *rand.Rand and *SobolRand implement it, so the existing random sampler and
// the deterministic low-discrepancy sampler share the same code paths.
//
// A Random must not be shared between goroutines. Render creates one per
// worker (random mode) or one per pixel/sample (deterministic mode).
type Random interface {
	// Float64 returns a value in [0, 1).
	Float64() float64
	// Intn returns a value in [0, n). Implementations define n <= 0.
	Intn(n int) int
}
