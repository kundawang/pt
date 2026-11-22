package pt

import (
	"fmt"
	"image"
	"math"
	"math/rand"
	"runtime"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

// DropSamplesFunc is a fault-injection hook invoked for every camera
// ray that the renderer is about to trace. Returning true makes the
// renderer discard that sample: the ray is never traced and nothing is
// written into the buffer, while every accepted sample is still
// accounted for. A nil hook (the default) never drops anything.
//
// It is intended for tests that prove the observability layer detects
// lost samples; it must never be used in a production render.
type DropSamplesFunc func(iteration, x, y, sampleIndex int) bool

// Renderer renders a Scene through a Camera with a Sampler into a
// Buffer, optionally across many iterations, and exposes race-free
// rendering statistics through Renderer.Stats.
//
// A Renderer must not be shared by two concurrent renders: the internal
// progress counters and the scene counter reset assume one render at a
// time. Multiple Renderers over independent scenes/buffers are safe to
// run in parallel.
type Renderer struct {
	Scene   *Scene
	Camera  *Camera
	Sampler Sampler
	Buffer  *Buffer

	// SamplesPerPixel is the number of camera rays per pixel per
	// iteration. Positive values select stratified sampling (rounded
	// down to a perfect square, matching the historical renderer),
	// negative values select |SamplesPerPixel| random samples.
	SamplesPerPixel int

	// NumCPU is the number of worker goroutines. Zero means
	// runtime.NumCPU().
	NumCPU int

	// Verbose keeps the historical progress-bar log output. It
	// defaults to true through NewRenderer.
	Verbose bool

	// DropSamples, when non-nil, intentionally discards selected
	// camera rays. See DropSamplesFunc.
	DropSamples DropSamplesFunc

	// Iteration cumulative boundary (all atomically maintained):
	iterations    uint64
	completedRows uint64
	primaryRays   uint64
	totalSamples  uint64
	elapsedNS     uint64
	running       uint64 // 0 = idle, 1 = an iteration is in progress

	// startNS records the beginning of the currently running
	// iteration so Stats can report live elapsed time.
	startNS int64

	mu sync.Mutex
}

// NewRenderer constructs a Renderer with a fresh w-by-h buffer and the
// same defaults as the historical package-level render helpers.
func NewRenderer(scene *Scene, camera *Camera, sampler Sampler, w, h int) *Renderer {
	return &Renderer{
		Scene:           scene,
		Camera:          camera,
		Sampler:         sampler,
		Buffer:          NewBuffer(w, h),
		SamplesPerPixel: 1,
		NumCPU:          runtime.NumCPU(),
		Verbose:         true,
	}
}

func (r *Renderer) workers() int {
	n := r.NumCPU
	if n <= 0 {
		n = runtime.NumCPU()
	}
	if n < 1 {
		n = 1
	}
	h := r.Buffer.H
	if h > 0 && n > h {
		// No point in more workers than scanlines.
		n = h
	}
	if n < 1 {
		n = 1
	}
	return n
}

func (r *Renderer) printf(format string, a ...interface{}) {
	if r.Verbose {
		fmt.Printf(format, a...)
	}
}

func (r *Renderer) showProgress(start time.Time, rays uint64, i, h int) {
	if !r.Verbose {
		return
	}
	pct := 0
	if h > 0 {
		pct = int(100 * float64(i) / float64(h))
	}
	elapsed := time.Since(start)
	rps := 0.0
	if seconds := elapsed.Seconds(); seconds > 0 {
		rps = float64(rays) / seconds
	}
	fmt.Printf("\r%4d / %d (%3d%%) [", i, h, pct)
	for p := 0; p < 100; p += 3 {
		if pct > p {
			fmt.Print("=")
		} else {
			fmt.Print(" ")
		}
	}
	fmt.Printf("] %s %s ", DurationString(elapsed), NumberString(rps))
}

// Run performs one render iteration into Buffer. It is the shared
// implementation behind Render, IterativeRender and ChannelRender. The
// scene ray counters are reset at its start, while the renderer-level
// cumulative counters are not.
func (r *Renderer) Run() {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.runIteration()
}

func (r *Renderer) runIteration() {
	scene := r.Scene
	camera := r.Camera
	sampler := r.Sampler
	buf := r.Buffer
	w, h := buf.W, buf.H
	ncpu := r.workers()

	scene.Compile()

	// Per-iteration boundary: reset scene and progress counters before
	// any worker starts. Renderer-level cumulative counters survive.
	scene.ResetRays()
	atomic.StoreUint64(&r.completedRows, 0)
	atomic.StoreUint64(&r.running, 1)
	start := time.Now()
	atomic.StoreInt64(&r.startNS, start.UnixNano())

	absSamples := int(math.Abs(float64(r.SamplesPerPixel)))
	stratified := r.SamplesPerPixel > 0
	stratRoot := int(math.Sqrt(float64(r.SamplesPerPixel)))
	if stratRoot < 1 {
		stratRoot = 1
	}
	iteration := int(atomic.LoadUint64(&r.iterations)) + 1
	ch := make(chan int, h)
	r.printf("%d x %d pixels, %d spp, %d cores\n", w, h, absSamples, ncpu)

	if w == 0 || h == 0 {
		// Degenerate buffer: there is no work for any worker. Close the
		// iteration boundary immediately with zero rows and rays.
		elapsed := time.Since(start)
		atomic.AddUint64(&r.elapsedNS, uint64(elapsed.Nanoseconds()))
		atomic.StoreUint64(&r.running, 0)
		atomic.AddUint64(&r.iterations, 1)
		return
	}

	for i := 0; i < ncpu; i++ {
		go func(worker int) {
			rnd := rand.New(rand.NewSource(time.Now().UnixNano() + int64(worker)))
			for y := worker; y < h; y += ncpu {
				for x := 0; x < w; x++ {
					if stratified {
						// stratified subsampling
						for u := 0; u < stratRoot; u++ {
							for v := 0; v < stratRoot; v++ {
								fu := (float64(u) + 0.5) / float64(stratRoot)
								fv := (float64(v) + 0.5) / float64(stratRoot)
								r.traceOne(scene, camera, sampler, buf, x, y, w, h, fu, fv, iteration, rnd)
							}
						}
					} else {
						// random subsampling
						for s := 0; s < absSamples; s++ {
							fu := rnd.Float64()
							fv := rnd.Float64()
							r.traceOne(scene, camera, sampler, buf, x, y, w, h, fu, fv, iteration, rnd)
						}
					}
					// Historical adaptive pass: pixels whose sample
					// standard deviation is high receive up to 128
					// extra random samples.
					v := buf.Variance(x, y)
					if v.R < 0.3 && v.G < 0.3 && v.B < 0.3 {
						continue
					}
					for s := 0; s < 128; s++ {
						fu := rnd.Float64()
						fv := rnd.Float64()
						r.traceOne(scene, camera, sampler, buf, x, y, w, h, fu, fv, iteration, rnd)
					}
				}
				atomic.AddUint64(&r.completedRows, 1)
				ch <- 1
			}
		}(i)
	}

	r.showProgress(start, scene.RayCount(), 0, h)
	for i := 0; i < h; i++ {
		<-ch
		r.showProgress(start, scene.RayCount(), i+1, h)
	}
	r.printf("\n")

	// Close the iteration boundary only after every worker has joined,
	// so a snapshot can never observe partial final state.
	elapsed := time.Since(start)
	atomic.AddUint64(&r.elapsedNS, uint64(elapsed.Nanoseconds()))
	atomic.StoreUint64(&r.running, 0)
	atomic.AddUint64(&r.iterations, 1)
}

// traceOne casts one camera ray and records the sample, honoring the
// DropSamples fault-injection hook. All failures are contained: a
// dropped or panicking sample must not lose pixels for its neighbors.
func (r *Renderer) traceOne(scene *Scene, camera *Camera, sampler Sampler, buf *Buffer,
	x, y, w, h int, fu, fv float64, iteration int, rnd *rand.Rand) {
	sampleIndex := int(atomic.AddUint64(&r.primaryRays, 1)) - 1
	if r.DropSamples != nil && r.DropSamples(iteration, x, y, sampleIndex) {
		// Intentionally lost sample: no ray, no buffer write. The
		// attempted primary ray was counted above so the discrepancy
		// remains observable.
		return
	}
	// Recover from an unexpected sampler panic: the render must keep
	// producing pixels even if a single sample blows up. A failed
	// sample is not written to the buffer and not counted as an
	// accepted sample, so its absence stays visible through the
	// sample-count statistics rather than masquerading as black.
	var sample Color
	var ok bool
	func() {
		defer func() {
			if rec := recover(); rec != nil {
				ok = false
			}
		}()
		ray := camera.CastRay(x, y, w, h, fu, fv, rnd)
		sample = sampler.Sample(scene, ray, rnd)
		ok = true
	}()
	if !ok {
		return
	}
	buf.AddSample(x, y, sample)
	atomic.AddUint64(&r.totalSamples, 1)
}

// Render runs one iteration and returns the buffer image.
func (r *Renderer) Render() image.Image {
	r.Run()
	return r.Buffer.Image()
}

// writeImage encodes one snapshot image; a failure is reported through
// the WaitGroup rather than killing unrelated iterations.
func (r *Renderer) writeImage(path string, im image.Image, wg *sync.WaitGroup) {
	defer wg.Done()
	if err := SavePNG(path, im); err != nil {
		if r.Verbose {
			fmt.Printf("warning: failed to write %s: %v\n", path, err)
		}
	}
}

// IterativeRender runs the given number of iterations, writing a PNG
// after each one, and returns the final image. Cumulative renderer
// statistics cover every iteration; the per-iteration scene ray counter
// is reset at the start of each iteration.
func (r *Renderer) IterativeRender(pathTemplate string, iterations int) image.Image {
	var wg sync.WaitGroup
	if iterations < 0 {
		iterations = 0
	}
	for i := 1; i <= iterations; i++ {
		r.mu.Lock()
		r.printf("\n[Iteration %d of %d]\n", i, iterations)
		r.runIteration()
		r.mu.Unlock()

		path := pathTemplate
		if strings.Contains(path, "%") {
			path = fmt.Sprintf(pathTemplate, i)
		}
		im := r.Buffer.Image()
		wg.Add(1)
		go r.writeImage(path, im, &wg)
	}
	wg.Wait()
	return r.Buffer.Image()
}

// ChannelRender renders forever, sending a fresh image after every
// iteration. The channel is unbuffered and closed when the caller
// cancels via the returned stop function; statistics accumulate across
// iterations exactly as in IterativeRender.
func (r *Renderer) ChannelRender() (<-chan image.Image, func()) {
	ch := make(chan image.Image)
	done := make(chan struct{})
	var stopOnce sync.Once
	stop := func() {
		stopOnce.Do(func() { close(done) })
	}
	go func() {
		defer close(ch)
		for {
			r.mu.Lock()
			r.runIteration()
			r.mu.Unlock()
			im := r.Buffer.Image()
			select {
			case <-done:
				return
			case ch <- im:
			}
		}
	}()
	return ch, stop
}

// Stats returns a read-only, point-in-time snapshot of the renderer.
// The returned value contains scalar fields only and is safe to retain
// and inspect while a render is in progress.
func (r *Renderer) Stats() RenderStats {
	buf := r.Buffer
	bstats := buf.Stats()
	sceneStats := r.Scene.Stats()

	elapsed := time.Duration(atomic.LoadUint64(&r.elapsedNS))
	running := atomic.LoadUint64(&r.running) == 1
	if running {
		// Include the live iteration's elapsed wall time.
		startNS := atomic.LoadInt64(&r.startNS)
		elapsed += time.Duration(time.Now().UnixNano() - startNS)
	}
	primary := atomic.LoadUint64(&r.primaryRays)
	samples := atomic.LoadUint64(&r.totalSamples)

	// The scene counter is reset per iteration, so its secondary count
	// describes the current iteration; the renderer-level cumulative
	// primary count is the authority for primary rays. Clamp every
	// derived value so the snapshot can never report negatives.
	secondary := sceneStats.SecondaryRays
	total := clampNonNeg(int64(primary) + int64(secondary))
	if samples > primary {
		// Accepted samples can never exceed attempted camera rays.
		samples = primary
	}

	return RenderStats{
		Iterations:          atomic.LoadUint64(&r.iterations),
		CompletedRows:       atomic.LoadUint64(&r.completedRows),
		Workers:             r.workers(),
		Width:               buf.W,
		Height:              buf.H,
		PrimaryRays:         primary,
		SecondaryRays:       secondary,
		TotalRays:           total,
		TotalSamples:        samples,
		MeanSamplesPerPixel: bstats.MeanSamplesPerPixel,
		Elapsed:             elapsed,
		Running:             running,
	}
}
