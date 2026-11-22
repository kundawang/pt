package pt

import (
	"fmt"
	"image"
	"math"
	"math/rand"
	"runtime"
	"strings"
	"sync"
	"time"
)

// SampleMode selects how the renderer obtains random numbers.
type SampleMode int

const (
	// SampleModeRandom is the historical behavior: each worker owns a
	// math/rand source seeded from the wall clock. Output changes from run to
	// run and across worker counts. This is the zero value, so callers that
	// do not opt in keep the old behavior.
	SampleModeRandom SampleMode = iota
	// SampleModeSobol draws every sample from a scrambled, low-discrepancy
	// Sobol sequence keyed by (seed, iteration, absolute pixel index, sample
	// ordinal, dimension). Output is bit-identical for the same config
	// regardless of worker count, scheduling or tiling.
	SampleModeSobol
)

func (m SampleMode) String() string {
	switch m {
	case SampleModeSobol:
		return "sobol"
	default:
		return "random"
	}
}

// RenderConfig controls scheduling and sampling for a render. The zero value
// is the fully backward-compatible configuration: random sampling, one worker
// per CPU, no tiling.
type RenderConfig struct {
	Mode    SampleMode // random (default) or deterministic Sobol
	Seed    uint64     // Sobol seed; ignored in random mode
	Workers int        // goroutines; <= 0 means runtime.NumCPU()
	Tiles   int        // spatial tiles per pass; <= 1 means one row at a time
	// DisableAdaptive turns off the variance-driven extra samples that are
	// otherwise added per pixel. It defaults to false so behavior is
	// unchanged; tests and fixed-workload renders enable it to compare
	// estimators at exactly samplesPerPixel primary samples.
	DisableAdaptive bool
	// Region, when non-nil, restricts each pass to a sub-rectangle of the
	// buffer. It is how a render farm splits one image across machines: every
	// client renders its region into a full-size buffer and the regions are
	// then merged. In Sobol mode region boundaries cannot change any pixel's
	// value, so a merged region render is bit-identical to rendering the
	// whole image on one machine.
	Region *image.Rectangle
	// Quiet suppresses the per-pass banner (but not the progress line).
	Quiet bool
}

// DefaultRenderConfig returns the historical render configuration.
func DefaultRenderConfig() RenderConfig {
	return RenderConfig{}
}

func (c RenderConfig) workers() int {
	n := c.Workers
	if n <= 0 {
		n = runtime.NumCPU()
	}
	if n < 1 {
		n = 1
	}
	return n
}

func (c RenderConfig) tiles() int {
	if c.Tiles < 1 {
		return 1
	}
	return c.Tiles
}

type rowRect struct{ x0, y0, x1, y1 int }

func splitRows(w, h int) []rowRect {
	result := make([]rowRect, h)
	for y := 0; y < h; y++ {
		result[y] = rowRect{0, y, w, y + 1}
	}
	return result
}

// splitTiles partitions the image into approximately count rectangles. Tile
// boundaries do not affect output in Sobol mode because sample values key on
// absolute pixel coordinates.
func splitTiles(w, h, count int) []rowRect {
	if count <= 1 {
		return []rowRect{{0, 0, w, h}}
	}
	targetX := int(math.Round(math.Sqrt(float64(count) * float64(w) / float64(h))))
	if targetX < 1 {
		targetX = 1
	}
	if targetX > count {
		targetX = count
	}
	targetY := (count + targetX - 1) / targetX
	if targetX > w {
		targetX = w
	}
	if targetY > h {
		targetY = h
	}
	result := make([]rowRect, 0, targetX*targetY)
	for ty := 0; ty < targetY; ty++ {
		y0 := ty * h / targetY
		y1 := (ty + 1) * h / targetY
		for tx := 0; tx < targetX; tx++ {
			x0 := tx * w / targetX
			x1 := (tx + 1) * w / targetX
			if x1 > x0 && y1 > y0 {
				result = append(result, rowRect{x0, y0, x1, y1})
			}
		}
	}
	return result
}

// clipRect clamps r to the image bounds [0,w)x[0,h). An empty or inverted
// rectangle yields an empty result rather than a panic.
func clipRect(r *image.Rectangle, w, h int) image.Rectangle {
	bounds := image.Rect(0, 0, w, h)
	return r.Intersect(bounds)
}

// intersectRect returns the overlap of a row rect and an axis-aligned
// rectangle, and whether the overlap is non-empty.
func intersectRect(r rowRect, clip image.Rectangle) (rowRect, bool) {
	x0, y0 := r.x0, r.y0
	x1, y1 := r.x1, r.y1
	if x0 < clip.Min.X {
		x0 = clip.Min.X
	}
	if y0 < clip.Min.Y {
		y0 = clip.Min.Y
	}
	if x1 > clip.Max.X {
		x1 = clip.Max.X
	}
	if y1 > clip.Max.Y {
		y1 = clip.Max.Y
	}
	if x1 <= x0 || y1 <= y0 {
		return rowRect{}, false
	}
	return rowRect{x0, y0, x1, y1}, true
}

func showProgress(start time.Time, rays uint64, i, h int) {
	if h <= 0 {
		return
	}
	pct := int(100 * float64(i) / float64(h))
	elapsed := time.Since(start)
	rps := float64(rays) / elapsed.Seconds()
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

// renderPass renders one full image pass into buf. In Sobol mode the sample
// ordinals used within a pass are disjoint from every other iteration
// because the iteration's seed is derived with SobolSeedForIteration.
func renderPass(scene *Scene, camera *Camera, sampler Sampler, samplesPerPixel int, buf *Buffer, cfg RenderConfig, iteration int) {
	w, h := buf.W, buf.H
	if w <= 0 || h <= 0 {
		return
	}
	nworkers := cfg.workers()
	scene.Compile()
	absSamples := int(math.Abs(float64(samplesPerPixel)))
	seed := cfg.Seed
	if cfg.Mode == SampleModeSobol {
		seed = SobolSeedForIteration(cfg.Seed, iteration)
	}
	if !cfg.Quiet {
		fmt.Printf("%d x %d pixels, %d spp, %d cores, %s sampling", w, h, absSamples, nworkers, cfg.Mode)
		if cfg.Mode == SampleModeSobol {
			fmt.Printf(" (seed=%d, iteration=%d)", cfg.Seed, iteration)
		}
		fmt.Println()
		if cfg.Region != nil {
			r := clipRect(cfg.Region, w, h)
			fmt.Printf("region: %d,%d - %d,%d\n", r.Min.X, r.Min.Y, r.Max.X, r.Max.Y)
		} else if cfg.tiles() > 1 {
			fmt.Printf("tiling: %d tiles\n", cfg.tiles())
		}
	}
	start := time.Now()
	scene.rays = 0

	rects := splitRows(w, h)
	if cfg.tiles() > 1 {
		rects = splitTiles(w, h, cfg.tiles())
	}
	if cfg.Region != nil {
		region := clipRect(cfg.Region, w, h)
		var clipped []rowRect
		for _, r := range rects {
			if rr, ok := intersectRect(r, region); ok {
				clipped = append(clipped, rr)
			}
		}
		rects = clipped
	}
	if len(rects) == 0 {
		if !cfg.Quiet {
			fmt.Println()
		}
		return
	}
	jobs := make(chan rowRect, len(rects))
	for _, r := range rects {
		jobs <- r
	}
	close(jobs)

	done := make(chan int, len(rects))
	var wg sync.WaitGroup
	sobol := (*Sobol)(nil)
	if cfg.Mode == SampleModeSobol {
		sobol = NewSobol(seed)
	}
	workerFn := func(rnd Random) {
		defer wg.Done()
		for r := range jobs {
			renderRect(scene, camera, sampler, samplesPerPixel, buf, cfg, sobol, rnd, r)
			done <- r.y1 - r.y0
		}
	}
	wg.Add(nworkers)
	for i := 0; i < nworkers; i++ {
		if cfg.Mode == SampleModeSobol {
			// Sobol mode shares the immutable source; no mutable RNG is
			// shared. The nil rnd is never consulted on this path.
			go workerFn(nil)
		} else {
			// Preserve historical behavior: each worker has its own
			// time-seeded source.
			rnd := rand.New(rand.NewSource(time.Now().UnixNano()))
			go workerFn(rnd)
		}
	}

	showProgress(start, scene.RayCount(), 0, h)
	completed := 0
	for range rects {
		completed += <-done
		showProgress(start, scene.RayCount(), completed, h)
	}
	wg.Wait()
	fmt.Println()
}

func renderRect(scene *Scene, camera *Camera, sampler Sampler, samplesPerPixel int, buf *Buffer, cfg RenderConfig, sobol *Sobol, rnd Random, rect rowRect) {
	for y := rect.y0; y < rect.y1; y++ {
		for x := rect.x0; x < rect.x1; x++ {
			renderPixel(scene, camera, sampler, samplesPerPixel, buf, cfg, sobol, rnd, x, y)
		}
	}
}

// renderPixel produces all samples for one pixel. In Sobol mode it is a pure
// function of the config, iteration, and absolute pixel coordinate, so worker
// count, job order and tiling cannot change the result.
func renderPixel(scene *Scene, camera *Camera, sampler Sampler, samplesPerPixel int, buf *Buffer, cfg RenderConfig, sobol *Sobol, rnd Random, x, y int) {
	w, h := buf.W, buf.H
	absSamples := int(math.Abs(float64(samplesPerPixel)))
	if cfg.Mode == SampleModeSobol {
		pixel := int64(y*w + x)
		renderPixelSobol(scene, camera, sampler, absSamples, buf, cfg, sobol, pixel, x, y, w, h)
		return
	}
	if samplesPerPixel <= 0 {
		// random subsampling
		for i := 0; i < absSamples; i++ {
			fu := rnd.Float64()
			fv := rnd.Float64()
			ray := camera.CastRay(x, y, w, h, fu, fv, rnd)
			sample := sampler.Sample(scene, ray, rnd)
			buf.AddSample(x, y, sample)
		}
	} else {
		// stratified subsampling
		n := int(math.Sqrt(float64(samplesPerPixel)))
		for u := 0; u < n; u++ {
			for v := 0; v < n; v++ {
				fu := (float64(u) + 0.5) / float64(n)
				fv := (float64(v) + 0.5) / float64(n)
				ray := camera.CastRay(x, y, w, h, fu, fv, rnd)
				sample := sampler.Sample(scene, ray, rnd)
				buf.AddSample(x, y, sample)
			}
		}
	}
	v := Clamp(buf.StandardDeviation(x, y).MaxComponent(), 0, 1)
	v = math.Pow(v, 2)
	extraSamples := 0
	if !cfg.DisableAdaptive {
		extraSamples = int(64 * v)
	}
	for i := 0; i < extraSamples; i++ {
		fu := rnd.Float64()
		fv := rnd.Float64()
		ray := camera.CastRay(x, y, w, h, fu, fv, rnd)
		sample := sampler.Sample(scene, ray, rnd)
		buf.AddSample(x, y, sample)
	}
}

// renderPixelSobol draws camera jitter, lens and material dimensions from one
// Sobol point per sample. Every sample gets a fresh point ordinal (dimension
// cursor reset to 0), so a long path never runs off the end of the Sobol
// dimension table and each sample's channels stay mutually stratified.
// Primary samples occupy ordinals [0, absSamples); adaptive samples continue
// at absSamples so they never repeat the primary point set.
func renderPixelSobol(scene *Scene, camera *Camera, sampler Sampler, absSamples int, buf *Buffer, cfg RenderConfig, sobol *Sobol, pixel int64, x, y, w, h int) {
	cursor := NewSobolRand(sobol, pixel, 0)
	addSample := func(sampleIndex int64) {
		// NextSample semantics: fresh Sobol point, dimension 0.
		cursor.SeekTo(pixel, sampleIndex, 0)
		fu := cursor.Float64()
		fv := cursor.Float64()
		ray := camera.CastRay(x, y, w, h, fu, fv, cursor)
		buf.AddSample(x, y, sampler.Sample(scene, ray, cursor))
	}
	for i := 0; i < absSamples; i++ {
		addSample(int64(i))
	}
	v := Clamp(buf.StandardDeviation(x, y).MaxComponent(), 0, 1)
	v = math.Pow(v, 2)
	extraSamples := 0
	if !cfg.DisableAdaptive {
		extraSamples = int(64 * v)
	}
	for i := 0; i < extraSamples; i++ {
		addSample(int64(absSamples + i))
	}
}

// RenderToBuffer renders with the given configuration and returns the
// accumulated sample buffer. In Sobol mode the result is reproducible: the
// same scene, camera, sampler and config produce identical pixels regardless
// of cfg.Workers or cfg.Tiles.
func RenderToBuffer(scene *Scene, camera *Camera, sampler Sampler, w, h, samplesPerPixel int, cfg RenderConfig) *Buffer {
	buf := NewBuffer(w, h)
	renderPass(scene, camera, sampler, samplesPerPixel, buf, cfg, 0)
	return buf
}

func writeImage(path string, buf *Buffer, channel Channel, wg *sync.WaitGroup) {
	defer wg.Done()
	im := buf.Image(channel)
	if err := SavePNG(path, im); err != nil {
		panic(err)
	}
}

func Render(scene *Scene, camera *Camera, sampler Sampler, w, h, samplesPerPixel int) image.Image {
	return RenderWithConfig(scene, camera, sampler, w, h, samplesPerPixel, DefaultRenderConfig())
}

// RenderWithConfig renders a single pass with explicit sampling and
// scheduling configuration. See RenderConfig and SampleMode.
func RenderWithConfig(scene *Scene, camera *Camera, sampler Sampler, w, h, samplesPerPixel int, cfg RenderConfig) image.Image {
	buf := NewBuffer(w, h)
	renderPass(scene, camera, sampler, samplesPerPixel, buf, cfg, 0)
	return buf.Image(ColorChannel)
}

// IterativeRender keeps the historical API and uses random sampling.
func IterativeRender(pathTemplate string, iterations int, scene *Scene, camera *Camera, sampler Sampler, w, h, samplesPerPixel int) image.Image {
	return IterativeRenderWithConfig(pathTemplate, iterations, scene, camera, sampler, w, h, samplesPerPixel, DefaultRenderConfig())
}

// IterativeRenderWithConfig progressively refines one buffer over several
// passes. In Sobol mode, pass i uses SobolSeedForIteration(cfg.Seed, i-1), so
// every iteration contributes a fresh deterministic point set and the final
// image remains reproducible.
func IterativeRenderWithConfig(pathTemplate string, iterations int, scene *Scene, camera *Camera, sampler Sampler, w, h, samplesPerPixel int, cfg RenderConfig) image.Image {
	var wg sync.WaitGroup
	scene.Compile()
	buf := NewBuffer(w, h)
	for i := 1; i <= iterations; i++ {
		fmt.Printf("\n[Iteration %d of %d]\n", i, iterations)
		renderPass(scene, camera, sampler, samplesPerPixel, buf, cfg, i-1)
		path := pathTemplate
		if strings.Contains(path, "%") {
			path = fmt.Sprintf(pathTemplate, i)
		}
		bufCopy := buf.Copy()
		wg.Add(1)
		go writeImage(path, bufCopy, ColorChannel, &wg)
		wg.Add(1)
		go writeImage("deviation.png", bufCopy, StandardDeviationChannel, &wg)
	}
	wg.Wait()
	return buf.Image(ColorChannel)
}

func ChannelRender(scene *Scene, camera *Camera, sampler Sampler, w, h, samplesPerPixel int) <-chan image.Image {
	return ChannelRenderWithConfig(scene, camera, sampler, w, h, samplesPerPixel, DefaultRenderConfig())
}

func ChannelRenderWithConfig(scene *Scene, camera *Camera, sampler Sampler, w, h, samplesPerPixel int, cfg RenderConfig) <-chan image.Image {
	ch := make(chan image.Image)
	go func() {
		scene.Compile()
		buf := NewBuffer(w, h)
		for i := 0; ; i++ {
			// renderPass derives SobolSeedForIteration(cfg.Seed, i), so each
			// streamed frame uses a fresh, reproducible point set.
			renderPass(scene, camera, sampler, samplesPerPixel, buf, cfg, i)
			ch <- buf.Image(ColorChannel)
		}
	}()
	return ch
}

func FrameRender(path string, iterations int, scene *Scene, camera *Camera, sampler Sampler, w, h, samplesPerPixel int, wg *sync.WaitGroup) {
	FrameRenderWithConfig(path, iterations, scene, camera, sampler, w, h, samplesPerPixel, DefaultRenderConfig(), wg)
}

func FrameRenderWithConfig(path string, iterations int, scene *Scene, camera *Camera, sampler Sampler, w, h, samplesPerPixel int, cfg RenderConfig, wg *sync.WaitGroup) {
	buf := NewBuffer(w, h)
	for i := 1; i <= iterations; i++ {
		renderPass(scene, camera, sampler, samplesPerPixel, buf, cfg, i-1)
	}
	wg.Add(1)
	go writeImage(path, buf.Copy(), ColorChannel, wg)
}
