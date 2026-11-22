package pt

import (
	"context"
	"math"
	"math/rand"
	"runtime"
	"sync"
	"time"
)

// Tile is a rectangular block of pixels that is rendered as a single
// schedulable unit. Tiles at the right and bottom edges of the image are
// clamped, so W and H may be smaller than the configured tile size.
type Tile struct {
	X, Y, W, H int
}

// DefaultTileSize is used when RenderConfig.TileSize is not positive.
const DefaultTileSize = 64

// RenderConfig controls how a Renderer schedules work.
type RenderConfig struct {
	// TileSize is the side length of a tile in pixels. Values <= 0 select
	// DefaultTileSize. Values larger than the image are clamped, so a huge
	// tile size simply yields a single tile.
	TileSize int
	// Workers is the maximum number of goroutines rendering concurrently.
	// Values <= 0 select runtime.NumCPU(), clamped to at least 1.
	Workers int
	// Progress, if non-nil, is called after each tile completes with the
	// number of completed tiles and the total tile count. It is never
	// called concurrently and never after Render returns.
	Progress func(done, total int)
}

func (c RenderConfig) tileSize() int {
	if c.TileSize <= 0 {
		return DefaultTileSize
	}
	return c.TileSize
}

func (c RenderConfig) workers() int {
	if c.Workers > 0 {
		return c.Workers
	}
	n := runtime.NumCPU()
	if n < 1 {
		n = 1
	}
	return n
}

// Tiles splits a w x h image into tiles of the configured size, in
// row-major order. A zero or negative dimension yields no tiles.
func (c RenderConfig) Tiles(w, h int) []Tile {
	if w <= 0 || h <= 0 {
		return nil
	}
	ts := c.tileSize()
	var tiles []Tile
	for y := 0; y < h; y += ts {
		for x := 0; x < w; x += ts {
			tw, th := ts, ts
			if x+tw > w {
				tw = w - x
			}
			if y+th > h {
				th = h - y
			}
			tiles = append(tiles, Tile{x, y, tw, th})
		}
	}
	return tiles
}

// Renderer renders a scene into a Buffer tile by tile. It supports
// cancellation via context, a bounded worker pool, progress callbacks and
// querying which tiles have completed. A Renderer may be reused across
// Render calls (e.g. after a cancellation), but Render must not be called
// concurrently on the same Renderer.
type Renderer struct {
	scene           *Scene
	camera          *Camera
	sampler         Sampler
	samplesPerPixel int
	config          RenderConfig

	mu        sync.Mutex
	cbMu      sync.Mutex // serializes Progress callbacks
	tiles     []Tile
	completed map[int]bool
	done      int
}

// NewRenderer creates a Renderer for the given scene, camera and sampler.
// samplesPerPixel follows the same convention as Render: positive values
// use stratified subsampling, non-positive values use random subsampling
// with abs(samplesPerPixel) samples.
func NewRenderer(scene *Scene, camera *Camera, sampler Sampler, samplesPerPixel int, config RenderConfig) *Renderer {
	return &Renderer{
		scene:           scene,
		camera:          camera,
		sampler:         sampler,
		samplesPerPixel: samplesPerPixel,
		config:          config,
	}
}

// Tiles returns the tile layout the renderer would use for a w x h image.
func (r *Renderer) Tiles(w, h int) []Tile {
	return r.config.Tiles(w, h)
}

// Progress returns the number of completed tiles and the total tile count
// of the most recent (or in-flight) Render call.
func (r *Renderer) Progress() (done, total int) {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.done, len(r.tiles)
}

// CompletedTiles returns the indices (into Tiles) of the tiles that
// finished during the most recent (or in-flight) Render call.
func (r *Renderer) CompletedTiles() []int {
	r.mu.Lock()
	defer r.mu.Unlock()
	result := make([]int, 0, len(r.completed))
	for i := range r.completed {
		result = append(result, i)
	}
	return result
}

// TileDone reports whether tile i (in Tiles order) has completed.
func (r *Renderer) TileDone(i int) bool {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.completed[i]
}

// Render renders into buf, adding samples to any already present. It
// returns nil on success. If ctx is cancelled, Render returns ctx.Err()
// promptly — even mid-tile — but samples from completed tiles (and
// completed pixels of an interrupted tile) remain in buf and are not
// lost. A zero-sized buffer renders nothing and returns ctx.Err()
// (nil when ctx is not cancelled).
func (r *Renderer) Render(ctx context.Context, buf *Buffer) error {
	if ctx == nil {
		ctx = context.Background()
	}
	tiles := r.config.Tiles(buf.W, buf.H)
	r.mu.Lock()
	r.tiles = tiles
	r.completed = make(map[int]bool, len(tiles))
	r.done = 0
	r.mu.Unlock()

	if len(tiles) == 0 {
		return ctx.Err()
	}

	workers := r.config.workers()
	jobs := make(chan int)
	var wg sync.WaitGroup
	for i := 0; i < workers; i++ {
		wg.Add(1)
		go func(seed int64) {
			defer wg.Done()
			rnd := rand.New(rand.NewSource(seed))
			for index := range jobs {
				t := tiles[index]
				if !r.renderTile(ctx, buf, t, rnd) {
					// Cancelled mid-tile: stop pulling jobs.
					return
				}
				r.mu.Lock()
				r.completed[index] = true
				r.done++
				done, total := r.done, len(tiles)
				r.mu.Unlock()
				if r.config.Progress != nil {
					r.cbMu.Lock()
					r.config.Progress(done, total)
					r.cbMu.Unlock()
				}
			}
		}(time.Now().UnixNano() + int64(i)*7919)
	}

	// Feed jobs; stop early if the context is cancelled. The jobs channel
	// is unbuffered so a cancelled worker never leaves a job stranded.
feed:
	for i := range tiles {
		select {
		case <-ctx.Done():
			break feed
		case jobs <- i:
		}
	}
	close(jobs)
	wg.Wait()
	return ctx.Err()
}

// renderTile renders a single tile into buf. It reports false if the
// context was cancelled before the tile was finished.
func (r *Renderer) renderTile(ctx context.Context, buf *Buffer, t Tile, rnd *rand.Rand) bool {
	scene := r.scene
	camera := r.camera
	sampler := r.sampler
	samplesPerPixel := r.samplesPerPixel
	w, h := buf.W, buf.H
	absSamples := int(math.Abs(float64(samplesPerPixel)))
	for y := t.Y; y < t.Y+t.H; y++ {
		for x := t.X; x < t.X+t.W; x++ {
			// Cancellation is checked per pixel, so cancelling while a
			// primary or secondary ray is being traced converges as soon
			// as the in-flight sample returns.
			if ctx.Err() != nil {
				return false
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
			extraSamples := int(64 * v)
			for i := 0; i < extraSamples; i++ {
				fu := rnd.Float64()
				fv := rnd.Float64()
				ray := camera.CastRay(x, y, w, h, fu, fv, rnd)
				sample := sampler.Sample(scene, ray, rnd)
				buf.AddSample(x, y, sample)
			}
		}
	}
	return true
}
