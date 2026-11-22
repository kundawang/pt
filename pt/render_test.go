package pt

import (
	"context"
	"math/rand"
	"runtime"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// constSampler ignores the scene and always returns the same color, making
// renders fully deterministic: every pixel gets exactly abs(spp) samples
// (standard deviation is zero, so no adaptive extra samples are added).
type constSampler struct{ c Color }

func (s constSampler) Sample(scene *Scene, ray Ray, rnd *rand.Rand) Color { return s.c }

// slowSampler simulates expensive primary/secondary ray tracing.
type slowSampler struct {
	d     time.Duration
	calls int64
}

func (s *slowSampler) Sample(scene *Scene, ray Ray, rnd *rand.Rand) Color {
	atomic.AddInt64(&s.calls, 1)
	time.Sleep(s.d)
	return White
}

func testCamera() *Camera {
	c := LookAt(V(0, 0, -5), V(0, 0, 0), V(0, 1, 0), 45)
	return &c
}

func TestTilesEdgeClamping(t *testing.T) {
	c := RenderConfig{TileSize: 16}
	tiles := c.Tiles(40, 20)
	// 3 x 2 grid; right column is 8 wide, bottom row is 4 tall.
	if len(tiles) != 6 {
		t.Fatalf("expected 6 tiles, got %d", len(tiles))
	}
	last := tiles[len(tiles)-1]
	if last != (Tile{32, 16, 8, 4}) {
		t.Fatalf("edge tile not clamped: %+v", last)
	}
	// Full coverage, no overlap.
	covered := make([]bool, 40*20)
	for _, tile := range tiles {
		for y := tile.Y; y < tile.Y+tile.H; y++ {
			for x := tile.X; x < tile.X+tile.W; x++ {
				if covered[y*40+x] {
					t.Fatalf("pixel (%d,%d) covered twice", x, y)
				}
				covered[y*40+x] = true
			}
		}
	}
	for i, c := range covered {
		if !c {
			t.Fatalf("pixel %d not covered", i)
		}
	}
}

func TestTilesZeroSize(t *testing.T) {
	c := RenderConfig{TileSize: 16}
	if tiles := c.Tiles(0, 10); tiles != nil {
		t.Fatalf("expected no tiles for zero width, got %d", len(tiles))
	}
	if tiles := c.Tiles(10, 0); tiles != nil {
		t.Fatalf("expected no tiles for zero height, got %d", len(tiles))
	}
	if tiles := c.Tiles(-3, -3); tiles != nil {
		t.Fatalf("expected no tiles for negative size, got %d", len(tiles))
	}
}

func TestTilesOversized(t *testing.T) {
	c := RenderConfig{TileSize: 10000}
	tiles := c.Tiles(40, 20)
	if len(tiles) != 1 || tiles[0] != (Tile{0, 0, 40, 20}) {
		t.Fatalf("oversized tile should yield a single clamped tile, got %+v", tiles)
	}
}

func TestRenderZeroSizeBuffer(t *testing.T) {
	r := NewRenderer(&Scene{}, testCamera(), constSampler{White}, 1, RenderConfig{
		TileSize: 8,
		Progress: func(done, total int) { t.Errorf("progress called for empty render") },
	})
	if err := r.Render(context.Background(), NewBuffer(0, 0)); err != nil {
		t.Fatalf("zero-size render should return nil, got %v", err)
	}
	done, total := r.Progress()
	if done != 0 || total != 0 {
		t.Fatalf("expected 0/0 progress, got %d/%d", done, total)
	}
	// Pre-cancelled context on an empty buffer reports the cancellation.
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if err := r.Render(ctx, NewBuffer(0, 5)); err != context.Canceled {
		t.Fatalf("expected context.Canceled, got %v", err)
	}
}

func TestRenderCompleteAndDeterministic(t *testing.T) {
	green := Color{0.25, 0.5, 0.75}
	reference := map[[2]int]Color{}
	for i, cfg := range []RenderConfig{
		{TileSize: 4, Workers: 1},
		{TileSize: 7, Workers: 3},
		{TileSize: 64, Workers: 8},
		{TileSize: 0, Workers: 0}, // defaults
	} {
		scene := &Scene{}
		buf := NewBuffer(20, 12)
		var mu sync.Mutex
		var calls [][2]int
		r := NewRenderer(scene, testCamera(), constSampler{green}, 4, RenderConfig{
			TileSize: cfg.TileSize,
			Workers:  cfg.Workers,
			Progress: func(done, total int) {
				mu.Lock()
				calls = append(calls, [2]int{done, total})
				mu.Unlock()
			},
		})
		if err := r.Render(context.Background(), buf); err != nil {
			t.Fatalf("config %d: %v", i, err)
		}
		// Every pixel: exactly 4 samples, exact color, zero variance.
		for y := 0; y < buf.H; y++ {
			for x := 0; x < buf.W; x++ {
				if n := buf.Samples(x, y); n != 4 {
					t.Fatalf("config %d: pixel (%d,%d) has %d samples, want 4", i, x, y, n)
				}
				if c := buf.Color(x, y); c != green {
					t.Fatalf("config %d: pixel (%d,%d) color %v, want %v", i, x, y, c, green)
				}
				if v := buf.Variance(x, y); v != Black {
					t.Fatalf("config %d: pixel (%d,%d) variance %v, want 0", i, x, y, v)
				}
				if i == 0 {
					reference[[2]int{x, y}] = buf.Color(x, y)
				} else if buf.Color(x, y) != reference[[2]int{x, y}] {
					t.Fatalf("config %d: pixel (%d,%d) differs across tile orders", i, x, y)
				}
			}
		}
		// Progress: monotonic, ends at total, total matches tile count.
		total := len(r.Tiles(buf.W, buf.H))
		if len(calls) != total {
			t.Fatalf("config %d: %d progress calls, want %d", i, len(calls), total)
		}
		for j, c := range calls {
			if c[0] != j+1 || c[1] != total {
				t.Fatalf("config %d: progress call %d = %v, want [%d %d]", i, j, c, j+1, total)
			}
		}
		done, tot := r.Progress()
		if done != total || tot != total {
			t.Fatalf("config %d: Progress() = %d/%d, want %d/%d", i, done, tot, total, total)
		}
		if n := len(r.CompletedTiles()); n != total {
			t.Fatalf("config %d: %d completed tiles, want %d", i, n, total)
		}
		for j := 0; j < total; j++ {
			if !r.TileDone(j) {
				t.Fatalf("config %d: tile %d not marked done", i, j)
			}
		}
	}
}

func goroutineCount(t *testing.T) int {
	t.Helper()
	// Let stragglers exit before counting.
	time.Sleep(50 * time.Millisecond)
	return runtime.NumGoroutine()
}

func TestRenderCancelMidway(t *testing.T) {
	scene := &Scene{}
	sampler := &slowSampler{d: 2 * time.Millisecond}
	buf := NewBuffer(64, 64)
	ctx, cancel := context.WithCancel(context.Background())
	var lastDone int64
	r := NewRenderer(scene, testCamera(), sampler, 1, RenderConfig{
		TileSize: 8,
		Workers:  4,
		Progress: func(done, total int) { atomic.StoreInt64(&lastDone, int64(done)) },
	})
	total := len(r.Tiles(buf.W, buf.H))

	before := goroutineCount(t)
	go func() {
		// Cancel once some tiles have completed but not all.
		for atomic.LoadInt64(&lastDone) < 2 {
			time.Sleep(time.Millisecond)
		}
		cancel()
		cancel() // repeated cancel must be harmless
	}()
	start := time.Now()
	err := r.Render(ctx, buf)
	elapsed := time.Since(start)
	if err != context.Canceled {
		t.Fatalf("expected context.Canceled, got %v", err)
	}
	if elapsed > 5*time.Second {
		t.Fatalf("cancellation did not converge promptly: %v", elapsed)
	}
	done, tot := r.Progress()
	if tot != total {
		t.Fatalf("total changed: %d, want %d", tot, total)
	}
	if done == 0 || done >= total {
		t.Fatalf("expected partial progress, got %d/%d", done, total)
	}
	completed := r.CompletedTiles()
	if len(completed) != done {
		t.Fatalf("CompletedTiles = %d, progress = %d", len(completed), done)
	}
	// Completed tiles kept their samples in the buffer: nothing was lost.
	tiles := r.Tiles(buf.W, buf.H)
	for _, idx := range completed {
		tile := tiles[idx]
		for y := tile.Y; y < tile.Y+tile.H; y++ {
			for x := tile.X; x < tile.X+tile.W; x++ {
				if buf.Samples(x, y) == 0 {
					t.Fatalf("completed tile %d has unsampled pixel (%d,%d)", idx, x, y)
				}
			}
		}
	}
	// Workers and WaitGroups are reclaimed.
	if after := goroutineCount(t); after > before+2 {
		t.Fatalf("goroutines leaked: before=%d after=%d", before, after)
	}
}

func TestRenderCancelDuringSecondaryRays(t *testing.T) {
	// The sampler spends a long time inside Sample (secondary rays);
	// cancellation must still converge without waiting for every pixel.
	scene := &Scene{}
	sampler := &slowSampler{d: 500 * time.Millisecond}
	buf := NewBuffer(16, 16)
	r := NewRenderer(scene, testCamera(), sampler, 1, RenderConfig{TileSize: 4, Workers: 4})
	ctx, cancel := context.WithCancel(context.Background())
	go func() {
		time.Sleep(100 * time.Millisecond)
		cancel()
	}()
	start := time.Now()
	err := r.Render(ctx, buf)
	elapsed := time.Since(start)
	if err != context.Canceled {
		t.Fatalf("expected context.Canceled, got %v", err)
	}
	// Workers finish at most one in-flight sample each after cancel.
	if elapsed > 2*time.Second {
		t.Fatalf("cancellation during sampling did not converge: %v", elapsed)
	}
}

func TestRenderPreCancelled(t *testing.T) {
	scene := &Scene{}
	buf := NewBuffer(32, 32)
	r := NewRenderer(scene, testCamera(), constSampler{White}, 1, RenderConfig{TileSize: 8, Workers: 2})
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	cancel()
	before := goroutineCount(t)
	if err := r.Render(ctx, buf); err != context.Canceled {
		t.Fatalf("expected context.Canceled, got %v", err)
	}
	done, _ := r.Progress()
	if done != 0 {
		t.Fatalf("pre-cancelled render completed %d tiles", done)
	}
	for y := 0; y < buf.H; y++ {
		for x := 0; x < buf.W; x++ {
			if buf.Samples(x, y) != 0 {
				t.Fatalf("pre-cancelled render wrote samples at (%d,%d)", x, y)
			}
		}
	}
	if after := goroutineCount(t); after > before+2 {
		t.Fatalf("goroutines leaked: before=%d after=%d", before, after)
	}
}

func TestRenderRestartAfterCancel(t *testing.T) {
	scene := &Scene{}
	green := Color{0.1, 0.2, 0.3}
	buf := NewBuffer(64, 64)
	cfg := RenderConfig{TileSize: 4, Workers: 4}
	r := NewRenderer(scene, testCamera(), constSampler{green}, 4, cfg)

	// First pass: cancel partway (synchronously, from the progress hook).
	ctx, cancel := context.WithCancel(context.Background())
	r.config.Progress = func(done, total int) {
		if done == 3 {
			cancel()
		}
	}
	if err := r.Render(ctx, buf); err != context.Canceled {
		t.Fatalf("expected context.Canceled, got %v", err)
	}
	partial, total := r.Progress()
	if partial == 0 || partial >= total {
		t.Fatalf("expected partial first pass, got %d/%d", partial, total)
	}

	// Restart with a fresh context: the renderer is reusable and the
	// buffer keeps accumulating samples.
	r.config.Progress = nil
	if err := r.Render(context.Background(), buf); err != nil {
		t.Fatalf("restart failed: %v", err)
	}
	done, _ := r.Progress()
	if done != total {
		t.Fatalf("restart progress = %d/%d", done, total)
	}
	for y := 0; y < buf.H; y++ {
		for x := 0; x < buf.W; x++ {
			n := buf.Samples(x, y)
			if n < 4 {
				t.Fatalf("pixel (%d,%d) has %d samples after restart", x, y, n)
			}
			if c := buf.Color(x, y); c != green {
				t.Fatalf("pixel (%d,%d) color %v, want %v", x, y, c, green)
			}
		}
	}
}

func TestRenderWorkerCounts(t *testing.T) {
	scene := &Scene{}
	for _, workers := range []int{1, 2, 7, 64, -3, 0} {
		buf := NewBuffer(24, 24)
		r := NewRenderer(scene, testCamera(), constSampler{White}, 1, RenderConfig{TileSize: 6, Workers: workers})
		if err := r.Render(context.Background(), buf); err != nil {
			t.Fatalf("workers=%d: %v", workers, err)
		}
		for y := 0; y < buf.H; y++ {
			for x := 0; x < buf.W; x++ {
				if buf.Samples(x, y) != 1 {
					t.Fatalf("workers=%d: pixel (%d,%d) samples=%d", workers, x, y, buf.Samples(x, y))
				}
			}
		}
	}
}

func TestRenderConfigWorkerClamp(t *testing.T) {
	if n := (RenderConfig{Workers: 5}).workers(); n != 5 {
		t.Fatalf("explicit workers not honored: %d", n)
	}
	if n := (RenderConfig{}).workers(); n < 1 {
		t.Fatalf("default workers out of range: %d", n)
	}
	if n := (RenderConfig{Workers: -1}).workers(); n < 1 {
		t.Fatalf("negative workers not clamped: %d", n)
	}
}

func TestIterativeRenderContextCancel(t *testing.T) {
	scene := &Scene{}
	scene.Add(NewSphere(V(0, 0, 0), 1, LightMaterial(White, 1)))
	sampler := &slowSampler{d: 50 * time.Millisecond}
	ctx, cancel := context.WithCancel(context.Background())
	go func() {
		time.Sleep(100 * time.Millisecond)
		cancel()
	}()
	before := goroutineCount(t)
	dir := t.TempDir()
	_, err := IterativeRenderContext(ctx, dir+"/out%03d.png", 10, scene, testCamera(), sampler, 16, 16, 1, RenderConfig{TileSize: 8, Workers: 2})
	if err != context.Canceled {
		t.Fatalf("expected context.Canceled, got %v", err)
	}
	if after := goroutineCount(t); after > before+2 {
		t.Fatalf("goroutines leaked: before=%d after=%d", before, after)
	}
}

func TestRenderCompatibility(t *testing.T) {
	// The default Render entry point still works and returns a full image.
	// The camera looks straight at an emissive sphere, so the image must
	// contain non-black pixels.
	scene := &Scene{}
	scene.Add(NewSphere(V(0, 0, 0), 1, LightMaterial(White, 1)))
	sampler := NewSampler(1, 1)
	im := Render(scene, testCamera(), sampler, 8, 8, 1)
	b := im.Bounds()
	if b.Dx() != 8 || b.Dy() != 8 {
		t.Fatalf("unexpected image bounds: %v", b)
	}
	// The scene was lit, so some pixel must be non-black.
	nonBlack := false
	for y := 0; y < 8 && !nonBlack; y++ {
		for x := 0; x < 8; x++ {
			r, g, bl, _ := im.At(x, y).RGBA()
			if r+g+bl > 0 {
				nonBlack = true
				break
			}
		}
	}
	if !nonBlack {
		t.Fatal("Render produced an all-black image")
	}
}

func TestStratifiedSampleCounts(t *testing.T) {
	// Positive spp uses stratified n x n subsampling: n^2 samples/pixel.
	scene := &Scene{}
	buf := NewBuffer(10, 10)
	r := NewRenderer(scene, testCamera(), constSampler{White}, 9, RenderConfig{TileSize: 5, Workers: 2})
	if err := r.Render(context.Background(), buf); err != nil {
		t.Fatal(err)
	}
	for y := 0; y < buf.H; y++ {
		for x := 0; x < buf.W; x++ {
			if n := buf.Samples(x, y); n != 9 {
				t.Fatalf("pixel (%d,%d) has %d samples, want 9", x, y, n)
			}
		}
	}
}

func TestProgressCallbackNotConcurrent(t *testing.T) {
	scene := &Scene{}
	buf := NewBuffer(64, 64)
	var inCallback int64
	r := NewRenderer(scene, testCamera(), constSampler{White}, 1, RenderConfig{
		TileSize: 4,
		Workers:  8,
		Progress: func(done, total int) {
			if !atomic.CompareAndSwapInt64(&inCallback, 0, 1) {
				t.Error("progress callback invoked concurrently")
			}
			time.Sleep(time.Millisecond)
			atomic.StoreInt64(&inCallback, 0)
		},
	})
	if err := r.Render(context.Background(), buf); err != nil {
		t.Fatal(err)
	}
}
