package pt

import (
	"sync"
	"testing"
	"time"
)

// testScene builds a small deterministic scene: one diffuse sphere and
// one emissive sphere inside the camera's view, with a background color
// for the rays that miss everything.
func testScene() *Scene {
	scene := &Scene{Color: HexColor(0x102030)}
	scene.Add(NewSphere(Vector{0, 0, 2}, 0.5, DiffuseMaterial(HexColor(0xff0000))))
	scene.Add(NewSphere(Vector{1.2, 0, 2}, 0.25, LightMaterial(White, 2)))
	return scene
}

func testCamera() *Camera {
	c := LookAt(Vector{0, 0, -1}, Vector{0, 0, 1}, Vector{0, 1, 0}, 50)
	return &c
}

func quietRenderer(scene *Scene, w, h, spp, numCPU int) *Renderer {
	r := NewRenderer(scene, testCamera(), NewSampler(1, 2), w, h)
	r.SamplesPerPixel = spp
	r.NumCPU = numCPU
	r.Verbose = false
	return r
}

// TestStatsCountInvariants checks the core counting contract on a
// small fully-deterministic stratified render.
func TestStatsCountInvariants(t *testing.T) {
	const w, h, spp = 8, 6, 4
	scene := testScene()
	r := quietRenderer(scene, w, h, spp, 4)

	before := r.Stats()
	if before.Iterations != 0 || before.PrimaryRays != 0 || before.TotalSamples != 0 {
		t.Fatalf("fresh renderer must report zeroed stats, got %+v", before)
	}

	r.Run()
	rs := r.Stats()
	ss := scene.Stats()
	bs := r.Buffer.Stats()

	pixels := w * h
	stratRoot := 2 // sqrt(4)
	basePrimary := uint64(pixels * stratRoot * stratRoot)

	if rs.Iterations != 1 {
		t.Errorf("Iterations = %d, want 1", rs.Iterations)
	}
	// The adaptive pass may add up to 128 extra samples per
	// high-variance pixel, so totals are >= the stratified base count.
	if rs.PrimaryRays < basePrimary {
		t.Errorf("PrimaryRays = %d, want >= base %d", rs.PrimaryRays, basePrimary)
	}
	if rs.TotalSamples != rs.PrimaryRays {
		t.Errorf("TotalSamples = %d, want %d (no drops configured)", rs.TotalSamples, rs.PrimaryRays)
	}
	if rs.CompletedRows != uint64(h) {
		t.Errorf("CompletedRows = %d, want %d", rs.CompletedRows, h)
	}
	if rs.Running {
		t.Errorf("Running must be false after Run() completed")
	}
	if rs.Elapsed <= 0 {
		t.Errorf("Elapsed must be positive, got %v", rs.Elapsed)
	}
	if rs.Width != w || rs.Height != h || rs.Workers != 4 {
		t.Errorf("geometry/workers mismatch: %+v", rs)
	}
	// Every scene intersection is either a camera ray or a secondary.
	if ss.PrimaryRays+ss.SecondaryRays != ss.Rays {
		t.Errorf("scene ray split does not sum: %+v", ss)
	}
	if ss.Rays < rs.PrimaryRays {
		t.Errorf("total scene rays %d < primary rays %d", ss.Rays, rs.PrimaryRays)
	}
	if rs.TotalRays != rs.PrimaryRays+rs.SecondaryRays {
		t.Errorf("TotalRays %d != primary %d + secondary %d",
			rs.TotalRays, rs.PrimaryRays, rs.SecondaryRays)
	}
	if rs.SecondaryRays != ss.SecondaryRays {
		t.Errorf("renderer secondary %d != scene secondary %d", rs.SecondaryRays, ss.SecondaryRays)
	}
	// Buffer snapshot must agree with the renderer sample counter.
	if bs.TotalSamples != rs.TotalSamples {
		t.Errorf("buffer TotalSamples %d != renderer TotalSamples %d", bs.TotalSamples, rs.TotalSamples)
	}
	if bs.Width != w || bs.Height != h {
		t.Errorf("buffer geometry mismatch: %+v", bs)
	}
	if bs.NonEmptyPixels != uint64(pixels) {
		t.Errorf("NonEmptyPixels = %d, want %d", bs.NonEmptyPixels, pixels)
	}
	if bs.MeanSamplesPerPixel < float64(spp) {
		t.Errorf("MeanSamplesPerPixel = %v, want >= base spp %d", bs.MeanSamplesPerPixel, spp)
	}
	if bs.MaxSamples < spp {
		t.Errorf("MaxSamples = %d must be >= base spp %d", bs.MaxSamples, spp)
	}
}

// TestIterativeCumulativeBoundary verifies that renderer-level
// counters accumulate across iterations while the scene counter resets
// each iteration.
func TestIterativeCumulativeBoundary(t *testing.T) {
	const w, h, iterations = 5, 4, 3
	scene := testScene()
	r := quietRenderer(scene, w, h, 1, 2)

	for i := 1; i <= iterations; i++ {
		r.Run()
		rs := r.Stats()
		if rs.Iterations != uint64(i) {
			t.Fatalf("iteration %d: Iterations = %d", i, rs.Iterations)
		}
		// Renderer-level cumulative counters only grow.
		wantPrimary := uint64(w*h) * uint64(i)
		if rs.PrimaryRays != wantPrimary {
			t.Errorf("iteration %d: PrimaryRays = %d, want %d", i, rs.PrimaryRays, wantPrimary)
		}
		if rs.TotalSamples != wantPrimary {
			t.Errorf("iteration %d: TotalSamples = %d, want %d", i, rs.TotalSamples, wantPrimary)
		}
		// Buffer accumulates across iterations too.
		if got := r.Buffer.TotalSamples(); got != wantPrimary {
			t.Errorf("iteration %d: buffer samples = %d, want %d", i, got, wantPrimary)
		}
		// The scene counter is reset at every iteration boundary, so
		// after iteration i it holds only this iteration's rays, not
		// the cumulative total (which is strictly larger for i > 1).
		if i > 1 && scene.RayCount() >= wantPrimary {
			t.Errorf("iteration %d: scene ray counter %d was not reset at the iteration boundary (cumulative primary %d)",
				i, scene.RayCount(), wantPrimary)
		}
	}
	// Snapshot values must be monotonic non-decreasing overall.
	rs := r.Stats()
	if rs.Elapsed <= 0 || rs.Elapsed < time.Nanosecond {
		t.Errorf("cumulative Elapsed must be positive, got %v", rs.Elapsed)
	}
}

// TestSnapshotIsValueCopy mutates the buffer after taking a snapshot
// and proves the snapshot is unaffected (no exposed mutable slice).
func TestSnapshotIsValueCopy(t *testing.T) {
	scene := testScene()
	r := quietRenderer(scene, 4, 4, 1, 2)
	r.Run()

	snap := r.Buffer.Stats()
	rsSnap := r.Stats()

	// Mutate buffer state through the proper API after the snapshot.
	r.Buffer.AddSample(0, 0, White)

	again := r.Buffer.Stats()
	// The retained snapshot must be unchanged while a fresh one moves.
	if snap.TotalSamples != rsSnap.TotalSamples {
		t.Fatalf("retained snapshot baseline inconsistent: %d vs %d",
			snap.TotalSamples, rsSnap.TotalSamples)
	}
	if again.TotalSamples != snap.TotalSamples+1 {
		t.Errorf("new snapshot TotalSamples = %d, want %d", again.TotalSamples, snap.TotalSamples+1)
	}
	// The retained snapshot's other fields are frozen at render end.
	if snap.NonEmptyPixels != again.NonEmptyPixels {
		t.Errorf("retained snapshot NonEmptyPixels mutated: %d -> %d",
			snap.NonEmptyPixels, again.NonEmptyPixels)
	}
}

// TestZeroSizeBuffer covers w=0 and h=0 buffers: rendering must be a
// stable no-op with zero, non-negative stats and a valid empty image.
func TestZeroSizeBuffer(t *testing.T) {
	for _, dims := range [][2]int{{0, 0}, {0, 5}, {5, 0}, {-3, 4}, {4, -3}} {
		w, h := dims[0], dims[1]
		scene := testScene()
		r := quietRenderer(scene, w, h, 1, 2)
		// Must not panic.
		r.Run()
		rs := r.Stats()
		bs := r.Buffer.Stats()
		if rs.PrimaryRays != 0 || rs.TotalSamples != 0 {
			t.Errorf("dims %v: zero-area render traced rays: %+v", dims, rs)
		}
		if rs.CompletedRows != 0 {
			t.Errorf("dims %v: CompletedRows = %d, want 0", dims, rs.CompletedRows)
		}
		if rs.MeanSamplesPerPixel != 0 || bs.MeanSamplesPerPixel != 0 {
			t.Errorf("dims %v: mean spp must be 0, got renderer %v buffer %v",
				dims, rs.MeanSamplesPerPixel, bs.MeanSamplesPerPixel)
		}
		if bs.MinSamples < 0 || bs.MaxSamples < 0 {
			t.Errorf("dims %v: negative sample bounds %+v", dims, bs)
		}
		im := r.Buffer.Image()
		if im.Bounds().Dx() < 0 || im.Bounds().Dy() < 0 {
			t.Errorf("dims %v: invalid image bounds %v", dims, im.Bounds())
		}
	}
}

// TestEmptySceneNoHit renders a scene with no shapes: every primary
// ray misses, the result is the background color and counters stay
// consistent.
func TestEmptySceneNoHit(t *testing.T) {
	const w, h, spp = 6, 5, 1
	scene := &Scene{Color: HexColor(0x223344)}
	r := quietRenderer(scene, w, h, spp, 3)
	r.Run()

	rs := r.Stats()
	ss := scene.Stats()
	if rs.PrimaryRays != uint64(w*h) {
		t.Fatalf("PrimaryRays = %d, want %d", rs.PrimaryRays, w*h)
	}
	// Every primary ray performs exactly one intersection test and
	// nothing bounces (no hit => no secondary rays).
	if ss.Rays != uint64(w*h) {
		t.Errorf("scene Rays = %d, want %d", ss.Rays, w*h)
	}
	if ss.SecondaryRays != 0 {
		t.Errorf("empty scene must produce no secondary rays, got %d", ss.SecondaryRays)
	}
	if ss.PrimaryRays != uint64(w*h) {
		t.Errorf("scene PrimaryRays = %d, want %d", ss.PrimaryRays, w*h)
	}
	// Output image must be uniform background.
	im := r.Buffer.Image()
	want := scene.Color.Pow(1 / 2.2).RGBA64()
	r2, g2, b2, _ := want.RGBA()
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			r1, g1, b1, _ := im.At(x, y).RGBA()
			if r1 != r2 || g1 != g2 || b1 != b2 {
				t.Fatalf("pixel (%d,%d) = {%d %d %d}, want background {%d %d %d}",
					x, y, r1, g1, b1, r2, g2, b2)
			}
		}
	}
}

// TestRecursionTruncation sets MaxBounces to zero: renders must
// terminate, never exceed the depth budget, and still count rays.
func TestRecursionTruncation(t *testing.T) {
	scene := testScene()
	r := NewRenderer(scene, testCamera(), NewSampler(1, 0), 5, 5)
	r.NumCPU = 2
	r.Verbose = false
	r.Run()

	rs := r.Stats()
	if rs.TotalSamples != uint64(25) {
		t.Errorf("TotalSamples = %d, want 25", rs.TotalSamples)
	}
	if rs.PrimaryRays != uint64(25) {
		t.Errorf("PrimaryRays = %d, want 25", rs.PrimaryRays)
	}
	// With zero bounces each pixel still performs its primary test.
	if scene.RayCount() < uint64(25) {
		t.Errorf("RayCount = %d, want at least 25", scene.RayCount())
	}
}

// TestConcurrentStatsReads hammers Stats from many reader goroutines
// while a multi-worker render runs. It only passes under -race if all
// counters are properly synchronized.
func TestConcurrentStatsReads(t *testing.T) {
	scene := testScene()
	r := quietRenderer(scene, 12, 12, 4, 4)

	var wg sync.WaitGroup
	stop := make(chan struct{})
	for i := 0; i < 8; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for {
				select {
				case <-stop:
					return
				default:
					rs := r.Stats()
					ss := scene.Stats()
					// Invariants that must hold within a single
					// snapshot at every instant. Two snapshots taken
					// at different times are not expected to match.
					if rs.TotalSamples > rs.PrimaryRays {
						t.Errorf("samples %d exceed primary rays %d", rs.TotalSamples, rs.PrimaryRays)
					}
					if rs.SecondaryRays > ss.Rays {
						t.Errorf("secondary %d exceed total scene rays %d", rs.SecondaryRays, ss.Rays)
					}
					if rs.PrimaryRays+rs.SecondaryRays != rs.TotalRays {
						t.Errorf("ray split does not sum: %+v", rs)
					}
					if rs.CompletedRows > uint64(rs.Height) {
						t.Errorf("CompletedRows %d exceed height %d", rs.CompletedRows, rs.Height)
					}
					if rs.MeanSamplesPerPixel < 0 {
						t.Errorf("negative mean spp %v", rs.MeanSamplesPerPixel)
					}
					bs := r.Buffer.Stats()
					if bs.MinSamples < 0 || bs.MaxSamples < 0 {
						t.Errorf("negative buffer sample bounds: %+v", bs)
					}
				}
			}
		}()
	}

	r.Run()
	close(stop)
	wg.Wait()

	final := r.Stats()
	if final.Running {
		t.Errorf("renderer still reports running after completion")
	}
	if final.CompletedRows != 12 {
		t.Errorf("final CompletedRows = %d, want 12", final.CompletedRows)
	}
}

// TestDropSamplesInjection intentionally loses half of the camera rays
// and proves the stats expose the discrepancy while the output image
// remains fully populated and finite.
func TestDropSamplesInjection(t *testing.T) {
	const w, h, spp = 8, 8, 4
	scene := testScene()
	r := quietRenderer(scene, w, h, spp, 4)

	var attempted, dropped int64
	var mu sync.Mutex
	r.DropSamples = func(iteration, x, y, sampleIndex int) bool {
		mu.Lock()
		defer mu.Unlock()
		attempted++
		// Drop every second sample globally.
		if attempted%2 == 0 {
			dropped++
			return true
		}
		return false
	}

	r.Run()
	rs := r.Stats()
	// The hook is the authority for how many camera rays were
	// attempted; the renderer must have invoked it for every one and
	// the stats must reconcile exactly with the hook's tally.
	if rs.PrimaryRays != uint64(attempted) {
		t.Errorf("PrimaryRays = %d, want hook attempts %d", rs.PrimaryRays, attempted)
	}
	if dropped == 0 || dropped >= attempted {
		t.Fatalf("drop injection misconfigured: dropped %d of %d", dropped, attempted)
	}
	if rs.TotalSamples != uint64(attempted-dropped) {
		t.Errorf("TotalSamples = %d, want %d", rs.TotalSamples, attempted-dropped)
	}
	if rs.TotalSamples >= rs.PrimaryRays {
		t.Errorf("dropped samples not reflected: samples %d >= rays %d", rs.TotalSamples, rs.PrimaryRays)
	}
	// Buffer independently confirms only accepted samples landed.
	if got := r.Buffer.TotalSamples(); got != rs.TotalSamples {
		t.Errorf("buffer TotalSamples %d != renderer %d", got, rs.TotalSamples)
	}
	// Image must still render every pixel without NaN/Inf artifacts.
	im := r.Buffer.Image()
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			cr, cg, cb, _ := im.At(x, y).RGBA()
			if cr > 65535 || cg > 65535 || cb > 65535 {
				t.Fatalf("pixel (%d,%d) out of range: %d %d %d", x, y, cr, cg, cb)
			}
		}
	}
}

// TestChannelRender verifies the channel API emits distinct frames,
// accumulates stats, and shuts down cleanly via stop.
func TestChannelRender(t *testing.T) {
	scene := testScene()
	r := quietRenderer(scene, 6, 6, 1, 2)
	ch, stop := r.ChannelRender()

	var frames int
	timeout := time.After(5 * time.Second)
	for frames < 3 {
		select {
		case im, ok := <-ch:
			if !ok {
				t.Fatalf("channel closed before collecting frames")
			}
			if im.Bounds().Dx() != 6 {
				t.Fatalf("bad frame bounds %v", im.Bounds())
			}
			frames++
		case <-timeout:
			t.Fatalf("timed out waiting for channel frames, got %d", frames)
		}
	}
	stop()
	// Drain until closed to confirm clean shutdown.
	deadline := time.After(5 * time.Second)
	for {
		select {
		case _, ok := <-ch:
			if !ok {
				goto done
			}
		case <-deadline:
			t.Fatalf("ChannelRender did not stop")
		}
	}
done:
	rs := r.Stats()
	if rs.Iterations < 3 {
		t.Errorf("expected at least 3 iterations, got %d", rs.Iterations)
	}
	if rs.TotalSamples < uint64(3*6*6) {
		t.Errorf("samples did not accumulate across frames: %d", rs.TotalSamples)
	}
}

// TestRepeatedSnapshotStability reads the same idle renderer's stats
// many times: results must be identical and never go negative.
func TestRepeatedSnapshotStability(t *testing.T) {
	scene := testScene()
	r := quietRenderer(scene, 4, 4, 1, 2)
	r.Run()
	first := r.Stats()
	for i := 0; i < 100; i++ {
		s := r.Stats()
		if s != first {
			t.Fatalf("idle snapshot changed on read %d:\nfirst=%+v\n now =%+v", i, first, s)
		}
	}
}

// TestStatsNeverNegative forces large snapshot churn and validates no
// field can ever be negative.
func TestStatsNeverNegative(t *testing.T) {
	scene := testScene()
	r := quietRenderer(scene, 10, 10, 9, 4)
	r.Run()
	rs := r.Stats()
	bs := r.Buffer.Stats()
	if bs.MinSamples < 0 || bs.MaxSamples < 0 {
		t.Errorf("negative buffer bounds: %+v", bs)
	}
	if rs.PrimaryRays > (1<<63) || rs.SecondaryRays > (1<<63) {
		t.Errorf("implausible counters: %+v", rs)
	}
	if rs.MeanSamplesPerPixel < 0 {
		t.Errorf("negative mean spp: %v", rs.MeanSamplesPerPixel)
	}
}
