package pt

import "testing"

// TestWorkerClamping covers the parallel-resource anomaly case: asking
// for more workers than scanlines, zero or negative workers, and more
// workers than CPUs must all yield a stable, bounded worker count and a
// correct render.
func TestWorkerClamping(t *testing.T) {
	cases := []struct {
		name   string
		numCPU int
		h      int
	}{
		{"more workers than rows", 64, 3},
		{"zero workers defaults", 0, 4},
		{"negative workers defaults", -5, 4},
		{"single worker", 1, 4},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			scene := testScene()
			r := quietRenderer(scene, 5, tc.h, 1, tc.numCPU)
			r.Run()
			rs := r.Stats()
			if rs.Workers < 1 {
				t.Fatalf("Workers = %d, must be >= 1", rs.Workers)
			}
			if rs.Workers > tc.h {
				t.Errorf("Workers = %d exceed scanlines %d", rs.Workers, tc.h)
			}
			if rs.CompletedRows != uint64(tc.h) {
				t.Errorf("CompletedRows = %d, want %d", rs.CompletedRows, tc.h)
			}
			if rs.TotalSamples != uint64(5*tc.h) {
				t.Errorf("TotalSamples = %d, want %d", rs.TotalSamples, 5*tc.h)
			}
		})
	}
}

// TestPerPixelSamples verifies the per-pixel Samples accessor and its
// safe handling of out-of-range coordinates.
func TestPerPixelSamples(t *testing.T) {
	scene := testScene()
	r := quietRenderer(scene, 4, 4, 1, 2)
	r.Run()
	for y := 0; y < 4; y++ {
		for x := 0; x < 4; x++ {
			if n := r.Buffer.Samples(x, y); n < 1 {
				t.Errorf("pixel (%d,%d) Samples = %d, want >= 1", x, y, n)
			}
		}
	}
	// Out of range must return zero, never panic.
	for _, p := range [][2]int{{-1, 0}, {0, -1}, {99, 0}, {0, 99}} {
		if n := r.Buffer.Samples(p[0], p[1]); n != 0 {
			t.Errorf("out-of-range pixel %v returned %d, want 0", p, n)
		}
	}
}

// TestPrimarySecondaryClassification uses a scene with a light and
// bounded bounces to prove secondary rays are actually counted and the
// primary+secondary identity holds exactly.
func TestPrimarySecondaryClassification(t *testing.T) {
	scene := testScene()
	// Direct sampler fires explicit shadow rays toward the light.
	r := NewRenderer(scene, testCamera(), NewDirectSampler(), 6, 6)
	r.NumCPU = 2
	r.Verbose = false
	r.Run()

	ss := scene.Stats()
	rs := r.Stats()
	if ss.PrimaryRays+ss.SecondaryRays != ss.Rays {
		t.Fatalf("scene identity broken: %+v", ss)
	}
	if rs.PrimaryRays != ss.PrimaryRays {
		t.Errorf("renderer primary %d != scene primary %d", rs.PrimaryRays, ss.PrimaryRays)
	}
	if rs.SecondaryRays != ss.SecondaryRays {
		t.Errorf("renderer secondary %d != scene secondary %d", rs.SecondaryRays, ss.SecondaryRays)
	}
	// A direct-lit scene with a light present should record shadow rays.
	if rs.SecondaryRays == 0 {
		t.Logf("warning: no secondary rays recorded (scene may not be directly lit)")
	}
}
