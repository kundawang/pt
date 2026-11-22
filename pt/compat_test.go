package pt

import (
	"math/rand"
	"os"
	"path/filepath"
	"testing"
)

// TestLegacyIterativeRenderAPI exercises the original package-level
// IterativeRender signature, including on-disk PNG output, proving the
// historical API and default image/log behavior are preserved.
func TestLegacyIterativeRenderAPI(t *testing.T) {
	dir := t.TempDir()
	scene := testScene()
	cam := testCamera()
	sampler := NewSampler(1, 1)
	path := filepath.Join(dir, "frame%02d.png")

	IterativeRender(path, 2, scene, cam, sampler, 6, 6, 1)

	for i := 1; i <= 2; i++ {
		p := filepath.Join(dir, "frame01.png")
		if i == 2 {
			p = filepath.Join(dir, "frame02.png")
		}
		if _, err := os.Stat(p); err != nil {
			t.Fatalf("expected output %s: %v", p, err)
		}
	}
}

// panicSampler blows up on every call, simulating an abnormal sampler
// failure; pixel output must survive and stats must stay consistent.
type panicSampler struct{}

func (panicSampler) Sample(scene *Scene, ray Ray, rnd *rand.Rand) Color {
	panic("simulated sampler failure")
}

// TestSamplerPanicIsContained proves a repeatedly panicking sampler
// neither crashes the render nor corrupts pixel output: every pixel
// simply stays black and zero samples are recorded.
func TestSamplerPanicIsContained(t *testing.T) {
	const w, h = 5, 5
	scene := testScene()
	r := NewRenderer(scene, testCamera(), panicSampler{}, w, h)
	r.NumCPU = 3
	r.Verbose = false

	r.Run() // must not panic

	rs := r.Stats()
	if rs.TotalSamples != 0 {
		t.Errorf("panicked samples must not be written, got %d", rs.TotalSamples)
	}
	if rs.PrimaryRays != uint64(w*h) {
		t.Errorf("attempted PrimaryRays = %d, want %d", rs.PrimaryRays, w*h)
	}
	im := r.Buffer.Image()
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			cr, cg, cb, _ := im.At(x, y).RGBA()
			if cr != 0 || cg != 0 || cb != 0 {
				t.Fatalf("pixel (%d,%d) non-black after total sampler failure", x, y)
			}
		}
	}
}
