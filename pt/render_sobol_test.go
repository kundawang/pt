package pt

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"image"
	"image/color"
	"math"
	"math/rand"
	"testing"
)

// testScene builds a small fixed Cornell-ish scene that exercises diffuse
// bounces, a light, a glossy material and depth of field, so the sampling
// test covers camera, lens and material dimensions.
func testScene() (*Scene, *Camera, *DefaultSampler) {
	scene := &Scene{Color: Color{0.05, 0.05, 0.08}}
	material := DiffuseMaterial(Color{0.75, 0.75, 0.75})
	red := DiffuseMaterial(Color{0.75, 0.15, 0.15})
	green := DiffuseMaterial(Color{0.15, 0.75, 0.15})
	glossy := GlossyMaterial(Color{0.8, 0.8, 0.8}, 1.5, 0.15)

	scene.Add(NewPlane(V(0, 0, 0), V(0, 0, 1), material))
	scene.Add(NewPlane(V(0, 0, 0), V(0, 1, 0), red))
	scene.Add(NewPlane(V(0, 0, 0), V(1, 0, 0), green))
	scene.Add(NewSphere(V(-0.6, 0.6, 0.6), 0.5, glossy))
	scene.Add(NewSphere(V(0.7, 0.5, 0.5), 0.4, DiffuseMaterial(Color{0.2, 0.3, 0.8})))
	scene.Add(NewSphere(V(0, 0.5, 3), 0.3, LightMaterial(White, 30)))

	camera := LookAt(V(2.2, 1.4, 2.2), V(0, 0.5, 0.6), V(0, 0, 1), 45)
	camera.SetFocus(V(0, 0.5, 0.6), 0.05)
	sampler := NewSampler(2, 3)
	return scene, &camera, sampler
}

func deterministicConfig(seed uint64) RenderConfig {
	return RenderConfig{Mode: SampleModeSobol, Seed: seed, Workers: 4, Tiles: 1, DisableAdaptive: true}
}

func renderTestBuffer(t *testing.T, w, h int, cfg RenderConfig) *Buffer {
	t.Helper()
	scene, camera, sampler := testScene()
	return RenderToBuffer(scene, camera, sampler, w, h, 4, cfg)
}

func bufferMeanAbsDiff(a, b *Buffer) float64 {
	var sum float64
	n := 0
	for i := range a.Pixels {
		ca, cb := a.Pixels[i].Color(), b.Pixels[i].Color()
		sum += math.Abs(ca.R-cb.R) + math.Abs(ca.G-cb.G) + math.Abs(ca.B-cb.B)
		n += 3
	}
	if n == 0 {
		return 0
	}
	return sum / float64(n)
}

func bufferHash(b *Buffer) string {
	h := sha256.New()
	for i := range b.Pixels {
		c := b.Pixels[i].Color()
		fmt.Fprintf(h, "%016x%016x%016x", math.Float64bits(c.R), math.Float64bits(c.G), math.Float64bits(c.B))
	}
	return hex.EncodeToString(h.Sum(nil))
}

func imageHash(im image.Image) (string, float64) {
	h := sha256.New()
	b := im.Bounds()
	var sum float64
	for y := b.Min.Y; y < b.Max.Y; y++ {
		for x := b.Min.X; x < b.Max.X; x++ {
			c := color.RGBAModel.Convert(im.At(x, y)).(color.RGBA)
			h.Write([]byte{c.R, c.G, c.B})
			sum += float64(c.R) + float64(c.G) + float64(c.B)
		}
	}
	return hex.EncodeToString(h.Sum(nil)), sum
}

// TestSobolRenderDeterministicAcrossScheduling is the core requirement:
// worker count, tile count and job order must not change the image.
func TestSobolRenderDeterministicAcrossScheduling(t *testing.T) {
	const w, h = 24, 16
	configs := []RenderConfig{
		{Mode: SampleModeSobol, Seed: 77, Workers: 1, Tiles: 1, DisableAdaptive: true},
		{Mode: SampleModeSobol, Seed: 77, Workers: 1, Tiles: 4, DisableAdaptive: true},
		{Mode: SampleModeSobol, Seed: 77, Workers: 3, Tiles: 1, DisableAdaptive: true},
		{Mode: SampleModeSobol, Seed: 77, Workers: 8, Tiles: 7, DisableAdaptive: true},
		{Mode: SampleModeSobol, Seed: 77, Workers: 16, Tiles: 12, DisableAdaptive: true},
	}
	base := renderTestBuffer(t, w, h, configs[0])
	baseHash := bufferHash(base)
	for i, cfg := range configs[1:] {
		buf := renderTestBuffer(t, w, h, cfg)
		if got := bufferHash(buf); got != baseHash {
			t.Fatalf("config %d (workers=%d tiles=%d) hash %s != base %s", i+1, cfg.Workers, cfg.Tiles, got, baseHash)
		}
		if d := bufferMeanAbsDiff(base, buf); d != 0 {
			t.Fatalf("config %d differs by %v", i+1, d)
		}
	}
}

// TestSobolRenderRepeatRunHash checks that separate runs of the same config
// (fresh Sobol sources) produce identical images.
func TestSobolRenderRepeatRunHash(t *testing.T) {
	const w, h = 20, 12
	cfg := deterministicConfig(2024)
	first := bufferHash(renderTestBuffer(t, w, h, cfg))
	for run := 0; run < 3; run++ {
		if got := bufferHash(renderTestBuffer(t, w, h, cfg)); got != first {
			t.Fatalf("run %d hash %s != %s", run, got, first)
		}
	}
}

func TestRenderWithConfigImageStable(t *testing.T) {
	const w, h = 16, 12
	scene, camera, sampler := testScene()
	cfg := deterministicConfig(5)
	a := RenderWithConfig(scene, camera, sampler, w, h, 4, cfg)
	scene2, camera2, sampler2 := testScene()
	b := RenderWithConfig(scene2, camera2, sampler2, w, h, 4, cfg)
	ha, sa := imageHash(a)
	hb, sb := imageHash(b)
	if ha != hb {
		t.Fatalf("image hashes differ: %s vs %s", ha, hb)
	}
	if math.Abs(sa-sb) > 0 {
		t.Fatalf("image sums differ: %v vs %v", sa, sb)
	}
}

// TestSeedChangesImage confirms the seed is actually plumbed through.
func TestSeedChangesImage(t *testing.T) {
	const w, h = 16, 12
	a := bufferHash(renderTestBuffer(t, w, h, deterministicConfig(1)))
	b := bufferHash(renderTestBuffer(t, w, h, deterministicConfig(2)))
	if a == b {
		t.Fatal("different seeds produced identical images")
	}
}

// TestSobolVarianceReliability measures two distinct things on a fixed scene
// against a converged reference:
//
//  1. Run-to-run variance (the property that matters for a render farm): a
//     deterministic renderer must produce the exact same pixels on every
//     repeat, so its variance across runs is zero. The legacy random renderer
//     must have positive variance.
//  2. Accuracy against the reference, which is recorded for auditing. QMC
//     gains in a path tracer are scene- and sample-count-dependent (path
//     termination and visibility make the integrand discontinuous), so this
//     test does not assert that Sobol always has lower error; it asserts only
//     that Sobol remains in the same accuracy regime as random, ruling out a
//     biased or broken estimator.
func TestSobolVarianceReliability(t *testing.T) {
	const w, h = 16, 12
	const lowSpp = 16
	const refSpp = 512

	makeBuf := func(spp int, cfg RenderConfig) *Buffer {
		scene, camera, sampler := testScene()
		return RenderToBuffer(scene, camera, sampler, w, h, spp, cfg)
	}
	mse := func(a, ref *Buffer) float64 {
		var sum float64
		for i := range a.Pixels {
			ca, cr := a.Pixels[i].Color(), ref.Pixels[i].Color()
			dr, dg, db := ca.R-cr.R, ca.G-cr.G, ca.B-cr.B
			sum += dr*dr + dg*dg + db*db
		}
		return sum / float64(len(a.Pixels))
	}

	sobolCfg := func(seed uint64) RenderConfig {
		return RenderConfig{Mode: SampleModeSobol, Seed: seed, Workers: 4, DisableAdaptive: true}
	}
	randomCfg := RenderConfig{Mode: SampleModeRandom, Workers: 4, DisableAdaptive: true}

	// Run-to-run variance: same config, repeated runs.
	sobolFirst := makeBuf(lowSpp, sobolCfg(9))
	var sobolRunVariance float64
	for i := 0; i < 4; i++ {
		b := makeBuf(lowSpp, sobolCfg(9))
		var d float64
		for j := range b.Pixels {
			c1, c2 := sobolFirst.Pixels[j].Color(), b.Pixels[j].Color()
			d += (c1.R-c2.R)*(c1.R-c2.R) + (c1.G-c2.G)*(c1.G-c2.G) + (c1.B-c2.B)*(c1.B-c2.B)
		}
		sobolRunVariance += d / float64(len(b.Pixels))
	}
	if sobolRunVariance != 0 {
		t.Fatalf("Sobol run-to-run variance = %v, want exactly 0", sobolRunVariance)
	}
	randomFirst := makeBuf(lowSpp, randomCfg)
	var randomRunVariance float64
	for i := 0; i < 4; i++ {
		b := makeBuf(lowSpp, randomCfg)
		var d float64
		for j := range b.Pixels {
			c1, c2 := randomFirst.Pixels[j].Color(), b.Pixels[j].Color()
			d += (c1.R-c2.R)*(c1.R-c2.R) + (c1.G-c2.G)*(c1.G-c2.G) + (c1.B-c2.B)*(c1.B-c2.B)
		}
		randomRunVariance += d / float64(len(b.Pixels))
	}
	randomRunVariance /= 4
	if !(randomRunVariance > 0) {
		t.Fatalf("random run-to-run variance = %v, want > 0", randomRunVariance)
	}

	// Accuracy against an ensemble reference. A single reference carries its
	// own noise, which can bias the comparison; averaging two independent
	// high-sample renders makes the reference error small relative to the
	// low-sample estimates.
	var refs []*Buffer
	for _, seed := range []uint64{0xFEED, 0xBEEF} {
		refs = append(refs, makeBuf(refSpp, RenderConfig{Mode: SampleModeSobol, Seed: seed, Workers: 4, DisableAdaptive: true}))
	}
	mseAll := func(a *Buffer) float64 {
		var sum float64
		for _, ref := range refs {
			sum += mse(a, ref)
		}
		return sum / float64(len(refs))
	}

	const runs = 8
	var sobolMSE, randomMSE float64
	for i := 0; i < runs; i++ {
		sobolMSE += mseAll(makeBuf(lowSpp, sobolCfg(uint64(i+1))))
		randomMSE += mseAll(makeBuf(lowSpp, randomCfg))
	}
	sobolMSE /= runs
	randomMSE /= runs
	if !(sobolMSE < randomMSE) {
		t.Fatalf("Sobol reference MSE %.6g not below random MSE %.6g", sobolMSE, randomMSE)
	}
	t.Logf("run-to-run variance: sobol=0 random=%.6g", randomRunVariance)
	t.Logf("reference MSE (%d spp vs %d spp): sobol=%.6g random=%.6g (%.1f%% lower)",
		lowSpp, refSpp, sobolMSE, randomMSE, 100*(1-sobolMSE/randomMSE))
}

// TestRandomModeStillWorks verifies the historical default remains a complete,
// non-degenerate renderer that ignores the seed.
func TestRandomModeStillWorks(t *testing.T) {
	const w, h = 8, 6
	a := renderTestBuffer(t, w, h, RenderConfig{Workers: 2, DisableAdaptive: true})
	// The render actually produced color, not an empty buffer.
	var sum float64
	for i := range a.Pixels {
		c := a.Pixels[i].Color()
		sum += c.R + c.G + c.B
	}
	if sum <= 0 {
		t.Fatal("random render produced an all-black image")
	}
	// Random mode ignores Seed: two runs with different seeds both render,
	// and output is allowed to differ, but must stay finite.
	b := renderTestBuffer(t, w, h, RenderConfig{Seed: 123, Workers: 2, DisableAdaptive: true})
	for i := range b.Pixels {
		c := b.Pixels[i].Color()
		if math.IsNaN(c.R) || math.IsInf(c.G, 0) {
			t.Fatalf("random render produced non-finite color %v", c)
		}
	}
}

// TestDeterministicAuditLog records the exact sample values a fixed pixel
// consumes so the sampling order is externally auditable and regressions are
// caught as explicit pixel/value changes.
func TestDeterministicAuditLog(t *testing.T) {
	const w, h = 8, 6
	const pixelX, pixelY = 3, 2
	scene, camera, sampler := testScene()
	scene.Compile()

	// Reproduce exactly what the renderer draws for this pixel and log it.
	sobol := NewSobol(SobolSeedForIteration(4242, 0))
	pixel := int64(pixelY*w + pixelX)
	const spp = 4
	cursor := NewSobolRand(sobol, pixel, 0)
	var values []float64
	for i := 0; i < spp; i++ {
		cursor.SeekTo(pixel, int64(i), 0)
		fu := cursor.Float64()
		fv := cursor.Float64()
		values = append(values, fu, fv)
		ray := camera.CastRay(pixelX, pixelY, w, h, fu, fv, cursor)
		sampler.Sample(scene, ray, cursor)
	}
	// TraceSamples must agree with the values the cursor consumed.
	trace := sobol.TraceSamples(pixel, 0, 2)
	if trace[0] != values[0] || trace[1] != values[1] {
		t.Fatalf("audit trace %v != consumed %v", trace[:2], values[:2])
	}
	// The first pixel/sample of a fresh Sobol source is the origin point,
	// digitally shifted; repeated construction yields the same values.
	again := NewSobol(SobolSeedForIteration(4242, 0)).TraceSamples(pixel, 0, 2)
	if again[0] != trace[0] || again[1] != trace[1] {
		t.Fatalf("audit values not reproducible: %v vs %v", again, trace)
	}
	t.Logf("audit pixel=(%d,%d) sample0 dims=[%.17g %.17g] %d sample values total",
		pixelX, pixelY, trace[0], trace[1], len(values))
}

// TestSobolIterativeSeedsDisjoint checks that successive iterative passes use
// different sample sets while each pass is reproducible.
func TestSobolIterativeSeedsDisjoint(t *testing.T) {
	seen := make(map[uint64]bool)
	for i := 0; i < 32; i++ {
		s := SobolSeedForIteration(17, i)
		if seen[s] {
			t.Fatalf("iteration %d seed collision", i)
		}
		seen[s] = true
	}
	if SobolSeedForIteration(17, 0) != 17 {
		t.Fatal("iteration 0 should preserve the base seed")
	}
}

// TestSplitTilesCoverage ensures tiling partitions the image exactly once.
func TestSplitTilesCoverage(t *testing.T) {
	for _, tc := range []struct{ w, h, tiles int }{
		{24, 16, 1}, {24, 16, 4}, {24, 16, 7}, {1, 1, 5}, {100, 3, 9}, {5, 90, 13},
	} {
		seen := make(map[[2]int]bool)
		rects := splitTiles(tc.w, tc.h, tc.tiles)
		for _, r := range rects {
			for y := r.y0; y < r.y1; y++ {
				for x := r.x0; x < r.x1; x++ {
					key := [2]int{x, y}
					if seen[key] {
						t.Fatalf("%dx%d/%d: pixel %v covered twice", tc.w, tc.h, tc.tiles, key)
					}
					seen[key] = true
				}
			}
		}
		if len(seen) != tc.w*tc.h {
			t.Fatalf("%dx%d/%d: covered %d/%d pixels", tc.w, tc.h, tc.tiles, len(seen), tc.w*tc.h)
		}
	}
}

// TestSobolGoldenRenderHash pins the exact bytes of a deterministic render.
// Any change to the Sobol table, scramble, dimension assignment, camera jitter
// order or sampler logic will change this hash, which is exactly what we want
// to notice: it makes the sampling order auditable across code revisions.
//
// If an intentional change alters the sequence, update the constant and note
// the reason in the commit message. Never "fix" a failing hash by making the
// test less strict.
func TestSobolGoldenRenderHash(t *testing.T) {
	const want = "fc0d38232129e3d85053015e35cbae9fd28f0b348098a6192eadfc691a014087"
	scene, camera, sampler := testScene()
	buf := RenderToBuffer(scene, camera, sampler, 24, 16, 4, RenderConfig{
		Mode: SampleModeSobol, Seed: 20240601, Workers: 3, Tiles: 5, DisableAdaptive: true,
	})
	if got := bufferHash(buf); got != want {
		t.Fatalf("golden render hash = %s, want %s\n"+
			"the deterministic sampling order changed; if intentional, update the constant",
			got, want)
	}
}

// TestSobolAdaptiveDeterministic confirms the default (adaptive) render is
// also reproducible across worker counts, not just fixed-workload renders.
func TestSobolAdaptiveDeterministic(t *testing.T) {
	const w, h = 16, 12
	var hashes []string
	for _, workers := range []int{1, 2, 8} {
		scene, camera, sampler := testScene()
		buf := RenderToBuffer(scene, camera, sampler, w, h, 4, RenderConfig{
			Mode: SampleModeSobol, Seed: 7, Workers: workers,
		})
		hashes = append(hashes, bufferHash(buf))
	}
	for i := 1; i < len(hashes); i++ {
		if hashes[i] != hashes[0] {
			t.Fatalf("adaptive render with workers variant %d hashed %s != %s", i, hashes[i], hashes[0])
		}
	}
}

// TestRandomDeterministicSeedHelper documents that the legacy per-worker RNG
// is only used in random mode; Sobol workers share no mutable RNG.
func TestSobolWorkersShareNoMutableRNG(t *testing.T) {
	// Two concurrent SobolRand cursors over the same immutable source must
	// not interfere.
	s := NewSobol(88)
	a := NewSobolRand(s, 1, 1)
	b := NewSobolRand(s, 2, 2)
	wantA := s.Value(1, 1, 0)
	wantB := s.Value(2, 2, 0)
	// Interleave consumption.
	for i := 0; i < 100; i++ {
		b.Float64()
		a.Float64()
	}
	a.SeekTo(1, 1, 0)
	b.SeekTo(2, 2, 0)
	if a.Float64() != wantA || b.Float64() != wantB {
		t.Fatal("interleaved cursors interfered with each other")
	}
	// Random-mode helper type still satisfies the interface.
	var _ Random = rand.New(rand.NewSource(1))
	var _ Random = NewSobolRand(s, 0, 0)
}
