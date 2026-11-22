package safety

import (
	"math/rand"
	"testing"

	"github.com/fogleman/pt/pt"
)

// TestPixelStability renders the validated regression cube through two
// independent load paths — the raw pt.LoadOBJ and SafeLoadOBJ — and asserts
// that every deterministic ray produces the identical hit and shading result.
// The full renderer seeds its RNG from the wall clock, so instead this test
// drives a fixed grid of camera rays with an explicit fixed-seed RNG, which
// isolates geometry correctness from sampling noise. Hardening must not move
// a single pixel of a well-formed scene.
func TestPixelStability(t *testing.T) {
	path := testdata(t, "regression-cube.obj")
	limits := DefaultLimits()

	// Reference path: the original, unhardened loader.
	rawMesh, err := pt.LoadOBJ(path, pt.DiffuseMaterial(pt.HexColor(0xB9121B)))
	if err != nil {
		t.Fatal(err)
	}
	rawMesh.Compile()

	// Hardened path.
	safeMesh, rep, err := SafeLoadOBJ(path, pt.DiffuseMaterial(pt.HexColor(0xB9121B)), limits)
	if err != nil {
		t.Fatalf("safe loader rejected valid cube: %v (%s)", err, rep.Stage)
	}
	safeMesh.Compile()

	camera := pt.LookAt(
		pt.Vector{X: 2.5, Y: 2, Z: -3},
		pt.Vector{X: 0.5, Y: 0.5, Z: 0.5},
		pt.Vector{X: 0, Y: 1, Z: 0},
		45)

	const w, h = 24, 18
	rnd := rand.New(rand.NewSource(99))
	mismatches := 0
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			ray := camera.CastRay(x, y, w, h, 0.5, 0.5, rnd)
			h1 := rawMesh.Intersect(ray)
			h2 := safeMesh.Intersect(ray)
			if !hitsEqual(h1, h2) {
				mismatches++
				if mismatches <= 5 {
					t.Errorf("pixel (%d,%d) diverged: raw T=%v safe T=%v",
						x, y, infOrT(h1), infOrT(h2))
				}
			}
			if h1.Ok() {
				// Shading inputs must also match: normals and materials.
				info1 := h1.Info(ray)
				info2 := h2.Info(ray)
				if !vectorsEqual(info1.Normal, info2.Normal) {
					t.Errorf("pixel (%d,%d) normal divergence: %v vs %v",
						x, y, info1.Normal, info2.Normal)
				}
				if info1.Material != info2.Material {
					t.Errorf("pixel (%d,%d) material divergence", x, y)
				}
			}
		}
	}
	if mismatches > 0 {
		t.Fatalf("%d/%d pixels changed after hardening a valid scene", mismatches, w*h)
	}
}

// TestSceneLevelStability places both meshes in full pt.Scene trees and
// compares scene-level intersection (k-d traversal), proving the tree built
// on hardened geometry is behaviorally identical.
func TestSceneLevelStability(t *testing.T) {
	path := testdata(t, "regression-cube.obj")
	m1, _ := pt.LoadOBJ(path, pt.SpecularMaterial(pt.White, 2))
	m2, _, err := SafeLoadOBJ(path, pt.SpecularMaterial(pt.White, 2), DefaultLimits())
	if err != nil {
		t.Fatal(err)
	}
	s1, s2 := pt.Scene{}, pt.Scene{}
	s1.Add(m1)
	s2.Add(m2)
	s1.Compile()
	s2.Compile()

	rnd := rand.New(rand.NewSource(7))
	for i := 0; i < 500; i++ {
		origin := pt.Vector{
			X: rnd.Float64()*6 - 3,
			Y: rnd.Float64()*6 - 2,
			Z: rnd.Float64()*6 - 4,
		}
		dir := pt.RandomUnitVector(rnd)
		ray := pt.Ray{Origin: origin, Direction: dir}
		h1 := s1.Intersect(ray)
		h2 := s2.Intersect(ray)
		if !hitsEqual(h1, h2) {
			t.Fatalf("scene ray %d diverged: %v vs %v", i, infOrT(h1), infOrT(h2))
		}
	}
}

func hitsEqual(a, b pt.Hit) bool {
	if a.Ok() != b.Ok() {
		return false
	}
	if !a.Ok() {
		return true // both miss
	}
	return floatsEqual(a.T, b.T)
}

func infOrT(h pt.Hit) float64 {
	if !h.Ok() {
		return -1
	}
	return h.T
}

func floatsEqual(a, b float64) bool {
	if a != a || b != b {
		return false
	}
	d := a - b
	if d < 0 {
		d = -d
	}
	scale := a
	if scale < 0 {
		scale = -scale
	}
	if scale < 1 {
		scale = 1
	}
	return d <= scale*1e-12
}

func vectorsEqual(a, b pt.Vector) bool {
	return floatsEqual(a.X, b.X) && floatsEqual(a.Y, b.Y) && floatsEqual(a.Z, b.Z)
}
