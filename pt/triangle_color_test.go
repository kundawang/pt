package pt

import (
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"testing"
)

// ---------------------------------------------------------------------------
// Test helpers
// ---------------------------------------------------------------------------

var (
	testRed     = Color{1, 0, 0}
	testGreen   = Color{0, 1, 0}
	testBlue    = Color{0, 0, 1}
	testWhite   = Color{1, 1, 1}
	testMagenta = Color{1, 0, 1}
)

func finiteColor(c Color) bool {
	return !math.IsNaN(c.R) && !math.IsInf(c.R, 0) &&
		!math.IsNaN(c.G) && !math.IsInf(c.G, 0) &&
		!math.IsNaN(c.B) && !math.IsInf(c.B, 0)
}

func assertFinite(t *testing.T, c Color, label string) {
	t.Helper()
	if !finiteColor(c) {
		t.Fatalf("%s is not finite: %+v", label, c)
	}
}

func assertFiniteVector(t *testing.T, v Vector, label string) {
	t.Helper()
	if math.IsNaN(v.X) || math.IsInf(v.X, 0) ||
		math.IsNaN(v.Y) || math.IsInf(v.Y, 0) ||
		math.IsNaN(v.Z) || math.IsInf(v.Z, 0) {
		t.Fatalf("%s is not finite: %+v", label, v)
	}
}

func colorsClose(a, b Color, eps float64) bool {
	return math.Abs(a.R-b.R) <= eps &&
		math.Abs(a.G-b.G) <= eps &&
		math.Abs(a.B-b.B) <= eps
}

func assertColor(t *testing.T, got, want Color, eps float64, label string) {
	t.Helper()
	assertFinite(t, got, label)
	if !colorsClose(got, want, eps) {
		t.Fatalf("%s = %+v, want %+v (eps %g)", label, got, want, eps)
	}
}

// albedoSampler returns the resolved material color at the first hit, so a
// render reproduces the per-point MaterialAt result deterministically with
// one sample per pixel (no random bounces).
type albedoSampler struct{}

func (albedoSampler) Sample(scene *Scene, ray Ray, rnd *rand.Rand) Color {
	hit := scene.Intersect(ray)
	if !hit.Ok() {
		return Black
	}
	return hit.Info(ray).Material.Color
}

// referenceBary is an independent barycentric implementation based on
// signed triangle areas (cross products projected onto the triangle
// normal), rather than the Gram-matrix solve used by Triangle.Barycentric.
// It returns weights in the order (w1, w2, w3), matching corners
// (V1, V2, V3); weights are signed, so exterior points get negatives.
func referenceBary(t *Triangle, p Vector) (w1, w2, w3 float64) {
	n := t.V2.Sub(t.V1).Cross(t.V3.Sub(t.V1))
	den := n.Dot(n)
	a1 := t.V2.Sub(p).Cross(t.V3.Sub(p)).Dot(n) / den
	a2 := t.V3.Sub(p).Cross(t.V1.Sub(p)).Dot(n) / den
	a3 := t.V1.Sub(p).Cross(t.V2.Sub(p)).Dot(n) / den
	return a1, a2, a3
}

// referenceVertexColor mirrors Triangle.VertexColorAt using the independent
// barycentric reference above, including the material-color fallback.
func referenceVertexColor(t *Triangle, p Vector) Color {
	w1, w2, w3 := referenceBary(t, p)
	base := t.Material.Color
	c1, c2, c3 := base, base, base
	if t.c1 {
		c1 = t.C1
	}
	if t.c2 {
		c2 = t.C2
	}
	if t.c3 {
		c3 = t.C3
	}
	c := Color{}
	c = c.Add(c1.MulScalar(w1))
	c = c.Add(c2.MulScalar(w2))
	c = c.Add(c3.MulScalar(w3))
	return c
}

// solidTexture is a procedural Texture so tests need no image files.
type solidTexture struct{ c Color }

func (s solidTexture) Sample(u, v float64) Color { return s.c }
func (s solidTexture) NormalSample(u, v float64) Vector {
	return Vector{0, 0, 1}
}
func (s solidTexture) BumpSample(u, v float64) Vector { return Vector{} }
func (s solidTexture) Pow(a float64) Texture          { return s }
func (s solidTexture) MulScalar(a float64) Texture {
	return solidTexture{s.c.MulScalar(a)}
}

var _ Texture = solidTexture{}

// quadColors are the four corner colors of the test square:
// (0,0) red, (1,0) green, (1,1) blue, (0,1) white.
var quadColors = [4]Color{
	{1, 0, 0}, // V00
	{0, 1, 0}, // V10
	{0, 0, 1}, // V11
	{1, 1, 1}, // V01
}

// coloredQuad builds a unit square in the z=0 plane split along diagonal
// V00->V11 into two vertex-colored triangles.
func coloredQuad() [2]*Triangle {
	v00 := Vector{0, 0, 0}
	v10 := Vector{1, 0, 0}
	v11 := Vector{1, 1, 0}
	v01 := Vector{0, 1, 0}
	mat := DiffuseMaterial(testMagenta)
	t1 := NewTriangle(v00, v10, v11, Vector{}, Vector{}, Vector{}, mat)
	t1.SetVertexColors(quadColors[0], quadColors[1], quadColors[2])
	t2 := NewTriangle(v00, v11, v01, Vector{}, Vector{}, Vector{}, mat)
	t2.SetVertexColors(quadColors[0], quadColors[2], quadColors[3])
	return [2]*Triangle{t1, t2}
}

// analyticQuadColor evaluates the expected linear barycentric color at point
// (x, y) using referenceBary on the appropriate triangle.
func analyticQuadColor(tri [2]*Triangle, x, y float64) Color {
	p := Vector{x, y, 0}
	if w1, w2, w3 := referenceBary(tri[0], p); w2 >= -1e-12 && w3 >= -1e-12 && w1 >= -1e-12 {
		c := Color{}
		c = c.Add(quadColors[0].MulScalar(w1))
		c = c.Add(quadColors[1].MulScalar(w2))
		c = c.Add(quadColors[2].MulScalar(w3))
		return c
	}
	w1, w2, w3 := referenceBary(tri[1], p)
	c := Color{}
	c = c.Add(quadColors[0].MulScalar(w1))
	c = c.Add(quadColors[2].MulScalar(w2))
	c = c.Add(quadColors[3].MulScalar(w3))
	return c
}

// quadCamera frames the unit square orthogonally with margin around the
// borders so every camera ray lands inside the square.
func quadCamera() Camera {
	return LookAt(Vector{0.5, 0.5, 3}, Vector{0.5, 0.5, 0}, Vector{0, 1, 0}, 45)
}

// renderAlbedo renders a scene with one deterministic sample per pixel and
// returns the linear-space pixel colors.
func renderAlbedo(t *testing.T, scene *Scene, camera *Camera, w, h int) []Color {
	t.Helper()
	scene.Compile()
	buf := NewBuffer(w, h)
	rnd := rand.New(rand.NewSource(1))
	sampler := albedoSampler{}
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			ray := camera.CastRay(x, y, w, h, 0.5, 0.5, rnd)
			buf.AddSample(x, y, sampler.Sample(scene, ray, rnd))
		}
	}
	out := make([]Color, w*h)
	for i := range buf.Pixels {
		c := buf.Pixels[i].Color()
		assertFinite(t, c, "rendered pixel")
		out[i] = c
	}
	return out
}

// ---------------------------------------------------------------------------
// 1. Four-color square against the independent barycentric reference
// ---------------------------------------------------------------------------

func TestTriangleVertexColorBarycentricReference(t *testing.T) {
	tri := coloredQuad()
	rnd := rand.New(rand.NewSource(42))
	for _, tr := range tri {
		// random interior points (convex combinations of the corners)
		for i := 0; i < 500; i++ {
			a := rnd.Float64()
			b := rnd.Float64() * (1 - a)
			c := 1 - a - b
			p := Vector{}
			p = p.Add(tr.V1.MulScalar(a))
			p = p.Add(tr.V2.MulScalar(b))
			p = p.Add(tr.V3.MulScalar(c))
			got := tr.VertexColorAt(p)
			want := referenceVertexColor(tr, p)
			assertColor(t, got, want, 1e-12, "interior vertex color")
		}
		// exact corners must return the exact per-vertex color
		assertColor(t, tr.VertexColorAt(tr.V1), tr.C1, 0, "V1 color")
		assertColor(t, tr.VertexColorAt(tr.V2), tr.C2, 0, "V2 color")
		assertColor(t, tr.VertexColorAt(tr.V3), tr.C3, 0, "V3 color")
		// edge midpoints
		mid := tr.V1.Add(tr.V2).MulScalar(0.5)
		want := tr.C1.MulScalar(0.5).Add(tr.C2.MulScalar(0.5))
		assertColor(t, tr.VertexColorAt(mid), want, 1e-12, "edge midpoint color")
	}
}

func TestTriangleVertexColorQuadRender(t *testing.T) {
	tri := coloredQuad()
	scene := Scene{}
	for _, tr := range tri {
		scene.Add(tr)
	}
	const w, h = 48, 48
	camera := quadCamera()
	pixels := renderAlbedo(t, &scene, &camera, w, h)
	hit := 0
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			ray := camera.CastRay(x, y, w, h, 0.5, 0.5, rand.New(rand.NewSource(1)))
			h2 := scene.Intersect(ray)
			if !h2.Ok() {
				continue
			}
			hit++
			p := ray.Position(h2.T)
			want := analyticQuadColor(tri, p.X, p.Y)
			assertColor(t, pixels[y*w+x], want, 1e-10, "quad pixel")
		}
	}
	if hit == 0 {
		t.Fatal("no pixels hit the quad")
	}
}

// ---------------------------------------------------------------------------
// 2. Shared edge: no seam between adjacent triangles
// ---------------------------------------------------------------------------

func TestVertexColorSharedEdgeSeam(t *testing.T) {
	tri := coloredQuad()
	t1, t2 := tri[0], tri[1]
	// The diagonal V00->V11 is shared: both triangles must interpolate to
	// exactly the same color at every point along it.
	rnd := rand.New(rand.NewSource(7))
	for i := 0; i < 1000; i++ {
		f := rnd.Float64()
		p := Vector{f, f, 0}
		c1 := t1.VertexColorAt(p)
		c2 := t2.VertexColorAt(p)
		assertColor(t, c1, c2, 1e-12, "shared edge color mismatch")
		// and it must match the independent linear reference on the edge
		want := quadColors[0].MulScalar(1 - f).Add(quadColors[2].MulScalar(f))
		assertColor(t, c1, want, 1e-12, "shared edge reference")
	}
	// pixel-level: adjacent pixels on either side of the diagonal must not
	// jump beyond the analytic gradient (a seam would show up as a spike).
	scene := Scene{}
	scene.Add(t1)
	scene.Add(t2)
	const w, h = 64, 64
	camera := quadCamera()
	pixels := renderAlbedo(t, &scene, &camera, w, h)
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			ray := camera.CastRay(x, y, w, h, 0.5, 0.5, rand.New(rand.NewSource(1)))
			h2 := scene.Intersect(ray)
			if !h2.Ok() {
				continue
			}
			p := ray.Position(h2.T)
			want := analyticQuadColor(tri, p.X, p.Y)
			assertColor(t, pixels[y*w+x], want, 1e-9, "seam pixel")
		}
	}
}

// ---------------------------------------------------------------------------
// 3. Texture / NormalTexture / Emittance / transparency composition
// ---------------------------------------------------------------------------

func TestVertexColorComposition(t *testing.T) {
	v1, v2, v3 := Vector{0, 0, 0}, Vector{1, 0, 0}, Vector{0, 1, 0}
	p := Vector{0.25, 0.25, 0} // weights (0.5, 0.25, 0.25)

	// no vertex colors, no texture: material constant is unchanged
	plain := NewTriangle(v1, v2, v3, Vector{}, Vector{}, Vector{}, DiffuseMaterial(testMagenta))
	if plain.HasVertexColors() {
		t.Fatal("new triangle should not have vertex colors by default")
	}
	assertColor(t, MaterialAt(plain, p).Color, testMagenta, 0, "plain material color")

	// vertex colors only: interpolated color replaces the constant
	vc := NewTriangle(v1, v2, v3, Vector{}, Vector{}, Vector{}, DiffuseMaterial(testMagenta))
	vc.SetVertexColors(testRed, testGreen, testBlue)
	wantVC := testRed.MulScalar(0.5).Add(testGreen.MulScalar(0.25)).Add(testBlue.MulScalar(0.25))
	assertColor(t, MaterialAt(vc, p).Color, wantVC, 1e-12, "vertex color material")

	// texture only: texture sample replaces the constant (legacy behavior)
	grey := Color{0.5, 0.5, 0.5}
	texOnly := NewTriangle(v1, v2, v3, Vector{}, Vector{}, Vector{}, DiffuseMaterial(testMagenta))
	texOnly.Material.Texture = solidTexture{grey}
	assertColor(t, MaterialAt(texOnly, p).Color, grey, 0, "texture-only material")

	// vertex colors + texture: texture modulates the vertex color
	both := NewTriangle(v1, v2, v3, Vector{}, Vector{}, Vector{}, DiffuseMaterial(testMagenta))
	both.SetVertexColors(testRed, testGreen, testBlue)
	both.Material.Texture = solidTexture{grey}
	assertColor(t, MaterialAt(both, p).Color, wantVC.Mul(grey), 1e-12, "vertex color * texture")

	// NormalTexture is orthogonal to color: vertex color is kept untouched
	normalMat := DiffuseMaterial(testMagenta)
	normalMat.NormalTexture = solidTexture{testWhite}
	nvc := NewTriangle(v1, v2, v3, Vector{}, Vector{}, Vector{}, normalMat)
	nvc.SetVertexColors(testRed, testGreen, testBlue)
	assertColor(t, MaterialAt(nvc, p).Color, wantVC, 1e-12, "normal texture keeps vertex color")

	// emittance is unchanged; the emitted color is Emittance * vertex color
	light := NewTriangle(v1, v2, v3, Vector{}, Vector{}, Vector{}, LightMaterial(testWhite, 30))
	light.SetVertexColors(testRed, testGreen, testBlue)
	lm := MaterialAt(light, p)
	if lm.Emittance != 30 {
		t.Fatalf("emittance = %v, want 30", lm.Emittance)
	}
	assertColor(t, lm.Color, wantVC, 1e-12, "light vertex color")

	// transparent material keeps its flags and tints with the vertex color
	trans := NewTriangle(v1, v2, v3, Vector{}, Vector{}, Vector{},
		TransparentMaterial(testMagenta, 1.5, 0, 0.5))
	trans.SetVertexColors(testRed, testGreen, testBlue)
	tm := MaterialAt(trans, p)
	if !tm.Transparent {
		t.Fatal("transparent flag lost with vertex colors")
	}
	assertColor(t, tm.Color, wantVC, 1e-12, "transparent vertex color")
}

// ---------------------------------------------------------------------------
// 4. Energy conservation: average render equals the analytic surface integral
// ---------------------------------------------------------------------------

func TestVertexColorEnergy(t *testing.T) {
	tri := coloredQuad()

	// exact integral over a single triangle: average color is the mean of
	// its three vertex colors (linear basis functions).
	mean1 := tri[0].C1.Add(tri[0].C2).Add(tri[0].C3).DivScalar(3)
	grid := 200
	sum := Color{}
	n := 0
	for i := 0; i < grid; i++ {
		for j := 0; j < grid; j++ {
			x := (float64(i) + 0.5) / float64(grid)
			y := (float64(j) + 0.5) / float64(grid)
			p := Vector{x, y, 0}
			if w1, w2, w3 := referenceBary(tri[0], p); w2 >= 0 && w3 >= 0 {
				sum = sum.Add(tri[0].VertexColorAt(p))
				n++
				_ = w1
			}
		}
	}
	assertColor(t, sum.DivScalar(float64(n)), mean1, 5e-3, "triangle integral")

	// rendered energy over the whole quad: both triangles cover equal area,
	// so the expected mean is the average of the two triangle means.
	mean2 := tri[1].C1.Add(tri[1].C2).Add(tri[1].C3).DivScalar(3)
	want := mean1.Add(mean2).DivScalar(2)

	scene := Scene{}
	scene.Add(tri[0])
	scene.Add(tri[1])
	const w, h = 96, 96
	camera := quadCamera()
	pixels := renderAlbedo(t, &scene, &camera, w, h)
	rsum, count := Color{}, 0
	for _, c := range pixels {
		// background pixels are black and excluded from the surface mean
		if c == (Color{}) {
			continue
		}
		rsum = rsum.Add(c)
		count++
	}
	if count == 0 {
		t.Fatal("no surface pixels rendered")
	}
	got := rsum.DivScalar(float64(count))
	assertColor(t, got, want, 0.02, "rendered energy")
}

// ---------------------------------------------------------------------------
// 5. Robustness: degenerate triangles, missing colors, non-finite colors,
//    out-of-range values, mirrored transforms and back-face hits
// ---------------------------------------------------------------------------

func TestVertexColorDegenerate(t *testing.T) {
	// all three corners collinear (and one case fully coincident)
	degen := NewTriangle(
		Vector{0, 0, 0}, Vector{1, 0, 0}, Vector{2, 0, 0},
		Vector{}, Vector{}, Vector{}, DiffuseMaterial(testMagenta))
	degen.SetVertexColors(testRed, testGreen, testBlue)
	for _, p := range []Vector{{0, 0, 0}, {1, 0, 0}, {0.5, 0, 0}, {3, 7, 0}} {
		c := degen.VertexColorAt(p)
		assertFinite(t, c, "degenerate color")
	}
	coincident := NewTriangle(
		Vector{1, 1, 1}, Vector{1, 1, 1}, Vector{1, 1, 1},
		Vector{}, Vector{}, Vector{}, DiffuseMaterial(testMagenta))
	coincident.SetVertexColors(testRed, testGreen, testBlue)
	c := coincident.VertexColorAt(Vector{1, 1, 1})
	assertFinite(t, c, "coincident color")
	// degenerate barycentric weights are undefined: equal-weight average is
	// the documented stable fallback
	want := testRed.Add(testGreen).Add(testBlue).DivScalar(3)
	assertColor(t, c, want, 1e-12, "coincident equal-weight fallback")
}

func TestVertexColorMissingVertices(t *testing.T) {
	v1, v2, v3 := Vector{0, 0, 0}, Vector{1, 0, 0}, Vector{0, 1, 0}
	tr := NewTriangle(v1, v2, v3, Vector{}, Vector{}, Vector{},
		DiffuseMaterial(Color{0.2, 0.4, 0.6}))
	if tr.HasVertexColors() {
		t.Fatal("triangle must start without vertex colors")
	}
	// only vertex 1 gets a color; the other corners use the material color
	tr.SetVertexColor(1, testRed)
	if !tr.HasVertexColors() {
		t.Fatal("triangle should report vertex colors after SetVertexColor")
	}
	c := tr.VertexColorAt(tr.V2)
	assertColor(t, c, testRed, 0, "explicit vertex color")
	c = tr.VertexColorAt(tr.V1)
	assertColor(t, c, tr.Material.Color, 0, "missing vertex falls back to material")
	// invalid indices are ignored
	tr.SetVertexColor(-1, testWhite)
	tr.SetVertexColor(3, testWhite)
	if tr.c1 || tr.c3 {
		t.Fatal("out-of-range SetVertexColor must be a no-op")
	}
	// clearing restores legacy behavior
	tr.ClearVertexColors()
	if tr.HasVertexColors() {
		t.Fatal("ClearVertexColors did not clear")
	}
	assertColor(t, MaterialAt(tr, tr.V1).Color, tr.Material.Color, 0,
		"cleared triangle uses material color")
}

func TestVertexColorSanitization(t *testing.T) {
	v1, v2, v3 := Vector{0, 0, 0}, Vector{1, 0, 0}, Vector{0, 1, 0}
	tr := NewTriangle(v1, v2, v3, Vector{}, Vector{}, Vector{},
		DiffuseMaterial(testMagenta))
	bad := Color{math.NaN(), math.Inf(1), math.Inf(-1)}
	high := Color{2, -1, 0.5}
	tr.SetVertexColors(bad, high, testBlue)
	// stored values are finite and clamped to [0, 1]
	for _, c := range [3]Color{tr.C1, tr.C2, tr.C3} {
		assertFinite(t, c, "stored vertex color")
		if c.R < 0 || c.R > 1 || c.G < 0 || c.G > 1 || c.B < 0 || c.B > 1 {
			t.Fatalf("stored color out of range: %+v", c)
		}
	}
	assertColor(t, tr.C1, Color{0, 0, 0}, 0, "non-finite -> 0")
	assertColor(t, tr.C2, Color{1, 0, 0.5}, 0, "out-of-range clamped")
	// interpolation output can never carry NaN either
	c := tr.VertexColorAt(Vector{0.25, 0.25, 0})
	assertFinite(t, c, "interpolated color")
	if c.R < 0 || c.R > 1 || c.G < 0 || c.G > 1 || c.B < 0 || c.B > 1 {
		t.Fatalf("interpolated color out of range: %+v", c)
	}
}

func TestVertexColorMirrorAndBackface(t *testing.T) {
	v1, v2, v3 := Vector{0, 0, 0}, Vector{1, 0, 0}, Vector{0, 1, 0}
	tr := NewTriangle(v1, v2, v3, Vector{}, Vector{}, Vector{},
		DiffuseMaterial(testMagenta))
	tr.SetVertexColors(testRed, testGreen, testBlue)

	// mirror transform: negative determinant flips winding order, but the
	// corner-to-color association must remain unchanged.
	mirrored := NewTransformedShape(tr, Scale(Vector{-1, 1, 1}))
	// world point (-0.25, 0.25) maps back to object point (0.25, 0.25),
	// whose weights are (0.5, 0.25, 0.25). Hit it from both sides.
	want := testRed.MulScalar(0.5).Add(testGreen.MulScalar(0.25)).Add(testBlue.MulScalar(0.25))
	for _, ray := range []Ray{
		{Vector{-0.25, 0.25, 1}, Vector{0, 0, -1}}, // front side
		{Vector{-0.25, 0.25, -1}, Vector{0, 0, 1}}, // back side
	} {
		hit := mirrored.Intersect(ray)
		if !hit.Ok() {
			t.Fatalf("mirrored triangle not hit from %v", ray.Origin)
		}
		info := hit.Info(ray)
		assertColor(t, info.Material.Color, want, 1e-12, "mirror/backface color")
		assertFiniteVector(t, info.Normal, "mirror/backface normal")
	}

	// without the mirror, the same rays on the untransformed triangle give
	// the same interpolated color, proving the mirror did not permute corners
	for _, ray := range []Ray{
		{Vector{0.25, 0.25, 1}, Vector{0, 0, -1}},
		{Vector{0.25, 0.25, -1}, Vector{0, 0, 1}},
	} {
		hit := tr.Intersect(ray)
		if !hit.Ok() {
			t.Fatalf("plain triangle not hit from %v", ray.Origin)
		}
		assertColor(t, hit.Info(ray).Material.Color, want, 1e-12, "plain-side color")
	}
}

func TestVertexColorMeshTransformAndSharing(t *testing.T) {
	tris := coloredQuad()
	original := tris[0]
	origColors := [3]Color{original.C1, original.C2, original.C3}
	origCorners := [3]Vector{original.V1, original.V2, original.V3}

	mesh := NewMesh(tris[:])
	// transforms move geometry only; colors stay attached to their corner
	mesh.Transform(Translate(Vector{10, -5, 3}).Mul(Scale(Vector{2, 2, 2})))
	for i, want := range origColors {
		var got Color
		switch i {
		case 0:
			got = tris[0].C1
		case 1:
			got = tris[0].C2
		case 2:
			got = tris[0].C3
		}
		assertColor(t, got, want, 0, "color order after transform")
	}
	// geometry did move
	if tris[0].V1 == origCorners[0] {
		t.Fatal("transform did not move vertices")
	}

	// Copy duplicates triangles including colors but does not share structs
	dup := mesh.Copy()
	if len(dup.Triangles) != len(mesh.Triangles) {
		t.Fatal("mesh copy lost triangles")
	}
	for i, src := range mesh.Triangles {
		dst := dup.Triangles[i]
		if src == dst {
			t.Fatal("Copy must not share Triangle structs")
		}
		if !src.HasVertexColors() || !dst.HasVertexColors() {
			t.Fatal("copy lost vertex colors")
		}
		assertColor(t, dst.C1, src.C1, 0, "copied C1")
		assertColor(t, dst.C2, src.C2, 0, "copied C2")
		assertColor(t, dst.C3, src.C3, 0, "copied C3")
	}
	// mutating the copy must not affect the original mesh
	dup.Triangles[0].SetVertexColors(testWhite, testWhite, testWhite)
	assertColor(t, mesh.Triangles[0].C1, origColors[0], 0, "original unchanged by copy edit")
}

// ---------------------------------------------------------------------------
// 6. OBJ loader: "v x y z r g b" vertex colors, ordering preserved
// ---------------------------------------------------------------------------

func TestLoadOBJVertexColors(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "colored.obj")
	obj := `
v 0 0 0 1 0 0
v 1 0 0 0 1 0
v 1 1 0 0 0 1
v 0 1 0 1 1 1
f 1 2 3
f 1 3 4
`
	if err := os.WriteFile(path, []byte(obj), 0o644); err != nil {
		t.Fatal(err)
	}
	mesh, err := LoadOBJ(path, DiffuseMaterial(testMagenta))
	if err != nil {
		t.Fatal(err)
	}
	if len(mesh.Triangles) != 2 {
		t.Fatalf("got %d triangles, want 2", len(mesh.Triangles))
	}
	t1, t2 := mesh.Triangles[0], mesh.Triangles[1]
	if !t1.HasVertexColors() || !t2.HasVertexColors() {
		t.Fatal("OBJ triangles lost vertex colors")
	}
	assertColor(t, t1.C1, testRed, 1e-12, "obj t1 C1")
	assertColor(t, t1.C2, testGreen, 1e-12, "obj t1 C2")
	assertColor(t, t1.C3, testBlue, 1e-12, "obj t1 C3")
	assertColor(t, t2.C1, testRed, 1e-12, "obj t2 C1")
	assertColor(t, t2.C2, testBlue, 1e-12, "obj t2 C2 (shared corner)")
	assertColor(t, t2.C3, testWhite, 1e-12, "obj t2 C3")
	// shared corner (v3) carries the same color on both triangles
	assertColor(t, t1.C3, t2.C2, 0, "shared vertex color")

	// OBJ without vertex colors: no vertex colors configured, legacy path
	plainPath := filepath.Join(dir, "plain.obj")
	plain := `
v 0 0 0
v 1 0 0
v 1 1 0
f 1 2 3
`
	if err := os.WriteFile(plainPath, []byte(plain), 0o644); err != nil {
		t.Fatal(err)
	}
	plainMesh, err := LoadOBJ(plainPath, DiffuseMaterial(testMagenta))
	if err != nil {
		t.Fatal(err)
	}
	if plainMesh.Triangles[0].HasVertexColors() {
		t.Fatal("colorless OBJ must not report vertex colors")
	}
}

// ---------------------------------------------------------------------------
// 7. Legacy regression: triangles without vertex colors are unchanged and the
//    full sampler path produces finite, in-range output.
// ---------------------------------------------------------------------------

func TestLegacyTriangleUnchanged(t *testing.T) {
	v1, v2, v3 := Vector{0, 0, 0}, Vector{1, 0, 0}, Vector{0, 1, 0}
	tr := NewTriangle(v1, v2, v3, Vector{}, Vector{}, Vector{},
		DiffuseMaterial(testMagenta))
	if tr.HasVertexColors() {
		t.Fatal("legacy triangle unexpectedly has vertex colors")
	}
	// MaterialAt output is identical to the pre-feature material+texture path
	assertColor(t, MaterialAt(tr, Vector{0.25, 0.25, 0}).Color, testMagenta, 0,
		"legacy constant color")
	tr.Material.Texture = solidTexture{Color{0.3, 0.6, 0.9}}
	assertColor(t, MaterialAt(tr, Vector{0.25, 0.25, 0}).Color,
		Color{0.3, 0.6, 0.9}, 0, "legacy texture color")

	// end-to-end render through DefaultSampler stays finite and in range
	scene := Scene{}
	scene.Add(tr)
	scene.Compile()
	sampler := NewDirectSampler()
	rnd := rand.New(rand.NewSource(3))
	camera := LookAt(Vector{0.25, 0.25, 2}, Vector{0.25, 0.25, 0},
		Vector{0, 1, 0}, 45)
	for y := 0; y < 8; y++ {
		for x := 0; x < 8; x++ {
			ray := camera.CastRay(x, y, 8, 8, 0.5, 0.5, rnd)
			c := sampler.Sample(&scene, ray, rnd)
			assertFinite(t, c, "legacy sampler output")
		}
	}

	// vertex-colored triangle through the real sampler must stay finite too
	tr.Material.Texture = nil
	tr.SetVertexColors(testRed, testGreen, testBlue)
	for y := 0; y < 8; y++ {
		for x := 0; x < 8; x++ {
			ray := camera.CastRay(x, y, 8, 8, 0.5, 0.5, rnd)
			c := sampler.Sample(&scene, ray, rnd)
			assertFinite(t, c, "vertex-color sampler output")
		}
	}
}
