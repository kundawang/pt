package pt

import (
	"math"
	"math/rand"
	"testing"
)

// renderScene renders the scene with a deterministic, quiet renderer.
func renderScene(scene *Scene, camera *Camera, w, h, spp int, seed int64) *Buffer {
	sampler := NewDirectSampler()
	r := NewRenderer(scene, camera, sampler, w, h)
	r.SamplesPerPixel = spp
	r.NumCPU = 4
	r.Verbose = false
	r.Seed = seed
	r.IterativeRender("", 0) // no-op; keeps API surface exercised
	r.run()
	return r.Buffer
}

func litPixels(buf *Buffer) int {
	n := 0
	for y := 0; y < buf.H; y++ {
		for x := 0; x < buf.W; x++ {
			if buf.Color(x, y).MaxComponent() > 0.01 {
				n++
			}
		}
	}
	return n
}

func assertNoNaN(t *testing.T, buf *Buffer) {
	t.Helper()
	for y := 0; y < buf.H; y++ {
		for x := 0; x < buf.W; x++ {
			c := buf.Color(x, y)
			for _, v := range []float64{c.R, c.G, c.B} {
				if math.IsNaN(v) || math.IsInf(v, 0) {
					t.Fatalf("invalid pixel at (%d, %d): %v", x, y, c)
				}
			}
		}
	}
}

// movingSphereScene returns a scene with a single emissive sphere sweeping
// from c0 to c1 over [t0, t1] on a black background.
func movingSphereScene(c0, c1 Vector, radius, t0, t1 float64) *Scene {
	scene := &Scene{}
	material := LightMaterial(White, 5)
	scene.Add(NewMovingSphere(c0, c1, radius, material, t0, t1))
	return scene
}

func staticSphereScene(center Vector, radius float64) *Scene {
	scene := &Scene{}
	material := LightMaterial(White, 5)
	scene.Add(NewSphere(center, radius, material))
	return scene
}

func testCamera() Camera {
	return LookAt(V(0, 0, 6), V(0, 0, 0), V(0, 1, 0), 40)
}

func TestCameraShutterTimes(t *testing.T) {
	rnd := rand.New(rand.NewSource(1))
	// default: zero-duration shutter at time 0
	c := testCamera()
	for i := 0; i < 100; i++ {
		if ray := c.CastRay(10, 10, 64, 64, 0.5, 0.5, rnd); ray.Time != 0 {
			t.Fatalf("default shutter: got time %v, want 0", ray.Time)
		}
	}
	// open interval: times sampled uniformly within [open, close]
	c.SetShutter(0.2, 0.8)
	lo, hi := math.Inf(1), math.Inf(-1)
	for i := 0; i < 1000; i++ {
		tm := c.CastRay(10, 10, 64, 64, 0.5, 0.5, rnd).Time
		if tm < 0.2 || tm > 0.8 {
			t.Fatalf("time %v outside [0.2, 0.8]", tm)
		}
		lo = math.Min(lo, tm)
		hi = math.Max(hi, tm)
	}
	if hi-lo < 0.5 {
		t.Fatalf("times not spread over the shutter interval: [%v, %v]", lo, hi)
	}
	// zero-duration shutter: every ray at the same instant
	c.SetShutter(0.5, 0.5)
	for i := 0; i < 100; i++ {
		if ray := c.CastRay(10, 10, 64, 64, 0.5, 0.5, rnd); ray.Time != 0.5 {
			t.Fatalf("zero-duration shutter: got time %v, want 0.5", ray.Time)
		}
	}
	// negative interval works too
	c.SetShutter(-1, -0.5)
	for i := 0; i < 100; i++ {
		tm := c.CastRay(10, 10, 64, 64, 0.5, 0.5, rnd).Time
		if tm < -1 || tm > -0.5 {
			t.Fatalf("time %v outside [-1, -0.5]", tm)
		}
	}
}

func TestRayTimePropagation(t *testing.T) {
	ray := Ray{V(0, 0, 0), V(0, 0, 1), 0.7}
	bounced := ray.WeightedBounce(0.3, 0.6, rand.New(rand.NewSource(1)))
	if bounced.Time != 0.7 {
		t.Fatalf("WeightedBounce lost time: %v", bounced.Time)
	}
	m := Translate(V(1, 2, 3))
	if moved := m.MulRay(ray); moved.Time != 0.7 {
		t.Fatalf("MulRay lost time: %v", moved.Time)
	}
}

func TestMovingShapeBoundingBoxCoversMotion(t *testing.T) {
	// the sphere ends far outside its initial bounding box
	shape := NewMovingSphere(V(0, 0, 0), V(10, 0, 0), 1, LightMaterial(White, 1), 0, 1)
	box := shape.BoundingBox()
	if !box.Contains(V(10, 0, 0)) {
		t.Fatalf("bounding box %v does not contain final position", box)
	}
	if !box.Contains(V(0, 0, 0)) {
		t.Fatalf("bounding box %v does not contain initial position", box)
	}
	if box.Min.X < -1.0001 || box.Max.X > 11.0001 {
		t.Fatalf("bounding box %v larger than the swept volume", box)
	}
}

func TestMovingShapeIntersectAtTimes(t *testing.T) {
	shape := NewMovingSphere(V(0, 0, 0), V(10, 0, 0), 1, LightMaterial(White, 1), 0, 1)
	// ray aimed along -x through the sphere's position at time 0.5
	ray := Ray{V(5, 5, 0), V(0, -1, 0), 0.5}
	hit := shape.Intersect(ray)
	if !hit.Ok() {
		t.Fatal("expected hit at time 0.5")
	}
	info := hit.Info(ray)
	if math.Abs(info.Position.X-5) > 1e-6 {
		t.Fatalf("hit at wrong position: %v", info.Position)
	}
	// at time 0 the sphere is at the origin: same ray must miss
	if hit := shape.Intersect(Ray{V(5, 5, 0), V(0, -1, 0), 0}); hit.Ok() {
		t.Fatal("unexpected hit at time 0")
	}
	// times beyond the interval clamp to the end position
	if hit := shape.Intersect(Ray{V(10, 5, 0), V(0, -1, 0), 2}); !hit.Ok() {
		t.Fatal("expected hit at clamped time 2")
	}
	if hit := shape.Intersect(Ray{V(0, 5, 0), V(0, -1, 0), -1}); !hit.Ok() {
		t.Fatal("expected hit at clamped time -1")
	}
	// zero-length motion interval: frozen at the start transform
	frozen := NewMovingSphere(V(1, 0, 0), V(9, 0, 0), 1, LightMaterial(White, 1), 0.5, 0.5)
	if hit := frozen.Intersect(Ray{V(1, 5, 0), V(0, -1, 0), 0.5}); !hit.Ok() {
		t.Fatal("expected hit on frozen shape")
	}
	if hit := frozen.Intersect(Ray{V(9, 5, 0), V(0, -1, 0), 0.5}); hit.Ok() {
		t.Fatal("frozen shape must not move")
	}
}

func TestZeroShutterMatchesStatic(t *testing.T) {
	// a moving sphere sampled at one frozen instant must render like a
	// static sphere at that instant's position
	moving := movingSphereScene(V(-2, 0, 0), V(2, 0, 0), 0.5, 0, 1)
	camera := testCamera()
	camera.SetShutter(0.5, 0.5)
	motionBuf := renderScene(moving, &camera, 64, 64, 16, 7)

	static := staticSphereScene(V(0, 0, 0), 0.5)
	camera2 := testCamera()
	staticBuf := renderScene(static, &camera2, 64, 64, 16, 7)

	bad := 0
	for y := 0; y < 64; y++ {
		for x := 0; x < 64; x++ {
			d := motionBuf.Color(x, y).Sub(staticBuf.Color(x, y)).MaxComponent()
			if math.Abs(d) > 0.05 {
				bad++
			}
		}
	}
	if bad > 64*64/50 {
		t.Fatalf("zero-duration shutter differs from static scene in %d pixels", bad)
	}
}

func TestMotionBlurSmear(t *testing.T) {
	// glowing sphere sweeping horizontally: expect a horizontal streak
	moving := movingSphereScene(V(-2, 0, 0), V(2, 0, 0), 0.5, 0, 1)
	camera := testCamera()
	camera.SetShutter(0, 1)
	motionBuf := renderScene(moving, &camera, 96, 64, 64, 7)
	assertNoNaN(t, motionBuf)

	// static control: same sphere frozen at the path midpoint
	static := staticSphereScene(V(0, 0, 0), 0.5)
	camera2 := testCamera()
	staticBuf := renderScene(static, &camera2, 96, 64, 64, 7)

	motionLit := litPixels(motionBuf)
	staticLit := litPixels(staticBuf)
	if motionLit <= staticLit*2 {
		t.Fatalf("expected a streak: moving lit %d, static lit %d", motionLit, staticLit)
	}

	// boundary pixels: well above/below the swept path everything is black
	for x := 0; x < 96; x++ {
		for _, y := range []int{0, 1, 62, 63} {
			if c := motionBuf.Color(x, y); c.MaxComponent() != 0 {
				t.Fatalf("boundary pixel (%d, %d) lit: %v", x, y, c)
			}
		}
	}
	// the streak reaches both ends of the path
	endLit := func(x0, x1 int) bool {
		for x := x0; x < x1; x++ {
			for y := 24; y < 40; y++ {
				if motionBuf.Color(x, y).MaxComponent() > 0.01 {
					return true
				}
			}
		}
		return false
	}
	if !endLit(8, 20) || !endLit(76, 88) {
		t.Fatal("streak does not extend to both ends of the motion")
	}
}

func TestDeterministicRender(t *testing.T) {
	render := func() *Buffer {
		scene := movingSphereScene(V(-2, 0, 0), V(2, 0, 0), 0.5, 0, 1)
		camera := testCamera()
		camera.SetShutter(0, 1)
		return renderScene(scene, &camera, 64, 64, 16, 42)
	}
	a := render()
	b := render()
	for i := range a.Pixels {
		if a.Pixels[i] != b.Pixels[i] {
			t.Fatalf("pixel %d differs between identical runs: %v vs %v",
				i, a.Pixels[i].M, b.Pixels[i].M)
		}
	}
}

func TestNegativeTimeRange(t *testing.T) {
	scene := movingSphereScene(V(-2, 0, 0), V(2, 0, 0), 0.5, -1, 0)
	camera := testCamera()
	camera.SetShutter(-1, 0)
	buf := renderScene(scene, &camera, 64, 64, 16, 7)
	assertNoNaN(t, buf)
	if litPixels(buf) == 0 {
		t.Fatal("negative time range rendered nothing")
	}
}

func TestHighSpeedMotion(t *testing.T) {
	// extreme velocity: the sphere crosses the whole frame many times over
	scene := movingSphereScene(V(-1000, 0, 0), V(1000, 0, 0), 0.5, 0, 1)
	camera := testCamera()
	camera.SetShutter(0, 1)
	buf := renderScene(scene, &camera, 64, 64, 16, 7)
	assertNoNaN(t, buf)
	if litPixels(buf) == 0 {
		t.Fatal("high-speed motion left no trace at all")
	}
}

// TestStaticSceneSmoke mirrors examples/hello.go: a minimal static scene
// rendered without any motion-blur configuration must be unaffected.
func TestStaticSceneSmoke(t *testing.T) {
	scene := Scene{}
	material := DiffuseMaterial(White)
	scene.Add(NewPlane(V(0, 0, 0), V(0, 0, 1), material))
	scene.Add(NewSphere(V(0, 0, 1), 1, material))
	scene.Add(NewSphere(V(0, 0, 5), 1, LightMaterial(White, 8)))
	camera := LookAt(V(3, 3, 3), V(0, 0, 0.5), V(0, 0, 1), 50)

	sampler := NewSampler(1, 2)
	r := NewRenderer(&scene, &camera, sampler, 64, 64)
	r.SamplesPerPixel = 4
	r.NumCPU = 4
	r.Verbose = false
	r.Seed = 7
	r.run()
	assertNoNaN(t, r.Buffer)
	if litPixels(r.Buffer) == 0 {
		t.Fatal("static scene rendered black")
	}
	// default shutter: every camera ray is at time 0
	ray := camera.CastRay(32, 32, 64, 64, 0.5, 0.5, rand.New(rand.NewSource(1)))
	if ray.Time != 0 {
		t.Fatalf("default shutter produced time %v", ray.Time)
	}
}
