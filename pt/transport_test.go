package pt

import (
	"image"
	"image/color"
	"math"
	"math/rand"
	"sync"
	"testing"
)

// ---------------------------------------------------------------------------
// legacySampler is a verbatim copy of the pre-refactor DefaultSampler
// implementation (including its use of the global rand source for
// LightModeRandom). It exists only to prove that the new transport
// boundary in transport.go is behaviorally equivalent.
// ---------------------------------------------------------------------------

type legacySampler struct {
	FirstHitSamples int
	MaxBounces      int
	DirectLighting  bool
	SoftShadows     bool
	LightMode       LightMode
	SpecularMode    SpecularMode
}

func (s *legacySampler) Sample(scene *Scene, ray Ray, rnd *rand.Rand) Color {
	return s.sample(scene, ray, true, s.FirstHitSamples, 0, rnd)
}

func (s *legacySampler) sample(scene *Scene, ray Ray, emission bool, samples, depth int, rnd *rand.Rand) Color {
	if depth > s.MaxBounces {
		return Black
	}
	hit := scene.Intersect(ray)
	if !hit.Ok() {
		return s.sampleEnvironment(scene, ray)
	}
	info := hit.Info(ray)
	material := info.Material
	result := Black
	if material.Emittance > 0 {
		if s.DirectLighting && !emission {
			return Black
		}
		result = result.Add(material.Color.MulScalar(material.Emittance * float64(samples)))
	}
	n := int(math.Sqrt(float64(samples)))
	var ma, mb BounceType
	if s.SpecularMode == SpecularModeAll || (depth == 0 && s.SpecularMode == SpecularModeFirst) {
		ma = BounceTypeDiffuse
		mb = BounceTypeSpecular
	} else {
		ma = BounceTypeAny
		mb = BounceTypeAny
	}
	for u := 0; u < n; u++ {
		for v := 0; v < n; v++ {
			for mode := ma; mode <= mb; mode++ {
				fu := (float64(u) + rnd.Float64()) / float64(n)
				fv := (float64(v) + rnd.Float64()) / float64(n)
				newRay, reflected, p := ray.Bounce(&info, fu, fv, mode, rnd)
				if mode == BounceTypeAny {
					p = 1
				}
				if p > 0 && reflected {
					// specular
					indirect := s.sample(scene, newRay, reflected, 1, depth+1, rnd)
					tinted := indirect.Mix(material.Color.Mul(indirect), material.Tint)
					result = result.Add(tinted.MulScalar(p))
				}
				if p > 0 && !reflected {
					// diffuse
					indirect := s.sample(scene, newRay, reflected, 1, depth+1, rnd)
					direct := Black
					if s.DirectLighting {
						direct = s.sampleLights(scene, info.Ray, rnd)
					}
					result = result.Add(material.Color.Mul(direct.Add(indirect)).MulScalar(p))
				}
			}
		}
	}
	return result.DivScalar(float64(n * n))
}

func (s *legacySampler) sampleEnvironment(scene *Scene, ray Ray) Color {
	if scene.Texture != nil {
		d := ray.Direction
		u := math.Atan2(d.Z, d.X) + scene.TextureAngle
		v := math.Atan2(d.Y, Vector{d.X, 0, d.Z}.Length())
		u = (u + math.Pi) / (2 * math.Pi)
		v = (v + math.Pi/2) / math.Pi
		return scene.Texture.Sample(u, v)
	}
	return scene.Color
}

func (s *legacySampler) sampleLights(scene *Scene, n Ray, rnd *rand.Rand) Color {
	nLights := len(scene.Lights)
	if nLights == 0 {
		return Black
	}

	if s.LightMode == LightModeAll {
		var result Color
		for _, light := range scene.Lights {
			result = result.Add(s.sampleLight(scene, n, rnd, light))
		}
		return result
	} else {
		// pick a random light
		light := scene.Lights[rand.Intn(nLights)]
		return s.sampleLight(scene, n, rnd, light).MulScalar(float64(nLights))
	}
}

func (s *legacySampler) sampleLight(scene *Scene, n Ray, rnd *rand.Rand, light Shape) Color {
	// get bounding sphere center and radius
	var center Vector
	var radius float64
	switch t := light.(type) {
	case *Sphere:
		radius = t.Radius
		center = t.Center
	default:
		// get bounding sphere from bounding box
		box := t.BoundingBox()
		radius = box.OuterRadius()
		center = box.Center()
	}

	// get random point in disk
	point := center
	if s.SoftShadows {
		for {
			x := rnd.Float64()*2 - 1
			y := rnd.Float64()*2 - 1
			if x*x+y*y <= 1 {
				l := center.Sub(n.Origin).Normalize()
				u := l.Cross(RandomUnitVector(rnd)).Normalize()
				v := l.Cross(u)
				point = Vector{}
				point = point.Add(u.MulScalar(x * radius))
				point = point.Add(v.MulScalar(y * radius))
				point = point.Add(center)
				break
			}
		}
	}

	// construct ray toward light point
	ray := Ray{n.Origin, point.Sub(n.Origin).Normalize()}

	// get cosine term
	diffuse := ray.Direction.Dot(n.Direction)
	if diffuse <= 0 {
		return Black
	}

	// check for light visibility
	hit := scene.Intersect(ray)
	if !hit.Ok() || hit.Shape != light {
		return Black
	}

	// compute solid angle (hemisphere coverage)
	hyp := center.Sub(n.Origin).Length()
	opp := radius
	theta := math.Asin(opp / hyp)
	adj := opp / math.Tan(theta)
	d := math.Cos(theta) * adj
	r := math.Sin(theta) * adj
	coverage := (r * r) / (d * d)

	// TODO: fix issue where hyp < opp (point inside sphere)
	if hyp < opp {
		coverage = 1
	}
	coverage = math.Min(coverage, 1)

	// get material properties from light
	material := MaterialAt(light, point)

	// combine factors
	m := material.Emittance * diffuse * coverage
	return material.Color.MulScalar(m)
}

// legacyOf mirrors a DefaultSampler configuration into a legacySampler.
func legacyOf(s *DefaultSampler) *legacySampler {
	return &legacySampler{
		s.FirstHitSamples, s.MaxBounces, s.DirectLighting,
		s.SoftShadows, s.LightMode, s.SpecularMode,
	}
}

// ---------------------------------------------------------------------------
// Test scenes
// ---------------------------------------------------------------------------

func testTexture() Texture {
	im := image.NewRGBA(image.Rect(0, 0, 4, 4))
	for y := 0; y < 4; y++ {
		for x := 0; x < 4; x++ {
			if (x+y)%2 == 0 {
				im.Set(x, y, color.RGBA{255, 64, 32, 255})
			} else {
				im.Set(x, y, color.RGBA{16, 128, 200, 255})
			}
		}
	}
	return NewTexture(im)
}

// testSceneA: diffuse sphere + spherical light + flat environment color.
func testSceneA() (*Scene, Camera) {
	scene := Scene{}
	scene.Color = Color{0.05, 0.07, 0.09}
	scene.Add(NewSphere(V(0, 0, 0), 1, DiffuseMaterial(HexColor(0xEFEFEF))))
	scene.Add(NewSphere(V(0, 4, 1), 1, LightMaterial(White, 6)))
	camera := LookAt(V(0, 1, 5), V(0, 0, 0), V(0, 1, 0), 45)
	scene.Compile()
	return &scene, camera
}

// testSceneB: glossy, transparent, textured and emissive shapes, two lights.
func testSceneB() (*Scene, Camera) {
	scene := Scene{}
	scene.Color = Color{0.02, 0.02, 0.03}
	tex := testTexture()
	textured := DiffuseMaterial(White)
	textured.Texture = tex
	textured.GlossTexture = tex
	scene.Add(NewSphere(V(-1.4, 0, 0), 1, GlossyMaterial(HexColor(0x334D5C), 1.5, Radians(20))))
	scene.Add(NewSphere(V(1.4, 0, 0), 1, TransparentMaterial(HexColor(0xEF8A62), 1.5, Radians(10), 0.3)))
	scene.Add(NewSphere(V(0, 0, 2.2), 1, textured))
	scene.Add(NewSphere(V(0, 5, 0), 1.5, LightMaterial(White, 4)))
	scene.Add(NewSphere(V(3, 3, 3), 0.75, LightMaterial(HexColor(0xFFD9A0), 2)))
	camera := LookAt(V(0, 1.2, 6), V(0, 0, 0.5), V(0, 1, 0), 40)
	scene.Compile()
	return &scene, camera
}

// ---------------------------------------------------------------------------
// 1. Per-pixel behavioral equivalence: legacy sampler vs new boundary
// ---------------------------------------------------------------------------

func equivalentConfigs() []*DefaultSampler {
	configs := []*DefaultSampler{
		NewSampler(1, 1),
		NewSampler(4, 4),
		NewSampler(9, 2),
		NewDirectSampler(),
		{2, 3, false, true, LightModeAll, SpecularModeNaive},
		{4, 2, true, false, LightModeAll, SpecularModeNaive},
		{4, 2, true, true, LightModeAll, SpecularModeFirst},
		{4, 2, true, true, LightModeAll, SpecularModeAll},
		{1, 0, true, true, LightModeAll, SpecularModeNaive},
	}
	// LightModeRandom consumes the global rand source in the legacy code,
	// so exact per-pixel comparison is only meaningful with LightModeAll.
	for _, s := range configs {
		s.LightMode = LightModeAll
	}
	return configs
}

func TestPerPixelEquivalence(t *testing.T) {
	scenes := []struct {
		name  string
		scene *Scene
		cam   Camera
	}{
		{"diffuse", nil, Camera{}},
		{"mixed", nil, Camera{}},
	}
	scenes[0].scene, scenes[0].cam = testSceneA()
	scenes[1].scene, scenes[1].cam = testSceneB()

	const w, h = 12, 12
	for _, sc := range scenes {
		for ci, sampler := range equivalentConfigs() {
			legacy := legacyOf(sampler)
			pipeline := sampler.Pipeline()
			mismatch := 0
			for y := 0; y < h; y++ {
				for x := 0; x < w; x++ {
					seed := int64(y*w+x) + 1
					ray := sc.cam.CastRay(x, y, w, h, 0.5, 0.5, nil)
					want := legacy.Sample(sc.scene, ray, rand.New(rand.NewSource(seed)))
					got := sampler.Sample(sc.scene, ray, rand.New(rand.NewSource(seed)))
					if got != want {
						t.Errorf("%s config %d pixel (%d,%d): DefaultSampler=%v, legacy=%v",
							sc.name, ci, x, y, got, want)
						mismatch++
					}
					// the hand-wired pipeline must agree exactly too
					got2 := pipeline.Sample(sc.scene, ray, rand.New(rand.NewSource(seed)))
					if got2 != want {
						t.Errorf("%s config %d pixel (%d,%d): Pipeline=%v, legacy=%v",
							sc.name, ci, x, y, got2, want)
						mismatch++
					}
					if mismatch > 5 {
						t.Fatalf("%s config %d: too many mismatches, aborting", sc.name, ci)
					}
				}
			}
		}
	}
}

// TestLightModeRandomEquivalence compares LightModeRandom statistically:
// the legacy sampler picks a light via the global rand source while the new
// DefaultLightSampler draws from the per-goroutine rnd (removing hidden
// shared state). Both are uniform over the lights, so sample means agree.
func TestLightModeRandomEquivalence(t *testing.T) {
	scene, _ := testSceneB()
	cfg := &DefaultSampler{1, 0, true, false, LightModeRandom, SpecularModeAll}
	legacy := legacyOf(cfg)
	ray := Ray{V(0, 1.2, 6), V(0, -0.2, -1).Normalize()}
	const n = 20000
	var sumNew, sumLegacy Color
	rnd := rand.New(rand.NewSource(42))
	for i := 0; i < n; i++ {
		sumNew = sumNew.Add(cfg.Sample(scene, ray, rnd))
		sumLegacy = sumLegacy.Add(legacy.Sample(scene, ray, rnd))
	}
	meanNew := sumNew.DivScalar(n)
	meanLegacy := sumLegacy.DivScalar(n)
	for i, d := range [3]float64{
		math.Abs(meanNew.R - meanLegacy.R),
		math.Abs(meanNew.G - meanLegacy.G),
		math.Abs(meanNew.B - meanLegacy.B),
	} {
		mag := math.Max(math.Abs(meanLegacy.R), 0.1)
		if d > 0.05*mag {
			t.Errorf("channel %d: new mean %v vs legacy mean %v (diff %g)", i, meanNew, meanLegacy, d)
		}
	}
}

// ---------------------------------------------------------------------------
// 2. Call order and call counts through the boundary
// ---------------------------------------------------------------------------

type eventLog struct {
	mu     sync.Mutex
	events []string
}

func (l *eventLog) add(e string) {
	l.mu.Lock()
	l.events = append(l.events, e)
	l.mu.Unlock()
}

func (l *eventLog) snapshot() []string {
	l.mu.Lock()
	defer l.mu.Unlock()
	return append([]string(nil), l.events...)
}

type recIntersector struct {
	log   *eventLog
	inner Intersector
}

func (r recIntersector) Intersect(ray Ray) Hit {
	r.log.add("intersect")
	return r.inner.Intersect(ray)
}

type recResolver struct {
	log   *eventLog
	inner HitResolver
}

func (r recResolver) Info(hit Hit, ray Ray) HitInfo {
	r.log.add("resolve")
	return r.inner.Info(hit, ray)
}

type recEvaluator struct {
	log   *eventLog
	inner MaterialEvaluator
}

func (r recEvaluator) MaterialAt(shape Shape, point Vector) Material {
	r.log.add("material")
	return r.inner.MaterialAt(shape, point)
}

type recEnvironment struct {
	log   *eventLog
	inner EnvironmentSampler
}

func (r recEnvironment) SampleEnvironment(scene *Scene, ray Ray) Color {
	r.log.add("environment")
	return r.inner.SampleEnvironment(scene, ray)
}

type recLights struct {
	log   *eventLog
	inner LightSampler
}

func (r recLights) SampleLights(scene *Scene, n Ray, rnd *rand.Rand) Color {
	r.log.add("lights")
	return r.inner.SampleLights(scene, n, rnd)
}

type recBouncer struct {
	log   *eventLog
	inner Bouncer
}

func (r recBouncer) Bounce(in Ray, info *HitInfo, u, v float64, mode BounceType, rnd *rand.Rand) (Ray, bool, float64) {
	r.log.add("bounce")
	return r.inner.Bounce(in, info, u, v, mode, rnd)
}

type stubRecurser struct {
	log   *eventLog
	depth *int
}

func (r stubRecurser) Recurse(scene *Scene, ray Ray, emission bool, samples, depth int, rnd *rand.Rand) Color {
	r.log.add("recurse")
	if r.depth != nil && depth > *r.depth {
		*r.depth = depth
	}
	return Black
}

func recordedPipeline(scene *Scene, log *eventLog, rec Recurser) *Pipeline {
	return &Pipeline{
		FirstHitSamples: 1,
		MaxBounces:      4,
		DirectLighting:  true,
		SpecularMode:    SpecularModeAll,
		Intersector:     recIntersector{log, scene},
		Resolver:        recResolver{log, DefaultHitResolver{recEvaluator{log, DefaultMaterialEvaluator{}}}},
		Environment:     recEnvironment{log, DefaultEnvironmentSampler{}},
		Lights:          recLights{log, DefaultLightSampler{false, LightModeAll}},
		Bouncer:         recBouncer{log, DefaultBouncer{}},
		Recurser:        rec,
	}
}

func equalStrings(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func TestCallOrderMiss(t *testing.T) {
	scene, _ := testSceneA()
	log := &eventLog{}
	p := recordedPipeline(scene, log, stubRecurser{log, nil})
	// ray into the void
	c := p.Sample(scene, Ray{V(0, 30, 0), V(0, 1, 0)}, rand.New(rand.NewSource(1)))
	if c != scene.Color {
		t.Errorf("miss: got %v, want scene color %v", c, scene.Color)
	}
	want := []string{"intersect", "environment"}
	if !equalStrings(log.snapshot(), want) {
		t.Errorf("miss events = %v, want %v", log.snapshot(), want)
	}
}

func TestCallOrderDepthCap(t *testing.T) {
	scene, _ := testSceneA()
	log := &eventLog{}
	p := recordedPipeline(scene, log, stubRecurser{log, nil})
	c := p.Recurse(scene, Ray{V(0, 1, 5), V(0, 0, -1)}, true, 1, 5, rand.New(rand.NewSource(1)))
	if c != Black {
		t.Errorf("depth cap: got %v, want Black", c)
	}
	if len(log.snapshot()) != 0 {
		t.Errorf("depth cap: events = %v, want none", log.snapshot())
	}
}

func TestCallOrderHit(t *testing.T) {
	// single specular sphere, no lights needed (LightModeAll over 1 light)
	scene := Scene{}
	scene.Add(NewSphere(V(0, 0, 0), 1, SpecularMaterial(White, 1.5)))
	scene.Add(NewSphere(V(0, 5, 0), 1, LightMaterial(White, 4)))
	scene.Compile()
	log := &eventLog{}
	p := recordedPipeline(&scene, log, stubRecurser{log, nil})
	ray := Ray{V(0, 0, 5), V(0, 0, -1)}
	p.Sample(&scene, ray, rand.New(rand.NewSource(7)))
	// SpecularModeAll at depth 0: diffuse mode then specular mode.
	// Diffuse: bounce -> recurse -> lights. Specular: bounce -> recurse.
	want := []string{
		"intersect", "resolve", "material",
		"bounce", "recurse", "lights",
		"bounce", "recurse",
	}
	if !equalStrings(log.snapshot(), want) {
		t.Errorf("hit events = %v, want %v", log.snapshot(), want)
	}
}

func TestCallOrderEmissiveEarlyReturn(t *testing.T) {
	scene := Scene{}
	scene.Add(NewSphere(V(0, 0, 0), 1, LightMaterial(White, 4)))
	scene.Compile()
	log := &eventLog{}
	p := recordedPipeline(&scene, log, stubRecurser{log, nil})
	ray := Ray{V(0, 0, 5), V(0, 0, -1)}
	// emission=false with DirectLighting on: emissive hit returns Black
	// immediately after material evaluation, before any bounce.
	c := p.Recurse(&scene, ray, false, 1, 1, rand.New(rand.NewSource(3)))
	if c != Black {
		t.Errorf("emissive early return: got %v, want Black", c)
	}
	want := []string{"intersect", "resolve", "material"}
	if !equalStrings(log.snapshot(), want) {
		t.Errorf("emissive events = %v, want %v", log.snapshot(), want)
	}
}

func TestRecurserSeesDepthIncrements(t *testing.T) {
	scene, _ := testSceneA()
	log := &eventLog{}
	maxDepth := 0
	p := recordedPipeline(scene, log, stubRecurser{log, &maxDepth})
	p.MaxBounces = 3
	p.Sample(scene, Ray{V(0, 1, 5), V(0, -0.2, -1).Normalize()}, rand.New(rand.NewSource(11)))
	if maxDepth != 1 {
		t.Errorf("recurser saw max depth %d, want 1 (stub does not recurse further)", maxDepth)
	}
}

// ---------------------------------------------------------------------------
// 3. Robustness: no panics, no shared mutable state
// ---------------------------------------------------------------------------

func TestNoHitReturnsEnvironment(t *testing.T) {
	scene, _ := testSceneA()
	p := NewSampler(1, 1).Pipeline()
	c := p.Sample(scene, Ray{V(0, 30, 0), V(0, 1, 0)}, rand.New(rand.NewSource(1)))
	if c != scene.Color {
		t.Errorf("got %v, want %v", c, scene.Color)
	}
}

type stubIntersector struct{ hit Hit }

func (s stubIntersector) Intersect(Ray) Hit { return s.hit }

func TestNilShapeHitDoesNotPanic(t *testing.T) {
	scene, _ := testSceneA()
	p := NewSampler(1, 1).Pipeline()
	p.Intersector = stubIntersector{Hit{nil, 1, nil}}
	c := p.Sample(scene, Ray{V(0, 0, 5), V(0, 0, -1)}, rand.New(rand.NewSource(1)))
	if c != scene.Color {
		t.Errorf("nil-shape hit: got %v, want environment %v", c, scene.Color)
	}
	// the resolver itself must also tolerate a nil Shape
	info := DefaultHitResolver{}.Info(Hit{nil, 1, nil}, Ray{V(0, 0, 5), V(0, 0, -1)})
	if info.Shape != nil || info.Material != (Material{}) {
		t.Errorf("nil-shape resolve: got %+v, want zero HitInfo", info)
	}
	// and the legacy entry point must not panic either
	_ = (&Hit{nil, 1, nil}).Info(Ray{V(0, 0, 5), V(0, 0, -1)})
	// nil shape material evaluation
	if m := MaterialAt(nil, Vector{}); m != (Material{}) {
		t.Errorf("MaterialAt(nil) = %+v, want zero Material", m)
	}
}

// cyclicBouncer always reflects the incoming ray straight back with p=1:
// a "material" that never terminates a path on its own.
type cyclicBouncer struct{}

func (cyclicBouncer) Bounce(in Ray, info *HitInfo, u, v float64, mode BounceType, rnd *rand.Rand) (Ray, bool, float64) {
	return in, true, 1
}

func TestCyclicMaterialBoundedByMaxBounces(t *testing.T) {
	scene, _ := testSceneA()
	maxDepth := 0
	p := &Pipeline{
		FirstHitSamples: 1,
		MaxBounces:      8,
		Bouncer:         cyclicBouncer{},
		Recurser:        nil, // set below
	}
	// count recursion depth via a wrapper that delegates back to p
	p.Recurser = &depthTracker{inner: p, max: &maxDepth}
	c := p.Sample(scene, Ray{V(0, 0, 5), V(0, 0, -1)}, rand.New(rand.NewSource(1)))
	if c != Black {
		t.Errorf("cyclic bouncer: got %v, want Black (env is black here? no—depth cap)", c)
	}
	if maxDepth != 9 {
		t.Errorf("cyclic bouncer: max depth %d, want 9 (MaxBounces+1)", maxDepth)
	}
}

type depthTracker struct {
	inner *Pipeline
	max   *int
}

func (d *depthTracker) Recurse(scene *Scene, ray Ray, emission bool, samples, depth int, rnd *rand.Rand) Color {
	if depth > *d.max {
		*d.max = depth
	}
	// recurse through the pipeline itself, keeping the wrapper in place
	return d.inner.Recurse(scene, ray, emission, samples, depth, rnd)
}

type panickingBouncer struct{}

func (panickingBouncer) Bounce(Ray, *HitInfo, float64, float64, BounceType, *rand.Rand) (Ray, bool, float64) {
	panic("boom")
}

func TestPanickingComponentRecovered(t *testing.T) {
	scene, _ := testSceneA()
	p := NewSampler(1, 1).Pipeline()
	p.Bouncer = panickingBouncer{}
	p.RecoverPanics = true
	// must not panic; the failed sub-path shades Black
	c := p.Sample(scene, Ray{V(0, 1, 5), V(0, -0.2, -1).Normalize()}, rand.New(rand.NewSource(1)))
	_ = c
}

func TestPanickingComponentPropagatesWithoutRecovery(t *testing.T) {
	scene, _ := testSceneA()
	p := NewSampler(1, 1).Pipeline()
	p.Bouncer = panickingBouncer{}
	defer func() {
		if recover() == nil {
			t.Error("expected panic to propagate when RecoverPanics is false")
		}
	}()
	p.Sample(scene, Ray{V(0, 1, 5), V(0, -0.2, -1).Normalize()}, rand.New(rand.NewSource(1)))
}

func TestZeroPipelineUsesDefaults(t *testing.T) {
	scene, _ := testSceneA()
	p := &Pipeline{FirstHitSamples: 1, MaxBounces: 1, DirectLighting: true}
	ray := Ray{V(0, 1, 5), V(0, -0.2, -1).Normalize()}
	a := p.Sample(scene, ray, rand.New(rand.NewSource(5)))
	b := p.Sample(scene, ray, rand.New(rand.NewSource(5)))
	if a != b {
		t.Errorf("zero pipeline not deterministic: %v vs %v", a, b)
	}
}

func TestSamplerFieldMutationPropagates(t *testing.T) {
	s := NewSampler(1, 1)
	s.LightMode = LightModeAll
	s.SoftShadows = false
	p := s.Pipeline()
	ls, ok := p.Lights.(DefaultLightSampler)
	if !ok {
		t.Fatalf("Pipeline().Lights is %T, want DefaultLightSampler", p.Lights)
	}
	if ls.LightMode != LightModeAll || ls.SoftShadows {
		t.Errorf("field mutation not reflected: %+v", ls)
	}
}

// TestConcurrentSampling hammers shared samplers, pipelines and components
// from many goroutines. Run with -race: the default components are
// stateless values and must not share mutable scratch state.
func TestConcurrentSampling(t *testing.T) {
	scene, cam := testSceneB()
	sampler := NewSampler(2, 2)
	sampler.LightMode = LightModeAll
	pipeline := sampler.Pipeline()
	const goroutines = 8
	const raysEach = 40
	results := make([][]Color, goroutines)
	var wg sync.WaitGroup
	for g := 0; g < goroutines; g++ {
		wg.Add(1)
		go func(g int) {
			defer wg.Done()
			rnd := rand.New(rand.NewSource(int64(g) + 1))
			out := make([]Color, 0, raysEach*2)
			for i := 0; i < raysEach; i++ {
				ray := cam.CastRay(i%8, i/8, 8, 8, 0.5, 0.5, rnd)
				out = append(out, sampler.Sample(scene, ray, rnd))
				out = append(out, pipeline.Sample(scene, ray, rnd))
			}
			results[g] = out
		}(g)
	}
	wg.Wait()
	// determinism check: rerunning goroutine g's stream sequentially must
	// reproduce its colors exactly (no cross-goroutine state leaked).
	for g := 0; g < goroutines; g++ {
		rnd := rand.New(rand.NewSource(int64(g) + 1))
		for i := 0; i < raysEach; i++ {
			ray := cam.CastRay(i%8, i/8, 8, 8, 0.5, 0.5, rnd)
			if got := sampler.Sample(scene, ray, rnd); got != results[g][2*i] {
				t.Fatalf("goroutine %d ray %d: concurrent result %v != sequential %v",
					g, i, results[g][2*i], got)
			}
			if got := pipeline.Sample(scene, ray, rnd); got != results[g][2*i+1] {
				t.Fatalf("goroutine %d ray %d (pipeline): concurrent %v != sequential %v",
					g, i, results[g][2*i+1], got)
			}
		}
	}
}

// TestRenderSmoke exercises the full Render path (its own goroutine pool)
// with both the facade and a hand-wired pipeline.
func TestRenderSmoke(t *testing.T) {
	scene, cam := testSceneA()
	sampler := NewSampler(1, 1)
	sampler.LightMode = LightModeAll
	im := Render(scene, &cam, sampler, 8, 8, 1)
	if im.Bounds().Dx() != 8 || im.Bounds().Dy() != 8 {
		t.Errorf("render size = %v, want 8x8", im.Bounds())
	}
	im2 := Render(scene, &cam, sampler.Pipeline(), 8, 8, 1)
	if im2.Bounds().Dx() != 8 {
		t.Errorf("pipeline render size = %v, want 8x8", im2.Bounds())
	}
}
