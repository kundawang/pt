package pt

import (
	"fmt"
	"math"
	"math/rand"
	"sync"
	"testing"
	"time"
)

// ---------------------------------------------------------------------------
// test fixtures
// ---------------------------------------------------------------------------

// linearIntersect is the reference brute-force query: the new tree must
// return the same nearest hit as scanning every shape in order.
func linearIntersect(shapes []Shape, r Ray) Hit {
	hit := NoHit
	for _, shape := range shapes {
		h := shape.Intersect(r)
		if h.T < hit.T {
			hit = h
		}
	}
	return hit
}

func sameHit(a, b Hit) bool {
	if a.Ok() != b.Ok() {
		return false
	}
	if !a.Ok() {
		return true
	}
	if a.Shape != b.Shape {
		return false
	}
	tol := 1e-9 * math.Max(1, math.Abs(a.T))
	return math.Abs(a.T-b.T) <= tol
}

func countingConfig(strategy SplitStrategy) TreeConfig {
	c := DefaultTreeConfig()
	c.Strategy = strategy
	c.CountCandidates = true
	return c
}

// randomSpheres: well separated primitives with jittered centers.
func randomSpheres(n int, rnd *rand.Rand) []Shape {
	shapes := make([]Shape, n)
	m := DiffuseMaterial(Black)
	for i := range shapes {
		c := Vector{rnd.Float64() * 100, rnd.Float64() * 100, rnd.Float64() * 100}
		shapes[i] = NewSphere(c, 0.5+rnd.Float64(), m)
	}
	return shapes
}

// overlappingSpheres: every primitive spans most of the scene, so median
// splits reference a large fraction of the shapes on both sides.
func overlappingSpheres(n int, rnd *rand.Rand) []Shape {
	shapes := make([]Shape, n)
	m := DiffuseMaterial(Black)
	for i := range shapes {
		c := Vector{rnd.Float64() * 4, rnd.Float64() * 4, rnd.Float64() * 4}
		shapes[i] = NewSphere(c, 18+rnd.Float64()*4, m)
	}
	return shapes
}

// mediumOverlapSpheres: boxes overlap only locally, so a good cost model can
// still separate most references while the median plane duplicates many.
func mediumOverlapSpheres(n int, rnd *rand.Rand) []Shape {
	shapes := make([]Shape, n)
	m := DiffuseMaterial(Black)
	for i := range shapes {
		c := Vector{rnd.Float64() * 40, rnd.Float64() * 40, rnd.Float64() * 40}
		shapes[i] = NewSphere(c, 2.5+rnd.Float64(), m)
	}
	return shapes
}

// longThinShapes: strongly anisotropic boxes stretched along one axis.
func longThinShapes(n int, rnd *rand.Rand) []Shape {
	shapes := make([]Shape, n)
	m := DiffuseMaterial(Black)
	for i := range shapes {
		cx := rnd.Float64() * 200
		cy := rnd.Float64() * 4
		cz := rnd.Float64() * 4
		shapes[i] = NewCube(
			Vector{cx - 40, cy - 0.2, cz - 0.2},
			Vector{cx + 40, cy + 0.2, cz + 0.2}, m)
	}
	return shapes
}

// flatCubes: all boxes are flat on Z, exercising the degenerate-axis path.
func flatCubes(n int, rnd *rand.Rand) []Shape {
	shapes := make([]Shape, n)
	m := DiffuseMaterial(Black)
	for i := range shapes {
		x, y := rnd.Float64()*50, rnd.Float64()*50
		shapes[i] = NewCube(
			Vector{x, y, 1},
			Vector{x + 1.5, y + 1.5, 1}, m)
	}
	return shapes
}

func extremeShapes(rnd *rand.Rand) []Shape {
	m := DiffuseMaterial(Black)
	return []Shape{
		NewSphere(Vector{1e8, 0, 0}, 1e7, m),
		NewSphere(Vector{-1e8, 0, 0}, 1e7, m),
		NewCube(Vector{0, -1e7, -1e7}, Vector{1e6, 1e7, 1e7}, m),
		NewSphere(Vector{0, 0, 0}, 1e-6, m),
	}
}

// testRays builds a deterministic mix of guaranteed-facing and random rays
// for a scene's bounding box.
func testRays(box Box, n int, rnd *rand.Rand) []Ray {
	rays := make([]Ray, 0, n)
	center := box.Center()
	for i := 0; i < n/2; i++ {
		// Aim from outside the bounds at a random interior target.
		target := box.Anchor(Vector{
			rnd.Float64(), rnd.Float64(), rnd.Float64()})
		origin := center.Add(target.Sub(center).MulScalar(3))
		rays = append(rays, Ray{origin, target.Sub(origin).Normalize()})
	}
	for len(rays) < n {
		origin := box.Anchor(Vector{
			rnd.Float64() * 2, rnd.Float64() * 2, rnd.Float64() * 2})
		rays = append(rays, Ray{origin, RandomUnitVector(rnd)})
	}
	return rays
}

// ---------------------------------------------------------------------------
// correctness: SAH and median trees must match linear scan exactly
// ---------------------------------------------------------------------------

func TestTreeHitsMatchLinear(t *testing.T) {
	rnd := rand.New(rand.NewSource(42))
	scenes := map[string][]Shape{
		"random":        randomSpheres(200, rnd),
		"mediumOverlap": mediumOverlapSpheres(200, rnd),
		"overlapping":   overlappingSpheres(200, rnd),
		"longThin":      longThinShapes(200, rnd),
		"flat":          flatCubes(200, rnd),
	}
	for name, shapes := range scenes {
		box := BoxForShapes(shapes)
		rays := testRays(box, 400, rand.New(rand.NewSource(7)))
		for _, strategy := range []SplitStrategy{SplitSAH, SplitMedian} {
			tree := NewTreeWithConfig(shapes, countingConfig(strategy))
			for i, r := range rays {
				got := tree.Intersect(r)
				want := linearIntersect(shapes, r)
				if !sameHit(got, want) {
					t.Fatalf("%s/%s ray %d: got {ok=%v shape=%p t=%v} want {ok=%v shape=%p t=%v}",
						name, strategyName(strategy), i,
						got.Ok(), got.Shape, got.T,
						want.Ok(), want.Shape, want.T)
				}
			}
		}
	}
}

func strategyName(s SplitStrategy) string {
	if s == SplitSAH {
		return "SAH"
	}
	return "median"
}

// ---------------------------------------------------------------------------
// degenerate / edge-case inputs
// ---------------------------------------------------------------------------

func TestTreeEmptyScene(t *testing.T) {
	tree := NewTreeWithConfig(nil, countingConfig(SplitSAH))
	st := tree.Stats()
	if st.Nodes != 1 || st.LeafNodes != 1 || st.ShapeRefs != 0 {
		t.Fatalf("empty scene stats: %+v", st)
	}
	h := tree.Intersect(Ray{Vector{}, Vector{1, 0, 0}})
	if h.Ok() {
		t.Fatal("empty tree reported a hit")
	}
}

func TestTreeSingleShape(t *testing.T) {
	s := NewSphere(Vector{}, 1, DiffuseMaterial(Black))
	for _, strategy := range []SplitStrategy{SplitSAH, SplitMedian} {
		tree := NewTreeWithConfig([]Shape{s}, countingConfig(strategy))
		st := tree.Stats()
		if st.LeafNodes != 1 || st.InteriorNodes != 0 || st.ShapeRefs != 1 {
			t.Fatalf("single shape stats (%s): %+v", strategyName(strategy), st)
		}
		r := Ray{Vector{0, 0, 5}, Vector{0, 0, -1}}
		h := tree.Intersect(r)
		if h.Shape != s || !sameHit(h, s.Intersect(r)) {
			t.Fatalf("single shape hit wrong: %+v", h)
		}
	}
}

func TestTreeDegenerateAndExtreme(t *testing.T) {
	rnd := rand.New(rand.NewSource(99))
	cases := map[string][]Shape{
		"flat":    flatCubes(100, rnd),
		"extreme": extremeShapes(rnd),
		"coincident": func() []Shape {
			m := DiffuseMaterial(Black)
			out := make([]Shape, 32)
			for i := range out {
				out[i] = NewSphere(Vector{1, 2, 3}, 0.5, m)
			}
			return out
		}(),
	}
	for name, shapes := range cases {
		for _, strategy := range []SplitStrategy{SplitSAH, SplitMedian} {
			tree := NewTreeWithConfig(shapes, countingConfig(strategy))
			if tree.Root == nil {
				t.Fatalf("%s/%s: nil root", name, strategyName(strategy))
			}
			rays := testRays(BoxForShapes(shapes), 200, rand.New(rand.NewSource(11)))
			for _, r := range rays {
				if got, want := tree.Intersect(r), linearIntersect(shapes, r); !sameHit(got, want) {
					t.Fatalf("%s/%s hit mismatch: got %v want %v",
						name, strategyName(strategy), got.T, want.T)
				}
			}
			st := tree.Stats()
			if st.Nodes == 0 || st.MaxDepth < 0 {
				t.Fatalf("%s/%s bad stats: %+v", name, strategyName(strategy), st)
			}
		}
	}
}

// unboundedShape reports a genuinely infinite bounding box to exercise the
// builder's non-finite-bounds fallback. Its intersection delegates to an
// embedded sphere so nearest-hit semantics stay well defined.
type unboundedShape struct {
	Shape
}

func (u *unboundedShape) BoundingBox() Box {
	inf := math.Inf(1)
	return Box{Vector{-inf, -inf, -inf}, Vector{inf, inf, inf}}
}

func TestTreeInfinitePrimitiveFallback(t *testing.T) {
	inner := NewSphere(Vector{0, 5, 0}, 1, DiffuseMaterial(Black))
	shapes := []Shape{
		&unboundedShape{inner},
		NewSphere(Vector{10, 5, 0}, 1, DiffuseMaterial(Black)),
		NewSphere(Vector{-10, 5, 0}, 1, DiffuseMaterial(Black)),
		NewSphere(Vector{0, 5, 10}, 1, DiffuseMaterial(Black)),
		NewSphere(Vector{0, 5, -10}, 1, DiffuseMaterial(Black)),
	}
	tree := NewTreeWithConfig(shapes, countingConfig(SplitSAH))
	if tree.Box.IsFinite() {
		t.Fatal("scene with an infinite-bounds shape must report non-finite bounds")
	}
	st := tree.Stats()
	// The non-finite root cannot host a split plane: one leaf, no interior.
	if st.InteriorNodes != 0 || st.LeafNodes != 1 || st.ShapeRefs != len(shapes) {
		t.Fatalf("infinite-primitive stats: %+v", st)
	}
	if st.MedianSplits != 0 {
		t.Fatalf("non-finite root must not attempt splits: %+v", st)
	}
	rays := testRays(BoxForShapes(shapes[1:]), 100, rand.New(rand.NewSource(31)))
	for _, r := range rays {
		if got, want := tree.Intersect(r), linearIntersect(shapes, r); !sameHit(got, want) {
			t.Fatalf("infinite-primitive hit mismatch: got %v want %v", got.T, want.T)
		}
	}

	// Plane uses the library's finite 1e9 sentinel; verify it builds and
	// intersects identically to a linear scan regardless.
	planeScene := []Shape{
		NewPlane(Vector{0, 0, 0}, Vector{0, 1, 0}, DiffuseMaterial(Black)),
		inner,
	}
	ptree := NewTreeWithConfig(planeScene, countingConfig(SplitSAH))
	r := Ray{Vector{0, 10, 0}, Vector{0, -1, 0}}
	if !sameHit(ptree.Intersect(r), linearIntersect(planeScene, r)) {
		t.Fatal("plane hit mismatch")
	}
}

func TestTreeDepthAndCapacityCaps(t *testing.T) {
	rnd := rand.New(rand.NewSource(5))
	shapes := randomSpheres(500, rnd)
	cfg := countingConfig(SplitSAH)
	cfg.MaxDepth = 3
	cfg.LeafCapacity = 2
	tree := NewTreeWithConfig(shapes, cfg)
	st := tree.Stats()
	if st.MaxDepth > 3 {
		t.Fatalf("depth cap violated: maxDepth=%d", st.MaxDepth)
	}
	if len(st.LeafShapes) == 0 {
		t.Fatal("no leaves recorded")
	}
	// No empty leaves are ever produced; interior nodes never exceed the
	// depth cap. Undersized leaves above the cap are legal: they result from
	// the no-progress termination (a plane that isolates no references).
	var walk func(n *Node, depth int)
	walk = func(n *Node, depth int) {
		if n.Axis == AxisNone {
			if len(n.Shapes) == 0 {
				t.Fatalf("empty leaf at depth %d", depth)
			}
			return
		}
		if depth >= cfg.MaxDepth {
			t.Fatalf("interior node at depth %d >= cap %d", depth, cfg.MaxDepth)
		}
		walk(n.Left, depth+1)
		walk(n.Right, depth+1)
	}
	walk(tree.Root, 0)
}

func TestTreeMalformedConfigFallbacks(t *testing.T) {
	rnd := rand.New(rand.NewSource(17))
	shapes := randomSpheres(200, rnd)
	box := BoxForShapes(shapes)
	rays := testRays(box, 100, rand.New(rand.NewSource(18)))
	// NaN/Inf and non-positive tuning values must be replaced by defaults;
	// the resulting tree must still match a linear scan.
	cfg := TreeConfig{
		Strategy:        SplitSAH,
		LeafCapacity:    -3,
		MaxDepth:        -1,
		SAHBins:         0,
		SAHTraversal:    math.NaN(),
		SAHEmptyBonus:   math.Inf(1),
		CountCandidates: true,
	}
	tree := NewTreeWithConfig(shapes, cfg)
	if tree.config.SAHBins != DefaultTreeConfig().SAHBins ||
		tree.config.LeafCapacity != DefaultTreeConfig().LeafCapacity ||
		tree.config.SAHTraversal != DefaultTreeConfig().SAHTraversal ||
		tree.config.SAHEmptyBonus != 1 {
		t.Fatalf("defaults not applied: %+v", tree.config)
	}
	for _, r := range rays {
		if got, want := tree.Intersect(r), linearIntersect(shapes, r); !sameHit(got, want) {
			t.Fatalf("fallback-config hit mismatch: got %v want %v", got.T, want.T)
		}
	}
	// Zero-value config (no options set at all) must build as default SAH.
	plain := NewTreeWithConfig(shapes, TreeConfig{})
	if plain.Stats().Nodes == 0 {
		t.Fatal("zero-value config failed to build")
	}
}

// ---------------------------------------------------------------------------
// statistics: node counts, candidate counts, box areas
// ---------------------------------------------------------------------------

func TestTreeStatsInvariants(t *testing.T) {
	rnd := rand.New(rand.NewSource(13))
	scenes := map[string][]Shape{
		"random":        randomSpheres(300, rnd),
		"mediumOverlap": mediumOverlapSpheres(300, rnd),
		"overlapping":   overlappingSpheres(300, rnd),
		"longThin":      longThinShapes(300, rnd),
	}
	for name, shapes := range scenes {
		tree := NewTreeWithConfig(shapes, countingConfig(SplitSAH))
		st := tree.Stats()
		if st.Nodes != st.InteriorNodes+st.LeafNodes {
			t.Fatalf("%s: node count mismatch %+v", name, st)
		}
		if st.ShapeRefs < len(shapes) || st.DuplicateRefs != st.ShapeRefs-len(shapes) {
			t.Fatalf("%s: ref accounting wrong: %+v", name, st)
		}
		if st.RootBoxArea < 0 || st.LeafBoxArea < 0 || st.TotalBoxArea < 0 {
			t.Fatalf("%s: negative box area: %+v", name, st)
		}
		if st.SAHSplits+st.MedianSplits != st.InteriorNodes {
			t.Fatalf("%s: split accounting: sah=%d median=%d interior=%d",
				name, st.SAHSplits, st.MedianSplits, st.InteriorNodes)
		}
		if st.BuildDuration <= 0 {
			t.Fatalf("%s: build duration not recorded", name)
		}
		// TreeStats.String must render without panicking and mention nodes.
		if s := st.String(); len(s) == 0 {
			t.Fatalf("%s: empty stats string", name)
		}
	}
}

// ---------------------------------------------------------------------------
// candidate reduction: SAH must beat linear scan, and overlap behavior
// compared against the median builder
// ---------------------------------------------------------------------------

type benchResult struct {
	build      time.Duration
	candidates uint64
	hits       int
	refs       int
}

func runBenchmark(t *testing.T, name string, shapes []Shape, strategy SplitStrategy, rays []Ray) benchResult {
	cfg := countingConfig(strategy)
	start := time.Now()
	tree := NewTreeWithConfig(shapes, cfg)
	build := time.Since(start)
	tree.ResetCandidates()
	hits := 0
	for _, r := range rays {
		h := tree.Intersect(r)
		if h.Ok() {
			hits++
		}
	}
	st := tree.Stats()
	t.Logf("%-12s %-7s build=%10v nodes=%5d refs=%6d dup=%6d candidates=%9d hits=%d | %s",
		name, strategyName(strategy), build.Round(time.Microsecond),
		st.Nodes, st.ShapeRefs, st.DuplicateRefs, tree.Candidates(), hits, st)
	return benchResult{build, tree.Candidates(), hits, st.ShapeRefs}
}

func TestTreeCandidateReduction(t *testing.T) {
	rnd := rand.New(rand.NewSource(21))
	scenes := map[string]struct {
		shapes []Shape
		// maxSAHVsMedian bounds the SAH/median candidate ratio. On
		// hit-heavy ray sets a tree that splits less can test slightly more
		// candidates per ray while storing far fewer duplicate references and
		// building faster; a 1.5x bound keeps that tradeoff honest.
		maxSAHVsMedian float64
		// maxSAHDuplication caps ShapeRefs relative to the input count.
		maxSAHDuplication float64
	}{
		"random":        {randomSpheres(1000, rnd), 1.1, 2.0},
		"mediumOverlap": {mediumOverlapSpheres(1000, rnd), 1.5, 8.0},
		"overlapping":   {overlappingSpheres(1000, rnd), 1.1, 1.01},
		"longThin":      {longThinShapes(1000, rnd), 1.5, 15.0},
	}
	for name, sc := range scenes {
		shapes := sc.shapes
		box := BoxForShapes(shapes)
		rays := testRays(box, 2000, rand.New(rand.NewSource(22)))
		sah := runBenchmark(t, name, shapes, SplitSAH, rays)
		med := runBenchmark(t, name, shapes, SplitMedian, rays)
		if sah.hits != med.hits {
			t.Fatalf("%s: hit count differs SAH=%d median=%d", name, sah.hits, med.hits)
		}
		linear := uint64(len(shapes) * len(rays))
		if sah.candidates >= linear {
			t.Fatalf("%s: SAH candidates %d not below linear %d",
				name, sah.candidates, linear)
		}
		ratio := float64(sah.candidates) / float64(med.candidates)
		if ratio > sc.maxSAHVsMedian {
			t.Fatalf("%s: SAH/median candidate ratio %.3f exceeds %.2f",
				name, ratio, sc.maxSAHVsMedian)
		}
		dupRatio := float64(sah.refs) / float64(len(shapes))
		if dupRatio > sc.maxSAHDuplication {
			t.Fatalf("%s: SAH reference duplication %.2fx exceeds %.2fx (refs=%d)",
				name, dupRatio, sc.maxSAHDuplication, sah.refs)
		}
		// The SAH tree must never take more than ~2x the median build time
		// (binned evaluation is O(n) per node; median sorts are O(n log n)).
		if sah.build > 2*med.build+time.Millisecond {
			t.Fatalf("%s: SAH build %s much slower than median %s",
				name, sah.build, med.build)
		}
	}
}

func BenchmarkTreeBuild(b *testing.B) {
	rnd := rand.New(rand.NewSource(1))
	for _, n := range []int{100, 1000, 10000} {
		shapes := randomSpheres(n, rnd)
		b.Run(fmt.Sprintf("SAH/%d", n), func(b *testing.B) {
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				NewTreeWithConfig(shapes, DefaultTreeConfig())
			}
		})
		b.Run(fmt.Sprintf("median/%d", n), func(b *testing.B) {
			cfg := DefaultTreeConfig()
			cfg.Strategy = SplitMedian
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				NewTreeWithConfig(shapes, cfg)
			}
		})
	}
}

// ---------------------------------------------------------------------------
// Scene: Compile reuse, explicit invalidation, and lock-free reads
// ---------------------------------------------------------------------------

func TestSceneCompileReuseAndInvalidate(t *testing.T) {
	scene := Scene{}
	rnd := rand.New(rand.NewSource(3))
	for _, s := range randomSpheres(50, rnd) {
		scene.Add(s)
	}
	scene.TreeConfig = countingConfig(SplitSAH)
	scene.Compile()
	first := scene.TreeStats()
	if first.Nodes == 0 {
		t.Fatal("tree not built")
	}
	// Repeated Compile must reuse the same index.
	scene.Compile()
	scene.Compile()
	if got := scene.TreeStats(); got.Nodes != first.Nodes || got.ShapeRefs != first.ShapeRefs {
		t.Fatalf("repeated Compile rebuilt the tree: %+v", got)
	}
	// Add invalidates; the next Compile rebuilds covering all shapes.
	scene.Add(NewSphere(Vector{200, 200, 200}, 1, DiffuseMaterial(Black)))
	scene.Compile()
	if got := scene.TreeStats(); got.ShapeRefs < len(scene.Shapes) {
		t.Fatalf("invalidate/rebuild lost shapes: refs=%d want>=%d",
			got.ShapeRefs, len(scene.Shapes))
	}
	// Explicit invalidation forces a rebuild too.
	scene.Invalidate()
	scene.Compile()
	if scene.TreeStats().Nodes == 0 {
		t.Fatal("tree missing after Invalidate + Compile")
	}
}

func TestSceneCandidateCounter(t *testing.T) {
	scene := Scene{}
	rnd := rand.New(rand.NewSource(4))
	shapes := randomSpheres(100, rnd)
	for _, s := range shapes {
		scene.Add(s)
	}
	scene.TreeConfig = countingConfig(SplitSAH)
	scene.Compile()
	rays := testRays(BoxForShapes(shapes), 100, rand.New(rand.NewSource(5)))
	scene.ResetTreeCandidates()
	var total uint64
	for _, r := range rays {
		scene.Intersect(r)
		total++
	}
	if c := scene.TreeCandidates(); c == 0 || c > uint64(len(shapes))*total {
		t.Fatalf("candidate counter out of range: %d (rays=%d shapes=%d)",
			c, total, len(shapes))
	}
	scene.ResetTreeCandidates()
	if scene.TreeCandidates() != 0 {
		t.Fatal("candidate reset failed")
	}
}

func TestSceneConcurrentReadOnly(t *testing.T) {
	scene := Scene{}
	rnd := rand.New(rand.NewSource(6))
	shapes := overlappingSpheres(500, rnd)
	for _, s := range shapes {
		scene.Add(s)
	}
	scene.TreeConfig = countingConfig(SplitSAH)
	scene.Compile()
	rays := testRays(BoxForShapes(shapes), 300, rand.New(rand.NewSource(7)))
	want := make([]Hit, len(rays))
	for i, r := range rays {
		want[i] = linearIntersect(shapes, r)
	}

	var wg sync.WaitGroup
	for g := 0; g < 8; g++ {
		wg.Add(1)
		go func(seed int64) {
			defer wg.Done()
			local := rand.New(rand.NewSource(seed))
			for k := 0; k < 2000; k++ {
				r := rays[local.Intn(len(rays))]
				got := scene.Intersect(r)
				_ = got
			}
		}(int64(g + 100))
	}
	wg.Wait()
	// Deterministic re-check after the concurrent phase.
	for i, r := range rays {
		if !sameHit(scene.Intersect(r), want[i]) {
			t.Fatalf("concurrent reads corrupted results at ray %d", i)
		}
	}
}

// ---------------------------------------------------------------------------
// Mesh integration: the shared builder must also work for triangle trees
// ---------------------------------------------------------------------------

func TestMeshTreeIntegration(t *testing.T) {
	rnd := rand.New(rand.NewSource(8))
	tris := make([]*Triangle, 0, 120)
	m := DiffuseMaterial(Black)
	for i := 0; i < 120; i++ {
		x := rnd.Float64() * 20
		y := rnd.Float64() * 20
		tris = append(tris, NewTriangle(
			Vector{x, y, 0}, Vector{x + 1, y, 0}, Vector{x, y + 1, 0},
			Vector{}, Vector{}, Vector{}, m))
	}
	mesh := NewMesh(tris)
	mesh.Compile()
	// Compile reuse.
	statsBefore := mesh.tree.Stats()
	mesh.Compile()
	if mesh.tree.Stats().Nodes != statsBefore.Nodes {
		t.Fatal("Mesh.Compile rebuilt an existing tree")
	}
	rays := testRays(mesh.BoundingBox(), 200, rand.New(rand.NewSource(9)))
	shapes := make([]Shape, len(tris))
	for i, tri := range tris {
		shapes[i] = tri
	}
	for _, r := range rays {
		if got, want := mesh.Intersect(r), linearIntersect(shapes, r); !sameHit(got, want) {
			t.Fatalf("mesh hit mismatch: got %v want %v", got.T, want.T)
		}
	}
}
